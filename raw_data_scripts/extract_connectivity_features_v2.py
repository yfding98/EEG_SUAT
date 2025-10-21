#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_connectivity_features_v2.py

改进版连接性特征提取：
1. 禁止拼接片段 - 只在连续数据上提取特征
2. 5秒窗口（可配置）
3. 自动检测并跳过跨片段边界的窗口
4. 个体归一化
5. 可选可视化

使用方法:
    python extract_connectivity_features_v2.py --input_file "path/to/file.set" --window_size 5
    python extract_connectivity_features_v2.py --input_dir "path/" --no_visualize
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import signal, stats
from scipy.signal import hilbert, butter, filtfilt, detrend
from scipy.stats import spearmanr, zscore
from sklearn.covariance import GraphicalLassoCV
import networkx as nx

# 导入加速版本的有向连接计算
try:
    from connectivity_acceleration_numba import (
        compute_granger_causality_pairwise_threaded,
        compute_transfer_entropy_pairwise_numba
    )
    HAS_ACCELERATION = True
except:
    HAS_ACCELERATION = False
    print("Warning: Acceleration modules not found, directed connectivity will be slower")

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

plt.style.use('default')
sns.set_palette("husl")


# ============================================================================
# 配置参数
# ============================================================================
class Config:
    """全局配置"""
    # 时间窗口参数
    WINDOW_SIZE = 5  # 改为5秒
    OVERLAP = 0  # 不重叠
    MIN_SEGMENT_DURATION = 5  # 最小片段长度（秒）
    
    # 频段定义
    FREQ_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    # 图网络阈值
    SPARSITY_THRESHOLD = 0.2
    
    # 可视化选项
    ENABLE_VISUALIZATION = True  # 是否生成可视化


# ============================================================================
# 数据预处理和归一化
# ============================================================================
def preprocess_segment_data(data, sfreq):
    """
    预处理片段数据：归一化 + 去趋势 + 滤波
    
    目的：
    1. 消除个体差异（Z-score归一化）
    2. 去除低频漂移（去趋势）
    3. 去除工频干扰和高频噪声（带通滤波）
    
    参数:
        data: (n_channels, n_samples) 原始数据
        sfreq: 采样率
    
    返回:
        processed_data: (n_channels, n_samples) 预处理后的数据
    """
    # 1. Z-score归一化每个通道（消除个体幅度差异）
    data_normalized = zscore(data, axis=1)
    
    # 处理全零通道
    data_normalized = np.nan_to_num(data_normalized, 0)
    
    # 2. 去趋势（去除低频漂移）
    data_detrended = detrend(data_normalized, axis=1, type='linear')
    
    # 3. 带通滤波（0.5-50Hz，去除极低频和高频噪声）
    try:
        data_filtered = bandpass_filter(data_detrended, sfreq, 0.5, min(50, sfreq/2.5))
    except:
        data_filtered = data_detrended
    
    return data_filtered


def detect_continuous_segments(raw, annotations=None):
    """
    检测数据中的连续片段（无拼接）
    
    参数:
        raw: MNE Raw对象
        annotations: MNE Annotations对象（如果有）
    
    返回:
        continuous_segments: list of (start_sample, end_sample, duration)
    """
    sfreq = raw.info['sfreq']
    n_samples = len(raw.times)
    
    # 方法1: 如果有annotations标记边界
    if annotations is not None and len(annotations) > 0:
        segments = []
        # 假设annotations标记了bad_segments或边界
        # 这里需要根据实际情况调整
        print("  Using annotations to detect segments")
        # TODO: 实现基于annotations的片段检测
    
    # 方法2: 检测数据的突变（拼接处通常有幅度突变）
    # 计算相邻样本的差异
    data = raw.get_data()
    
    # 计算所有通道的RMS
    rms = np.sqrt(np.mean(data**2, axis=0))
    
    # 计算RMS的变化率
    rms_diff = np.abs(np.diff(rms))
    rms_diff_normalized = rms_diff / (np.median(rms_diff) + 1e-10)
    
    # 检测突变点（变化率超过阈值）
    threshold = 10.0  # 可调整
    change_points = np.where(rms_diff_normalized > threshold)[0]
    
    if len(change_points) == 0:
        # 没有检测到拼接，整个数据是连续的
        print(f"  No concatenation detected, treating as single continuous segment")
        return [(0, n_samples, n_samples / sfreq)]
    
    # 有拼接点，分割成多个连续片段
    print(f"  Detected {len(change_points)} potential concatenation points")
    
    segments = []
    start = 0
    
    for cp in change_points:
        if cp - start > sfreq * Config.MIN_SEGMENT_DURATION:  # 至少5秒
            segments.append((start, cp, (cp - start) / sfreq))
        start = cp + 1
    
    # 最后一段
    if n_samples - start > sfreq * Config.MIN_SEGMENT_DURATION:
        segments.append((start, n_samples, (n_samples - start) / sfreq))
    
    print(f"  Found {len(segments)} continuous segments")
    for i, (s, e, d) in enumerate(segments):
        print(f"    Segment {i+1}: {d:.1f}s ({s} - {e} samples)")
    
    return segments


def segment_continuous_data(raw, window_size=5, overlap=0):
    """
    将连续数据切割成固定窗口，避免跨边界
    
    参数:
        raw: MNE Raw对象
        window_size: 窗口大小（秒）
        overlap: 重叠大小（秒）
    
    返回:
        windows: list of dict with 'data', 'start_time', 'end_time', 'segment_id'
    """
    sfreq = raw.info['sfreq']
    n_channels = len(raw.ch_names)
    
    # 检测连续片段
    continuous_segments = detect_continuous_segments(raw)
    
    windows = []
    window_samples = int(window_size * sfreq)
    step_samples = int((window_size - overlap) * sfreq)
    
    # 在每个连续片段内切割窗口
    for seg_id, (seg_start, seg_end, seg_duration) in enumerate(continuous_segments):
        print(f"\n  Processing continuous segment {seg_id+1}/{len(continuous_segments)} ({seg_duration:.1f}s)")
        
        start = seg_start
        seg_window_count = 0
        
        while start + window_samples <= seg_end:
            end = start + window_samples
            
            # 提取数据
            data = raw.get_data(start=start, stop=end)
            
            # 预处理：归一化 + 去趋势 + 滤波
            data_processed = preprocess_segment_data(data, sfreq)
            
            windows.append({
                'data': data_processed,
                'start_time': start / sfreq,
                'end_time': end / sfreq,
                'n_channels': n_channels,
                'sfreq': sfreq,
                'segment_id': seg_id,
                'is_concatenated': False  # 标记为非拼接
            })
            
            start += step_samples
            seg_window_count += 1
        
        print(f"    Extracted {seg_window_count} valid windows from this segment")
    
    return windows


def bandpass_filter(data, sfreq, lowcut, highcut, order=4):
    """带通滤波"""
    nyq = 0.5 * sfreq
    low = lowcut / nyq
    high = highcut / nyq
    
    # 确保频率在有效范围内
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)


# ============================================================================
# 连接性指标计算（安全的、对拼接不敏感的）
# ============================================================================
def compute_pearson_correlation(data):
    """Pearson相关 - 对拼接不敏感✅"""
    corr_matrix = np.corrcoef(data)
    return corr_matrix


def compute_partial_correlation(data):
    """偏相关 - 对拼接不敏感✅"""
    n_channels = data.shape[0]
    
    try:
        model = GraphicalLassoCV(cv=3, max_iter=100)
        model.fit(data.T)
        precision = model.precision_
        
        partial_corr = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    partial_corr[i, j] = -precision[i, j] / np.sqrt(precision[i, i] * precision[j, j])
        
        np.fill_diagonal(partial_corr, 1.0)
        
    except:
        corr = np.corrcoef(data)
        try:
            inv_corr = np.linalg.pinv(corr)
            partial_corr = np.zeros((n_channels, n_channels))
            for i in range(n_channels):
                for j in range(n_channels):
                    if i != j:
                        partial_corr[i, j] = -inv_corr[i, j] / np.sqrt(inv_corr[i, i] * inv_corr[j, j])
            np.fill_diagonal(partial_corr, 1.0)
        except:
            partial_corr = corr.copy()
    
    return partial_corr


def compute_spearman_correlation(data):
    """Spearman秩相关 - 对拼接不敏感✅"""
    n_channels = data.shape[0]
    spearman_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            rho, _ = spearmanr(data[i], data[j])
            spearman_matrix[i, j] = rho
            spearman_matrix[j, i] = rho
    
    return spearman_matrix


def compute_coherence(data, sfreq, fmin, fmax):
    """
    相干性 - 部分敏感⚠️
    5秒窗口应该还可以接受
    """
    n_channels = data.shape[0]
    coherence_matrix = np.zeros((n_channels, n_channels))
    
    nperseg = min(256, data.shape[1]//4)
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            try:
                f, Cxy = signal.coherence(data[i], data[j], sfreq, nperseg=nperseg)
                freq_mask = (f >= fmin) & (f <= fmax)
                mean_coh = np.mean(Cxy[freq_mask]) if np.any(freq_mask) else 0
                coherence_matrix[i, j] = mean_coh
                coherence_matrix[j, i] = mean_coh
            except:
                coherence_matrix[i, j] = 0
                coherence_matrix[j, i] = 0
    
    np.fill_diagonal(coherence_matrix, 1.0)
    return coherence_matrix


def compute_amplitude_envelope_correlation(data, sfreq, fmin, fmax):
    """
    振幅包络相关 - 对拼接不太敏感✅
    因为使用相关性而不是相位
    """
    try:
        filtered_data = bandpass_filter(data, sfreq, fmin, fmax)
        analytic_signal = hilbert(filtered_data, axis=1)
        envelope = np.abs(analytic_signal)
        
        # 低通滤波包络
        if sfreq > 2:
            envelope = bandpass_filter(envelope, sfreq, 0.1, min(0.5, sfreq/4))
        
        aec_matrix = np.corrcoef(envelope)
        return aec_matrix
    except:
        return np.eye(data.shape[0])


# ============================================================================
# 特征提取主函数
# ============================================================================
def extract_safe_features(window, config=Config()):
    """
    提取对拼接不敏感的安全特征
    
    优先使用：
    1. 相关性特征（Pearson, Spearman, Partial）✅✅✅
    2. 包络相关（AEC）✅
    3. 相干性（Coherence）⚠️（5秒窗口应该可以）
    
    避免使用：
    - 相位同步（PLV, PLI, wPLI）❌
    - Granger因果性❌
    - 传递熵❌
    """
    data = window['data']
    sfreq = window['sfreq']
    start_time = window['start_time']
    end_time = window['end_time']
    n_channels = window['n_channels']
    segment_id = window['segment_id']
    
    features = {
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time,
        'n_channels': n_channels,
        'segment_id': segment_id,
        'is_concatenated': window['is_concatenated']
    }
    
    # 1. 线性相关（最安全）✅✅✅
    features['pearson'] = compute_pearson_correlation(data)
    features['partial_corr'] = compute_partial_correlation(data)
    
    # 2. 非参数相关（安全）✅✅✅
    features['spearman'] = compute_spearman_correlation(data)
    
    # 3. 频域相干（对每个频段）⚠️
    for band_name, (fmin, fmax) in config.FREQ_BANDS.items():
        try:
            features[f'coherence_{band_name}'] = compute_coherence(data, sfreq, fmin, fmax)
        except:
            features[f'coherence_{band_name}'] = np.eye(n_channels)
    
    # 4. 包络耦合（对低频段）✅
    for band_name, (fmin, fmax) in config.FREQ_BANDS.items():
        if band_name in ['delta', 'theta', 'alpha']:
            try:
                features[f'aec_{band_name}'] = compute_amplitude_envelope_correlation(data, sfreq, fmin, fmax)
            except:
                features[f'aec_{band_name}'] = np.eye(n_channels)
    
    return features


# ============================================================================
# 可视化
# ============================================================================
def visualize_connectivity_matrices(features, channel_names, output_dir, window_id):
    """
    可视化连接矩阵（简化版）
    """
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    n_channels = len(channel_names)
    
    # 只可视化关键矩阵
    key_matrices = {
        'pearson': {'title': 'Pearson Correlation', 'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 1},
        'spearman': {'title': 'Spearman Correlation', 'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 1},
        'coherence_alpha': {'title': 'Coherence (Alpha)', 'cmap': 'viridis', 'vmin': 0, 'vmax': 1},
        'aec_alpha': {'title': 'AEC (Alpha)', 'cmap': 'coolwarm', 'vmin': -1, 'vmax': 1},
    }
    
    available = [k for k in key_matrices if k in features and isinstance(features[k], np.ndarray)]
    
    if len(available) < 2:
        return []
    
    # 创建2x2对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, key in enumerate(available[:4]):
        config = key_matrices[key]
        matrix = features[key]
        
        sns.heatmap(matrix, 
                   cmap=config['cmap'],
                   vmin=config['vmin'], 
                   vmax=config['vmax'],
                   xticklabels=channel_names,
                   yticklabels=channel_names,
                   square=True,
                   ax=axes[idx])
        
        axes[idx].set_title(config['title'], fontsize=12, fontweight='bold')
    
    # 隐藏多余子图
    for idx in range(len(available), 4):
        axes[idx].axis('off')
    
    plt.suptitle(f'Connectivity Matrices - Window {window_id}\n'
                 f'Time: {features["start_time"]:.1f}s - {features["end_time"]:.1f}s',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'window_{window_id:04d}_connectivity.png'
    filepath = os.path.join(viz_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return [filepath]


# ============================================================================
# 文件处理
# ============================================================================
def process_single_file(input_file, output_dir=None, config=Config()):
    """处理单个.set文件"""
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(input_file)}")
    print(f"{'='*80}")
    
    # 读取数据
    try:
        raw = mne.io.read_raw_eeglab(input_file, preload=True, verbose='ERROR')
    except Exception as e:
        print(f"ERROR: Failed to load file: {e}")
        return []
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_file), 
                                  os.path.basename(input_file).replace('.set', '_connectivity_v2'))
    os.makedirs(output_dir, exist_ok=True)
    
    # 分割数据（检测连续性）
    print(f"\n  Segmenting into {config.WINDOW_SIZE}s windows (no concatenation)...")
    windows = segment_continuous_data(raw, window_size=config.WINDOW_SIZE, overlap=config.OVERLAP)
    print(f"  Found {len(windows)} valid non-concatenated windows")
    
    if len(windows) == 0:
        print("  WARNING: No valid windows found!")
        return []
    
    # 提取特征
    all_features = []
    for idx, window in enumerate(windows):
        if idx % 10 == 0:
            print(f"  Processing window {idx+1}/{len(windows)}...")
        
        features = extract_safe_features(window, config)
        features['window_id'] = idx
        features['file_name'] = os.path.basename(input_file)
        all_features.append(features)
        
        # 可视化（可选）
        if config.ENABLE_VISUALIZATION and idx < 5:  # 只可视化前5个
            try:
                visualize_connectivity_matrices(features, raw.ch_names, output_dir, idx)
            except Exception as e:
                print(f"    Warning: Visualization failed: {e}")
    
    # 保存结果
    print(f"\n  Saving results...")
    output_files = save_features(all_features, output_dir, raw.ch_names)
    
    print(f"\n✓ Completed: {input_file}")
    print(f"  Output: {output_dir}")
    print(f"  Saved {len(output_files)} files")
    print(f"  Extracted {len(all_features)} windows from {len(set([f['segment_id'] for f in all_features]))} continuous segments")
    
    return output_files


def save_features(all_features, output_dir, channel_names):
    """保存特征"""
    output_files = []
    
    # 1. 保存元数据到CSV
    metadata = []
    for features in all_features:
        metadata.append({
            'window_id': features['window_id'],
            'segment_id': features['segment_id'],
            'start_time': features['start_time'],
            'end_time': features['end_time'],
            'duration': features['duration'],
            'n_channels': features['n_channels'],
            'is_concatenated': features['is_concatenated'],
            'file_name': features['file_name']
        })
    
    df_meta = pd.DataFrame(metadata)
    csv_path = os.path.join(output_dir, 'windows_metadata.csv')
    df_meta.to_csv(csv_path, index=False)
    output_files.append(csv_path)
    print(f"    ✓ Saved metadata: {csv_path}")
    
    # 2. 保存连接矩阵到.npz
    for idx, features in enumerate(all_features):
        arrays_dict = {}
        
        for key, value in features.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                arrays_dict[key] = value
        
        if arrays_dict:
            npz_path = os.path.join(output_dir, f'connectivity_matrices_seg{idx:04d}.npz')
            np.savez_compressed(npz_path, **arrays_dict)
            output_files.append(npz_path)
    
    print(f"    ✓ Saved {len(all_features)} connectivity matrix files")
    
    # 3. 保存通道名
    channel_file = os.path.join(output_dir, 'channel_names.txt')
    with open(channel_file, 'w') as f:
        for ch in channel_names:
            f.write(f"{ch}\n")
    output_files.append(channel_file)
    
    # 4. 保存汇总
    summary = {
        'n_windows': len(all_features),
        'n_segments': len(set([f['segment_id'] for f in all_features])),
        'n_channels': all_features[0]['n_channels'],
        'window_size': all_features[0]['duration'],
        'total_duration': sum([f['duration'] for f in all_features]),
        'n_concatenated': sum([1 for f in all_features if f['is_concatenated']])
    }
    
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    output_files.append(summary_file)
    
    return output_files


def process_batch(input_dir, pattern="*_merged_*.set", output_base_dir=None, config=Config()):
    """批量处理"""
    print(f"\n{'='*80}")
    print(f"Batch Processing")
    print(f"{'='*80}")
    print(f"Input: {input_dir}")
    print(f"Pattern: {pattern}")
    print(f"Window size: {config.WINDOW_SIZE}s")
    print(f"Visualization: {'Enabled' if config.ENABLE_VISUALIZATION else 'Disabled'}")
    
    input_path = Path(input_dir)
    set_files = list(input_path.rglob(pattern))
    
    print(f"\nFound {len(set_files)} files")
    
    if len(set_files) == 0:
        print("ERROR: No files found!")
        return []
    
    processed = []
    failed = []
    
    for idx, set_file in enumerate(set_files):
        print(f"\n{'='*80}")
        print(f"File {idx+1}/{len(set_files)}")
        print(f"{'='*80}")
        
        try:
            if output_base_dir:
                rel_path = set_file.relative_to(input_path)
                output_dir = os.path.join(output_base_dir, 
                                         str(rel_path.parent), 
                                         set_file.stem + '_connectivity_v2')
            else:
                output_dir = None
            
            output_files = process_single_file(str(set_file), output_dir, config)
            
            if output_files:
                processed.append((str(set_file), output_dir))
        
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            failed.append(str(set_file))
    
    print(f"\n{'='*80}")
    print(f"Batch Complete")
    print(f"{'='*80}")
    print(f"Success: {len(processed)}/{len(set_files)}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    
    return processed


# ============================================================================
# 命令行接口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="提取EEG连接性特征（改进版 - 5秒窗口，无拼接）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_file', help="单个.set文件")
    group.add_argument('--input_dir', help="输入目录（批量）")
    
    parser.add_argument('--output_dir', help="输出目录")
    parser.add_argument('--pattern', default="*_merged_*.set", help="文件匹配模式")
    parser.add_argument('--window_size', type=float, default=5, help="窗口大小（秒），默认5")
    parser.add_argument('--overlap', type=float, default=0, help="重叠（秒），默认0")
    parser.add_argument('--no_visualize', action='store_true', help="禁用可视化")
    
    args = parser.parse_args()
    
    # 配置
    config = Config()
    config.WINDOW_SIZE = args.window_size
    config.OVERLAP = args.overlap
    config.ENABLE_VISUALIZATION = not args.no_visualize
    
    # 处理
    if args.input_file:
        process_single_file(args.input_file, args.output_dir, config)
    else:
        process_batch(args.input_dir, args.pattern, args.output_dir, config)
    
    print("\n✓ All done!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # 默认参数
        sys.argv.extend([
            '--input_dir', r'E:\DataSet\EEG\EEG dataset_SUAT_processed',
            '--output_dir', r'E:\output\connectivity_features_v2',
            '--pattern', '*_merged_*.set',
            '--window_size', '6',
            '--no_visualize'  # 批量处理时默认不可视化
        ])
    sys.exit(main())

