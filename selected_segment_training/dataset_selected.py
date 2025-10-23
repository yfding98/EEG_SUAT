#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_selected.py

处理_selected.set文件的数据加载器
支持多频段特征提取和窗口切分
"""

import os
import numpy as np
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy import signal
from typing import List, Tuple, Dict
import re

# 抑制MNE警告
mne.set_log_level('ERROR')


# 频段定义（参考最新文献）
FREQUENCY_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 80),
    'hfo': (80, 250)  # High-Frequency Oscillations - 重要的致痫指标
}


def parse_channel_names_from_filename(filename):
    """
    从_selected文件名中解析标记的异常通道
    
    例如: SZ2_postICA_selected_T4_F8_Sph_R.set -> ['T4', 'F8', 'Sph-R']
    
    参数:
        filename: 文件名
    
    返回:
        channels: 通道名称列表
    """
    # 移除.set后缀
    name = filename.replace('.set', '')
    
    # 找到_selected_的位置
    if '_selected_' not in name:
        return []
    
    # 提取_selected_之后的部分
    suffix = name.split('_selected_')[1]
    
    # 按下划线分割得到通道名
    parts = suffix.split('_')
    
    channels = []
    i = 0
    while i < len(parts):
        # 检查是否是 Sph/R 或 Sphe/L 这种模式
        if i + 1 < len(parts) and parts[i] in ['Sph', 'Sphe'] and parts[i+1] in ['L', 'R']:
            channels.append(f"{parts[i]}-{parts[i+1]}")
            i += 2
        else:
            channels.append(parts[i])
            i += 1
    
    return channels


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    带通滤波
    
    参数:
        data: (n_channels, n_samples)
        lowcut: 低频截止
        highcut: 高频截止
        fs: 采样率
        order: 滤波器阶数
    
    返回:
        filtered_data: (n_channels, n_samples)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # 确保频率在有效范围内
    low = max(low, 0.01)
    high = min(high, 0.99)
    
    if low >= high:
        return np.zeros_like(data)
    
    try:
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data, axis=1)
        return filtered
    except Exception as e:
        print(f"Warning: Bandpass filter failed: {e}")
        return np.zeros_like(data)


def extract_multiband_features(data, fs):
    """
    提取多频段特征
    
    参数:
        data: (n_channels, n_samples)
        fs: 采样率
    
    返回:
        bands: list of (n_channels, n_samples), 每个频段的数据
    """
    bands = []
    
    for band_name, (lowcut, highcut) in FREQUENCY_BANDS.items():
        # 调整频段范围以适应采样率
        highcut_adjusted = min(highcut, fs / 2 - 1)
        
        if lowcut < highcut_adjusted:
            band_data = bandpass_filter(data, lowcut, highcut_adjusted, fs)
            bands.append(band_data)
        else:
            # 如果频段超出采样率范围，用零填充
            bands.append(np.zeros_like(data))
    
    return bands


class SelectedSegmentDataset(Dataset):
    """
    处理_selected.set文件的数据集
    
    功能:
    1. 读取_selected.set文件
    2. 从文件名解析异常通道标签
    3. 按窗口切分数据
    4. 提取多频段特征
    5. 统一采样率（重采样）
    """
    
    def __init__(
        self,
        data_root,
        window_size=6.0,
        window_stride=3.0,
        pattern="*_selected_*.set",
        use_multiband=True,
        target_sfreq=250.0  # 目标采样率
    ):
        """
        参数:
            data_root: 数据根目录
            window_size: 窗口大小（秒）
            window_stride: 窗口步长（秒）
            pattern: 文件匹配模式
            use_multiband: 是否使用多频段特征
            target_sfreq: 目标采样率（Hz），所有数据会重采样到此采样率
        """
        self.data_root = Path(data_root)
        self.window_size = window_size
        self.window_stride = window_stride
        self.use_multiband = use_multiband
        self.target_sfreq = target_sfreq
        
        # 查找所有_selected文件
        self.file_list = list(self.data_root.rglob(pattern))
        
        if not self.file_list:
            raise ValueError(f"No files found matching pattern: {pattern}")
        
        print(f"Found {len(self.file_list)} files")
        
        # 预处理：读取所有文件并切分窗口
        self.windows = []
        self.channel_names = None
        
        self._prepare_windows()
    
    def _prepare_windows(self):
        """预处理所有文件，切分窗口"""
        print("\nPreparing windows...")
        
        for file_path in self.file_list:
            print(f"\nProcessing: {file_path.name}")
            
            # 解析异常通道
            abnormal_channels = parse_channel_names_from_filename(file_path.name)
            
            if not abnormal_channels:
                print(f"  Warning: Could not parse abnormal channels from filename")
                continue
            
            print(f"  Abnormal channels: {abnormal_channels}")
            
            try:
                # 读取EEG数据
                raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose='ERROR')
                
                original_sfreq = raw.info['sfreq']
                
                # 重采样到目标采样率（如果需要）
                if abs(original_sfreq - self.target_sfreq) > 1e-3:
                    print(f"  Resampling from {original_sfreq}Hz to {self.target_sfreq}Hz...")
                    raw.resample(self.target_sfreq, npad='auto', verbose='ERROR')
                
                # 获取数据
                data = raw.get_data()  # (n_channels, n_samples)
                fs = raw.info['sfreq']  # 应该等于target_sfreq
                
                # 保存通道名称（第一次）
                if self.channel_names is None:
                    self.channel_names = raw.ch_names
                
                # 检查通道一致性
                if raw.ch_names != self.channel_names:
                    print(f"  Warning: Channel mismatch, skipping file")
                    continue
                
                n_channels, n_samples = data.shape
                
                # 计算窗口参数
                window_samples = int(self.window_size * fs)
                stride_samples = int(self.window_stride * fs)
                
                # 创建标签（one-hot）- 对整个文件创建一次
                labels = np.zeros(n_channels, dtype=np.float32)
                matched_channels = []
                unmatched_channels = []
                
                for ch_name in abnormal_channels:
                    # 尝试多种匹配方式
                    matched = False
                    
                    # 1. 精确匹配
                    if ch_name in self.channel_names:
                        ch_idx = self.channel_names.index(ch_name)
                        labels[ch_idx] = 1.0
                        matched_channels.append(ch_name)
                        matched = True
                    # 2. 大小写不敏感匹配
                    elif ch_name.upper() in [c.upper() for c in self.channel_names]:
                        for i, c in enumerate(self.channel_names):
                            if c.upper() == ch_name.upper():
                                labels[i] = 1.0
                                matched_channels.append(f"{ch_name}->{self.channel_names[i]}")
                                matched = True
                                break
                    # 3. 尝试去掉连字符再匹配
                    elif ch_name.replace('-', '') in self.channel_names:
                        ch_idx = self.channel_names.index(ch_name.replace('-', ''))
                        labels[ch_idx] = 1.0
                        matched_channels.append(f"{ch_name}->{ch_name.replace('-', '')}")
                        matched = True
                    
                    if not matched:
                        unmatched_channels.append(ch_name)
                
                # 输出匹配信息（仅一次）
                if matched_channels:
                    print(f"  ✓ Matched channels: {matched_channels}")
                if unmatched_channels:
                    print(f"  ✗ Unmatched channels: {unmatched_channels}")
                    print(f"  Available channels: {self.channel_names}")
                
                # 检查是否有标记的通道
                if labels.sum() == 0:
                    print(f"  Warning: No abnormal channels matched in channel list")
                    print(f"  Abnormal from filename: {abnormal_channels}")
                    print(f"  Available channels: {self.channel_names}")
                    continue
                
                # 切分窗口
                n_windows = 0
                for start_sample in range(0, n_samples - window_samples + 1, stride_samples):
                    end_sample = start_sample + window_samples
                    
                    # 提取窗口数据
                    window_data = data[:, start_sample:end_sample]
                    
                    # 添加窗口（使用相同的标签）
                    self.windows.append({
                        'data': window_data,
                        'labels': labels,
                        'fs': fs,
                        'file': file_path.name,
                        'abnormal_channels': abnormal_channels
                    })
                    n_windows += 1
                
                print(f"  Extracted {n_windows} windows (window_size={self.window_size}s, stride={self.window_stride}s)")
                print(f"  Labeled {int(labels.sum())} abnormal channels out of {n_channels} total")
                
            except Exception as e:
                print(f"  Error processing file: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not self.windows:
            raise ValueError("No valid windows extracted from any file")
        
        print(f"\n{'='*80}")
        print(f"Dataset prepared:")
        print(f"  Total windows: {len(self.windows)}")
        print(f"  Channels: {len(self.channel_names)}")
        print(f"  Window size: {self.window_size}s")
        print(f"  Window stride: {self.window_stride}s")
        print(f"  Unified sampling rate: {self.target_sfreq}Hz")
        print(f"  Expected samples per window: {int(self.window_size * self.target_sfreq)}")
        print(f"{'='*80}\n")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        data = window['data']  # (n_channels, n_samples)
        labels = window['labels']  # (n_channels,)
        fs = window['fs']
        
        if self.use_multiband:
            # 提取多频段特征
            bands = extract_multiband_features(data, fs)
            
            # 裁剪到合理范围并转换为torch tensors（避免overflow）
            bands_tensor = []
            for band in bands:
                # 裁剪到float32可表示的范围
                band_clipped = np.clip(band, -1e10, 1e10)
                bands_tensor.append(torch.from_numpy(band_clipped.astype(np.float32)))
            
            labels_tensor = torch.from_numpy(labels)
            
            return {
                'bands': bands_tensor,
                'labels': labels_tensor,
                'file': window['file']
            }
        else:
            # 单频段（原始数据）
            data_tensor = torch.from_numpy(data.astype(np.float32))
            labels_tensor = torch.from_numpy(labels)
            
            return {
                'data': data_tensor,
                'labels': labels_tensor,
                'file': window['file']
            }


def collate_multiband(batch):
    """
    自定义collate函数，处理多频段数据
    """
    # 收集所有bands
    n_bands = len(batch[0]['bands'])
    all_bands = [[] for _ in range(n_bands)]
    all_labels = []
    all_files = []
    
    for item in batch:
        bands = item['bands']
        labels = item['labels']
        
        for i, band in enumerate(bands):
            all_bands[i].append(band)
        
        all_labels.append(labels)
        all_files.append(item['file'])
    
    # Stack
    stacked_bands = [torch.stack(band_list, dim=0) for band_list in all_bands]
    stacked_labels = torch.stack(all_labels, dim=0)
    
    return {
        'bands': stacked_bands,
        'labels': stacked_labels,
        'files': all_files
    }


def create_dataloaders(
    data_root,
    batch_size=16,
    window_size=6.0,
    window_stride=3.0,
    val_split=0.15,
    test_split=0.15,
    num_workers=0,
    seed=42,
    target_sfreq=250.0
):
    """
    创建训练、验证和测试数据加载器
    
    参数:
        data_root: 数据根目录
        batch_size: 批大小
        window_size: 窗口大小（秒）
        window_stride: 窗口步长（秒）
        val_split: 验证集比例
        test_split: 测试集比例
        num_workers: 数据加载线程数
        seed: 随机种子
        target_sfreq: 目标采样率（Hz）
    
    返回:
        train_loader, val_loader, test_loader, channel_names
    """
    # 创建完整数据集
    full_dataset = SelectedSegmentDataset(
        data_root=data_root,
        window_size=window_size,
        window_stride=window_stride,
        use_multiband=True,
        target_sfreq=target_sfreq
    )
    
    # 划分数据集
    total_size = len(full_dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - test_size - val_size
    
    from torch.utils.data import random_split
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # 创建DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_multiband,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multiband,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multiband,
        pin_memory=True
    )
    
    print(f"\nDataLoader Summary:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, full_dataset.channel_names


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录（包含_selected.set文件）')
    parser.add_argument('--window_size', type=float, default=6.0,
                        help='窗口大小（秒）')
    parser.add_argument('--window_stride', type=float, default=3.0,
                        help='窗口步长（秒）')
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    
    print("="*80)
    print("测试数据加载器")
    print("="*80)
    
    try:
        train_loader, val_loader, test_loader, channel_names = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            window_size=args.window_size,
            window_stride=args.window_stride
        )
        
        print(f"\n通道名称: {channel_names}")
        print(f"\n读取一个batch:")
        
        batch = next(iter(train_loader))
        print(f"  Bands: {len(batch['bands'])} 个频段")
        for i, band in enumerate(batch['bands']):
            print(f"    Band {i}: shape={band.shape}")
        print(f"  Labels: shape={batch['labels'].shape}")
        print(f"  Files: {batch['files'][:3]}...")
        
        print("\n✓ 数据加载器测试成功！")
        
    except Exception as e:
        print(f"\n✗ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()

