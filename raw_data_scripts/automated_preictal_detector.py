import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import zscore
import matplotlib.patches as mpatches
from matplotlib.widgets import CheckButtons, Slider, Button
import tkinter as tk
from tkinter import ttk, filedialog
import csv


# ======================================================
# 工具函数
# ======================================================
def sliding_windows(data, sfreq, win_sec=2, step_sec=0.5):
    """生成滑动窗口 (start, end)"""
    win_size = int(win_sec * sfreq)
    step_size = int(step_sec * sfreq)
    starts = np.arange(0, len(data) - win_size, step_size)
    return [(s, s + win_size) for s in starts]


def bandpower(data, sfreq, band, nperseg=None):
    """计算某一频带功率"""
    fmin, fmax = band
    freqs, psd = welch(data, sfreq, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    band_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[band_idx], dx=freq_res)


def merge_intervals(intervals, gap=0.5):
    """合并相邻的异常区间"""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end, score in intervals[1:]:
        last_start, last_end, last_score = merged[-1]
        if start - last_end <= gap:
            merged[-1] = (last_start, max(last_end, end), max(last_score, score))
        else:
            merged.append((start, end, score))
    return merged


# ======================================================
# 异常检测
# ======================================================
def compute_global_stats(raw, sfreq, win_sec=2, step_sec=0.5):
    """计算全局参考统计量 (RMS 和 频带占比)"""
    rms_vals = []
    band_ratios_all = {"delta": [], "theta": [], "alpha": [], "beta": [], "gamma": []}

    for ch in raw.ch_names:
        data = raw.get_data(picks=[ch])[0]
        windows = sliding_windows(data, sfreq, win_sec, step_sec)
        for s, e in windows:
            seg = data[s:e]
            rms = np.sqrt(np.mean(seg**2))
            rms_vals.append(rms)

            bands = {
                "delta": (0.5, 4),
                "theta": (4, 7),
                "alpha": (8, 13),
                "beta": (14, 30),
                "gamma": (30, 80),
            }
            band_powers = {b: bandpower(seg, sfreq, rng) for b, rng in bands.items()}
            total_power = sum(band_powers.values()) + 1e-12
            for b in bands:
                band_ratios_all[b].append(band_powers[b] / total_power)

    ref_stats = {
        "rms_mean": np.mean(rms_vals),
        "rms_std": np.std(rms_vals),
        "bands_mean": {b: np.mean(vals) for b, vals in band_ratios_all.items()},
        "bands_std": {b: np.std(vals) for b, vals in band_ratios_all.items()},
    }
    return ref_stats


def detect_abnormal(data, sfreq, ref_stats, threshold=2.5):
    """检测单窗口是否异常"""
    rms = np.sqrt(np.mean(data**2))

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 7),
        "alpha": (8, 13),
        "beta": (14, 30),
        "gamma": (30, 80),
    }
    band_powers = {b: bandpower(data, sfreq, rng) for b, rng in bands.items()}
    total_power = sum(band_powers.values()) + 1e-12
    band_ratios = {b: p / total_power for b, p in band_powers.items()}

    # 计算Z-score
    scores = {}
    rms_z = (rms - ref_stats['rms_mean']) / (ref_stats['rms_std'] + 1e-6)
    scores["rms"] = rms_z
    for b in bands:
        scores[b] = (band_ratios[b] - ref_stats['bands_mean'][b]) / (ref_stats['bands_std'][b] + 1e-6)

    # 判定是否异常
    abnormal = False
    if scores["rms"] > threshold:
        abnormal = True
    if scores["delta"] > threshold or scores["theta"] > threshold:
        abnormal = True
    if scores["beta"] > threshold or scores["gamma"] > threshold:
        abnormal = True

    score = np.mean(list(scores.values()))
    return abnormal, score


# ======================================================
# 绘图
# ======================================================
def plot_channel_with_bands(raw, channel, sfreq, win_sec, step_sec, ref_stats, abnormal_intervals, outdir):
    """绘制通道波形 + 频带功率变化"""
    data = raw.get_data(picks=[channel])[0]
    times = np.arange(len(data)) / sfreq

    windows = sliding_windows(data, sfreq, win_sec, step_sec)
    centers = []
    band_trends = {"delta": [], "theta": [], "alpha": [], "beta": [], "gamma": []}

    for s, e in windows:
        seg = data[s:e]
        bands = {
            "delta": (0.5, 4),
            "theta": (4, 7),
            "alpha": (8, 13),
            "beta": (14, 30),
            "gamma": (30, 80),
        }
        band_powers = {b: bandpower(seg, sfreq, rng) for b, rng in bands.items()}
        total_power = sum(band_powers.values()) + 1e-12
        for b in bands:
            band_trends[b].append(band_powers[b] / total_power)
        centers.append((s + e) / 2 / sfreq)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # (1) 原始EEG + 异常区间
    axes[0].plot(times, data, color="black", lw=0.5)
    axes[0].set_title(f"Channel {channel} - EEG with Abnormal Intervals")
    for (start, end, score) in abnormal_intervals:
        axes[0].axvspan(start, end, color="red", alpha=0.3)

    # (2) 频带功率随时间
    for b, vals in band_trends.items():
        axes[1].plot(centers, vals, label=b)
        axes[1].axhline(ref_stats["bands_mean"][b], linestyle="--", lw=0.8, alpha=0.5)

    axes[1].set_title("Relative Bandpower Over Time")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{outdir}/{channel}_bands.png", dpi=150)
    plt.close(fig)


def plot_gantt(abnormal_dict, outdir, filename="abnormal_gantt.png"):
    """绘制所有通道异常区间的甘特图"""
    channels = list(abnormal_dict.keys())
    fig, ax = plt.subplots(figsize=(12, max(6, len(channels) * 0.4)))

    for i, ch in enumerate(channels):
        intervals = abnormal_dict[ch]
        for (start, end, score) in intervals:
            ax.barh(y=i, width=end - start, left=start, height=0.4,
                    color="red", alpha=0.6)

    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    ax.set_xlabel("Time (s)")
    ax.set_title("Abnormal EEG Intervals per Channel")

    red_patch = mpatches.Patch(color='red', alpha=0.6, label='Abnormal interval')
    ax.legend(handles=[red_patch])

    plt.tight_layout()
    plt.savefig(f"{outdir}/{filename}", dpi=150)
    plt.close(fig)


def calculate_original_time_from_annotations(current_time, raw):
    """通过raw.annotations中的boundary信息计算原始数据中的时间戳"""
    try:
        # 获取采样频率
        sfreq = raw.info['sfreq']
        
        # 计算当前时间对应的采样点
        current_sample = int(current_time * sfreq)
        
        # 累计所有boundary annotations之前删除的采样点数
        total_deleted_samples = 0
        for ann in raw.annotations:
            if ann['description'] == 'boundary':
                # 计算boundary annotation在裁剪后数据中的采样点位置
                # 需要先计算这个boundary在裁剪后数据中的位置
                boundary_original_start = ann['onset']
                boundary_duration = ann['duration']
                
                # 计算这个boundary之前已经删除的总时间
                previous_deleted_time = 0
                for prev_ann in raw.annotations:
                    if prev_ann['description'] == 'boundary' and prev_ann['onset'] < boundary_original_start:
                        previous_deleted_time += prev_ann['duration']
                
                # 计算boundary在裁剪后数据中的位置
                boundary_cropped_start = boundary_original_start - previous_deleted_time
                boundary_cropped_sample = int(boundary_cropped_start * sfreq)
                
                if boundary_cropped_sample <= current_sample:
                    # 计算删除的采样点数
                    deleted_samples = int(boundary_duration * sfreq)
                    total_deleted_samples += deleted_samples
        
        # 转换为原始时间戳
        original_time = (current_sample + total_deleted_samples) / sfreq
        return original_time
        
    except Exception as e:
        print(f"Warning: Could not calculate original time from annotations: {e}")
        return current_time

def create_interactive_gantt(abnormal_dict, outdir, filename="interactive_gantt.png", raw=None):
    """创建交互式甘特图，支持通道选择、时间窗口滚动和时间段标记"""
    channels = list(abnormal_dict.keys())
    if not channels:
        print("No channels to plot.")
        return
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(14, max(6, len(channels) * 0.4)))
    plt.subplots_adjust(bottom=0.25, left=0.1)
    
    # 计算原始数据的总时间范围
    if raw is not None:
        # 获取原始数据的总时长（包括所有删除的片段）
        original_total_time = raw.times[-1]  # 裁剪后数据的最后时间
        # 加上所有删除片段的总时长
        total_deleted_time = sum([ann['duration'] for ann in raw.annotations if ann['description'] == 'boundary'])
        original_total_time += total_deleted_time
    else:
        original_total_time = max([max([end for _, end, _ in intervals]) for intervals in abnormal_dict.values()]) if any(abnormal_dict.values()) else 100
    
    window_size = min(100, original_total_time)  # 默认窗口大小
    current_start = 0
    
    # 存储标注的时间段
    marked_segments = []
    
    # 更新绘图
    def update_plot():
        ax.clear()
        ax.set_xlim(current_start, current_start + window_size)
        ax.set_ylim(-0.5, len(channels) - 0.5)
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels(channels)
        ax.set_xlabel("Time (s) - Original Data Timeline")
        ax.set_title("Interactive Abnormal EEG Intervals (Select channels, scroll time)")
        
        # 绘制删除的发作片段（灰色区域）- 使用原始数据时间
        if raw is not None:
            for ann in raw.annotations:
                if ann['description'] == 'boundary':
                    original_start = ann['onset']
                    original_end = original_start + ann['duration']
                    if original_start < current_start + window_size and original_end > current_start:
                        ax.axvspan(original_start, original_end, color='gray', alpha=0.3, label='Rejected segments')
        
        # 绘制选中通道的区间 - 需要转换为原始数据时间
        for i, ch in enumerate(channels):
            if ch in selected_channels:
                intervals = abnormal_dict[ch]
                for start, end, score in intervals:
                    # 将裁剪后数据的时间转换为原始数据时间
                    if raw is not None:
                        original_start = calculate_original_time_from_annotations(start, raw)
                        original_end = calculate_original_time_from_annotations(end, raw)
                    else:
                        original_start, original_end = start, end
                    
                    if original_start < current_start + window_size and original_end > current_start:
                        ax.barh(y=i, width=original_end - original_start, left=original_start, height=0.4,
                                color="red", alpha=0.6)
                        # 标记异常通道的时间段
                        mid_time = (original_start + original_end) / 2
                        ax.text(mid_time, i, f"{ch}", ha='center', va='center', fontsize=8, color='white')
        
        # 绘制已标注的时间段 - 已经是原始数据时间
        for start, end, chs in marked_segments:
            if start < current_start + window_size and end > current_start:
                ax.axvspan(start, end, color='yellow', alpha=0.3)
                ax.text((start + end) / 2, len(channels) / 2, f"Marked: {chs}", 
                       ha='center', va='center', fontsize=10, color='red', weight='bold')
    
    # 通道选择复选框 - 初始全不选
    selected_channels = set()  # 初始全不选
    checkbox_ax = plt.axes([0.02, 0.02, 0.15, 0.15])
    checkboxes = CheckButtons(checkbox_ax, channels, [False] * len(channels))
    
    def toggle_channel(label):
        if label in selected_channels:
            selected_channels.remove(label)
        else:
            selected_channels.add(label)
        update_plot()
        plt.draw()
    
    checkboxes.on_clicked(toggle_channel)
    
    # 时间窗口滑块
    slider_ax = plt.axes([0.25, 0.02, 0.5, 0.03])
    slider = Slider(slider_ax, 'Time Start', 0, max(0, original_total_time - window_size), valinit=current_start)
    
    def update_time(val):
        nonlocal current_start
        current_start = val
        update_plot()
        plt.draw()
    
    slider.on_changed(update_time)
    
    # 窗口大小滑块
    size_ax = plt.axes([0.8, 0.02, 0.15, 0.03])
    size_slider = Slider(size_ax, 'Window Size', 10, min(200, original_total_time), valinit=window_size)
    
    def update_size(val):
        nonlocal window_size
        window_size = val
        update_plot()
        plt.draw()
    
    size_slider.on_changed(update_size)
    
    # 标记时间段按钮
    mark_ax = plt.axes([0.25, 0.08, 0.1, 0.03])
    mark_button = Button(mark_ax, 'Mark Segments')
    
    def mark_segments(event):
        # 找到当前窗口中所有异常通道的时间段（裁剪后数据时间）
        segments = []
        for ch in selected_channels:
            intervals = abnormal_dict[ch]
            for start, end, score in intervals:
                # 将裁剪后数据时间转换为原始数据时间进行判断
                if raw is not None:
                    original_start = calculate_original_time_from_annotations(start, raw)
                    original_end = calculate_original_time_from_annotations(end, raw)
                else:
                    original_start, original_end = start, end
                
                if original_start < current_start + window_size and original_end > current_start:
                    segments.append((original_start, original_end, ch))
        
        # 合并重叠的时间段
        if segments:
            segments.sort()
            merged_segments = []
            current_seg = segments[0]
            for seg in segments[1:]:
                if seg[0] <= current_seg[1]:  # 重叠
                    current_seg = (current_seg[0], max(current_seg[1], seg[1]), current_seg[2] + f",{seg[2]}")
                else:
                    merged_segments.append(current_seg)
                    current_seg = seg
            merged_segments.append(current_seg)
            
            # 添加到标注列表（已经是原始数据时间）
            for start, end, chs in merged_segments:
                marked_segments.append((start, end, chs))
            
            # 高亮时间段
            for start, end, chs in merged_segments:
                ax.axvspan(start, end, color='yellow', alpha=0.3)
                ax.text((start + end) / 2, len(channels) / 2, f"Marked: {chs}", 
                       ha='center', va='center', fontsize=10, color='red', weight='bold')
        
        plt.draw()
    
    mark_button.on_clicked(mark_segments)
    
    # 保存标注按钮
    save_ax = plt.axes([0.4, 0.08, 0.1, 0.03])
    save_button = Button(save_ax, 'Save Marked')
    
    def save_marked_segments(event):
        if marked_segments:
            # 保存到CSV文件
            csv_path = os.path.join(outdir, "marked_abnormal_segments.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['开始时间', '结束时间', '异常通道列表'])
                for start, end, chs in marked_segments:
                    writer.writerow([start, end, chs])
            print(f"Saved {len(marked_segments)} marked segments to {csv_path}")
        else:
            print("No marked segments to save.")
    
    save_button.on_clicked(save_marked_segments)
    
    # 初始绘图
    update_plot()
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=150, bbox_inches='tight')
    plt.show()
    
    return marked_segments


# ======================================================
# 主流程
# ======================================================
def analyze_file(eeg_file, win_sec=2, step_sec=0.5, threshold=2.5):
    outdir = os.path.splitext(eeg_file)[0] + "_results"
    os.makedirs(outdir, exist_ok=True)

    # 读取EEG
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
    sfreq = raw.info["sfreq"]

    # 检查医生标注的通道
    effect_channels_file = os.path.join(os.path.dirname(eeg_file), "effect_channels.csv")
    if os.path.exists(effect_channels_file):
        effect_channels = pd.read_csv(effect_channels_file, header=None)[0].tolist()
        channels = [ch for ch in raw.ch_names if ch in effect_channels]
    else:
        channels = raw.ch_names

    # 计算全局参考
    ref_stats = compute_global_stats(raw, sfreq, win_sec, step_sec)

    results = {}

    # 遍历通道
    for ch in channels:
        data = raw.get_data(picks=[ch])[0]
        windows = sliding_windows(data, sfreq, win_sec, step_sec)

        abnormal_intervals = []
        for s, e in windows:
            seg = data[s:e]
            flag, score = detect_abnormal(seg, sfreq, ref_stats, threshold)
            if flag:
                abnormal_intervals.append((s/sfreq, e/sfreq, score))

        merged = merge_intervals(abnormal_intervals, gap=0.5)
        results[ch] = merged

        # 绘图
        plot_channel_with_bands(raw, ch, sfreq, win_sec, step_sec, ref_stats, merged, outdir)

    # 输出CSV
    rows = []
    for ch, intervals in results.items():
        for (start, end, score) in intervals:
            rows.append([ch, start, end, score])
    df = pd.DataFrame(rows, columns=["Channel", "Start(s)", "End(s)", "Score"])
    df.to_csv(f"{outdir}/abnormal_intervals.csv", index=False)

    # 绘制甘特图
    plot_gantt(results, outdir)
    
    # 创建交互式甘特图
    marked_segments = create_interactive_gantt(results, outdir, raw=raw)
    
    return marked_segments


# ======================================================
# 批量处理函数
# ======================================================
def process_multiple_files(root_dir, win_sec=2, step_sec=0.5, threshold=2.5):
    """批量处理目录下所有以'_reject_1_postICA'结尾的.set文件"""
    set_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_reject_1_postICA.set'):
                set_files.append(os.path.join(subdir, file))
    
    print(f"Found {len(set_files)} matching .set files.")
    
    for set_path in set_files:
        print(f"Processing: {set_path}")
        try:
            analyze_file(set_path, win_sec, step_sec, threshold)
            print(f"Completed: {set_path}")
        except Exception as e:
            print(f"Error processing {set_path}: {e}")


# ======================================================
# 入口
# ======================================================
if __name__ == "__main__":
    # 单个文件处理
    eeg_file = r"E:\DataSet\EEG\EEG dataset_SUAT_processed\头皮数据-6例\江仁坤\SZ1_preICA_reject_1_postICA.set"   # 修改为你的EEG文件路径
    analyze_file(eeg_file)
    
    # 批量处理（取消注释以使用）
    # root_directory = r"E:\DataSet\EEG\EEG dataset_SUAT_processed\头皮数据-6例"  # 修改为你的根目录
    # process_multiple_files(root_directory)
