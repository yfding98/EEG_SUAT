import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import zscore
import matplotlib.patches as mpatches
from matplotlib.widgets import CheckButtons, Slider, Button, SpanSelector, RadioButtons
import tkinter as tk
from tkinter import ttk, filedialog
import csv


# ======================================================
# 时间转换函数（基于MATLAB脚本逻辑）
# ======================================================
def get_boundary_segments(raw):
    """从raw.annotations中提取所有boundary事件，并计算它们在原始数据中的时间段
    按照MATLAB脚本的逻辑：cumulative_offset累加每个boundary的duration
    """
    try:
        sfreq = raw.info['sfreq']
        boundaries = []
        cumulative_offset = 0
        
        # 按onset排序所有boundary annotations
        boundary_anns = [ann for ann in raw.annotations if ann['description'] == 'boundary']
        boundary_anns.sort(key=lambda x: x['onset'])
        
        for ann in boundary_anns:
            # boundary的latency（在裁剪后数据中的位置）+ cumulative_offset = 原始数据位置
            start_sample = ann['onset'] * sfreq + cumulative_offset
            dur_samples = ann['duration'] * sfreq
            end_sample = start_sample + dur_samples
            
            start_time = start_sample / sfreq
            end_time = end_sample / sfreq
            
            boundaries.append((start_time, end_time))
            cumulative_offset += dur_samples
        
        return boundaries
        
    except Exception as e:
        print(f"Warning: Could not extract boundary segments: {e}")
        return []


def calculate_original_time_from_cropped(cropped_time, raw):
    """将裁剪后数据的时间转换为原始数据的时间
    逻辑：找到所有在当前时间之前的boundary，累加它们的duration
    """
    try:
        sfreq = raw.info['sfreq']
        cropped_sample = cropped_time * sfreq
        
        # 按onset排序所有boundary annotations
        boundary_anns = [ann for ann in raw.annotations if ann['description'] == 'boundary']
        boundary_anns.sort(key=lambda x: x['onset'])
        
        cumulative_offset = 0
        for ann in boundary_anns:
            # boundary在裁剪后数据中的位置
            boundary_cropped_sample = ann['onset'] * sfreq
            
            if boundary_cropped_sample <= cropped_sample:
                # 这个boundary在当前时间之前，需要累加其duration
                dur_samples = ann['duration'] * sfreq
                cumulative_offset += dur_samples
            else:
                break
        
        # 原始时间 = 裁剪后采样点 + 累计删除的采样点
        original_sample = cropped_sample + cumulative_offset
        original_time = original_sample / sfreq
        
        return original_time
        
    except Exception as e:
        print(f"Warning: Could not calculate original time: {e}")
        return cropped_time


def convert_all_times_to_original(time_array, raw):
    """批量将裁剪后的时间数组转换为原始时间数组"""
    return np.array([calculate_original_time_from_cropped(t, raw) for t in time_array])


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
    """合并相邻的异常区间（时间已经是原始时间）"""
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
def plot_channel_with_bands(raw, channel, sfreq, win_sec, step_sec, ref_stats, abnormal_intervals, outdir, original_times):
    """绘制通道波形 + 频带功率变化（使用原始时间轴）"""
    data = raw.get_data(picks=[channel])[0]

    windows = sliding_windows(data, sfreq, win_sec, step_sec)
    centers_original = []
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
        # 转换为原始时间
        center_cropped = (s + e) / 2 / sfreq
        center_original = calculate_original_time_from_cropped(center_cropped, raw)
        centers_original.append(center_original)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # (1) 原始EEG + 异常区间（使用原始时间）
    axes[0].plot(original_times, data, color="black", lw=0.5)
    axes[0].set_title(f"Channel {channel} - EEG with Abnormal Intervals (Original Timeline)")
    for (start, end, score) in abnormal_intervals:
        axes[0].axvspan(start, end, color="red", alpha=0.3)

    # (2) 频带功率随时间（使用原始时间）
    for b, vals in band_trends.items():
        axes[1].plot(centers_original, vals, label=b)
        axes[1].axhline(ref_stats["bands_mean"][b], linestyle="--", lw=0.8, alpha=0.5)

    axes[1].set_title("Relative Bandpower Over Time (Original Timeline)")
    axes[1].set_xlabel("Time (s) - Original")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{outdir}/{channel}_bands.png", dpi=150)
    plt.close(fig)


def plot_gantt(abnormal_dict, boundary_segments, outdir, filename="abnormal_gantt.png"):
    """绘制所有通道异常区间的甘特图（原始时间轴）"""
    channels = list(abnormal_dict.keys())
    fig, ax = plt.subplots(figsize=(12, max(6, len(channels) * 0.4)))

    # 绘制删除的片段（灰色）
    for start, end in boundary_segments:
        ax.axvspan(start, end, color='gray', alpha=0.2)

    for i, ch in enumerate(channels):
        intervals = abnormal_dict[ch]
        for (start, end, score) in intervals:
            ax.barh(y=i, width=end - start, left=start, height=0.4,
                    color="red", alpha=0.6)

    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    ax.set_xlabel("Time (s) - Original Timeline")
    ax.set_title("Abnormal EEG Intervals per Channel (Original Timeline)")

    gray_patch = mpatches.Patch(color='gray', alpha=0.2, label='Rejected segments')
    red_patch = mpatches.Patch(color='red', alpha=0.6, label='Abnormal interval')
    ax.legend(handles=[gray_patch, red_patch])

    plt.tight_layout()
    plt.savefig(f"{outdir}/{filename}", dpi=150)
    plt.close(fig)


def create_interactive_gantt(abnormal_dict, boundary_segments, original_total_time, outdir, filename="interactive_gantt.png"):
    """创建交互式甘特图（所有时间都是原始时间轴）
    支持：
    1. 滚动选择通道
    2. 框选时间段标记
    3. 取消标记
    """
    channels = list(abnormal_dict.keys())
    if not channels:
        print("No channels to plot.")
        return []
    
    # 创建图形和坐标轴 - 调整布局以容纳滚动通道选择器
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes([0.25, 0.25, 0.7, 0.65])  # 主绘图区域
    
    window_size = min(100, original_total_time)  # 默认窗口大小
    current_start = 0
    
    # 存储标注的时间段
    marked_segments = []
    
    # 通道选择相关
    selected_channels = set()
    channels_per_page = 15  # 每页显示的通道数
    current_channel_page = 0
    max_channel_pages = max(1, (len(channels) + channels_per_page - 1) // channels_per_page)
    
    # SpanSelector 用于框选
    span_selector = None
    
    # 获取当前页的通道
    def get_current_page_channels():
        start_idx = current_channel_page * channels_per_page
        end_idx = min(start_idx + channels_per_page, len(channels))
        return channels[start_idx:end_idx]
    
    # 更新绘图
    def update_plot():
        ax.clear()
        ax.set_xlim(current_start, current_start + window_size)
        ax.set_ylim(-0.5, len(channels) - 0.5)
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels(channels)
        ax.set_xlabel("Time (s) - Original Data Timeline")
        ax.set_title("Interactive Abnormal EEG Intervals (Select channels, drag to mark time)")
        
        # 绘制删除的发作片段（灰色区域）
        for start, end in boundary_segments:
            if start < current_start + window_size and end > current_start:
                ax.axvspan(start, end, color='gray', alpha=0.3, label='Rejected segments')
        
        # 绘制选中通道的区间（已经是原始时间）
        for i, ch in enumerate(channels):
            if ch in selected_channels:
                intervals = abnormal_dict[ch]
                for start, end, score in intervals:
                    if start < current_start + window_size and end > current_start:
                        ax.barh(y=i, width=end - start, left=start, height=0.4,
                                color="red", alpha=0.6)
                        mid_time = (start + end) / 2
                        ax.text(mid_time, i, f"{ch}", ha='center', va='center', fontsize=8, color='white')
        
        # 绘制已标注的时间段
        for idx, (start, end, chs) in enumerate(marked_segments):
            if start < current_start + window_size and end > current_start:
                ax.axvspan(start, end, color='yellow', alpha=0.3, edgecolor='orange', linewidth=2)
                ax.text((start + end) / 2, len(channels) / 2, f"#{idx+1}: {chs}", 
                       ha='center', va='center', fontsize=9, color='red', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        fig.canvas.draw_idle()
    
    # 定义通道切换函数
    def toggle_channel(label):
        if label in selected_channels:
            selected_channels.remove(label)
        else:
            selected_channels.add(label)
        update_plot()
    
    # 更新通道选择复选框
    def update_channel_checkboxes():
        checkbox_ax.clear()
        page_channels = get_current_page_channels()
        page_states = [ch in selected_channels for ch in page_channels]
        
        nonlocal checkboxes
        checkboxes = CheckButtons(checkbox_ax, page_channels, page_states)
        checkboxes.on_clicked(toggle_channel)
        
        # 显示页码
        checkbox_ax.text(0.5, -0.05, f'Page {current_channel_page + 1}/{max_channel_pages}', 
                        transform=checkbox_ax.transAxes, ha='center', fontsize=9)
        fig.canvas.draw_idle()
    
    # 通道选择复选框 - 初始全不选，支持分页
    checkbox_ax = plt.axes([0.02, 0.35, 0.18, 0.55])
    checkboxes = None
    update_channel_checkboxes()
    
    # 通道页面导航按钮
    prev_page_ax = plt.axes([0.02, 0.30, 0.08, 0.03])
    prev_page_button = Button(prev_page_ax, 'Prev Page')
    
    def prev_page(event):
        nonlocal current_channel_page
        if current_channel_page > 0:
            current_channel_page -= 1
            update_channel_checkboxes()
    
    prev_page_button.on_clicked(prev_page)
    
    next_page_ax = plt.axes([0.12, 0.30, 0.08, 0.03])
    next_page_button = Button(next_page_ax, 'Next Page')
    
    def next_page(event):
        nonlocal current_channel_page
        if current_channel_page < max_channel_pages - 1:
            current_channel_page += 1
            update_channel_checkboxes()
    
    next_page_button.on_clicked(next_page)
    
    # 时间窗口滑块
    slider_ax = plt.axes([0.25, 0.15, 0.5, 0.03])
    slider = Slider(slider_ax, 'Time Start', 0, max(0, original_total_time - window_size), valinit=current_start)
    
    def update_time(val):
        nonlocal current_start
        current_start = val
        update_plot()
    
    slider.on_changed(update_time)
    
    # 窗口大小滑块
    size_ax = plt.axes([0.25, 0.10, 0.5, 0.03])
    size_slider = Slider(size_ax, 'Window Size', 10, min(200, original_total_time), valinit=window_size)
    
    def update_size(val):
        nonlocal window_size
        window_size = val
        slider.valmax = max(0, original_total_time - window_size)
        update_plot()
    
    size_slider.on_changed(update_size)
    
    # SpanSelector - 框选时间段
    def onselect(xmin, xmax):
        """当用户框选一个时间段时"""
        if len(selected_channels) == 0:
            print("请先选择至少一个通道！")
            return
        
        # 收集框选区域内所有选中通道的异常片段
        segments_in_range = []
        for ch in selected_channels:
            intervals = abnormal_dict[ch]
            for start, end, score in intervals:
                # 检查是否与框选区域有重叠
                if not (end < xmin or start > xmax):
                    overlap_start = max(start, xmin)
                    overlap_end = min(end, xmax)
                    segments_in_range.append((overlap_start, overlap_end, ch))
        
        if segments_in_range:
            # 合并重叠的时间段
            segments_in_range.sort()
            merged_temp = []
            current_seg = segments_in_range[0]
            for seg in segments_in_range[1:]:
                if seg[0] <= current_seg[1]:  # 重叠
                    current_seg = (current_seg[0], max(current_seg[1], seg[1]), current_seg[2] + f",{seg[2]}")
                else:
                    merged_temp.append(current_seg)
                    current_seg = seg
            merged_temp.append(current_seg)
            
            # 添加标注（使用框选的边界）
            chs_str = ','.join(sorted(set([ch for _, _, chs in merged_temp for ch in chs.split(',')])))
            marked_segments.append((xmin, xmax, chs_str))
            print(f"已标记时间段: {xmin:.2f}s - {xmax:.2f}s, 通道: {chs_str}")
            update_plot()
        else:
            print("框选区域内没有选中通道的异常片段！")
    
    span_selector = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                                 props=dict(alpha=0.3, facecolor='yellow'),
                                 interactive=True, drag_from_anywhere=True)
    
    # 取消最后一个标记按钮
    undo_ax = plt.axes([0.25, 0.05, 0.1, 0.03])
    undo_button = Button(undo_ax, 'Undo Mark')
    
    def undo_mark(event):
        if marked_segments:
            removed = marked_segments.pop()
            print(f"已取消标记: {removed[0]:.2f}s - {removed[1]:.2f}s")
            update_plot()
        else:
            print("没有可取消的标记！")
    
    undo_button.on_clicked(undo_mark)
    
    # 清除所有标记按钮
    clear_ax = plt.axes([0.37, 0.05, 0.1, 0.03])
    clear_button = Button(clear_ax, 'Clear All')
    
    def clear_all(event):
        nonlocal marked_segments
        if marked_segments:
            count = len(marked_segments)
            marked_segments = []
            print(f"已清除所有 {count} 个标记！")
            update_plot()
        else:
            print("没有标记需要清除！")
    
    clear_button.on_clicked(clear_all)
    
    # 保存标注按钮
    save_ax = plt.axes([0.49, 0.05, 0.1, 0.03])
    save_button = Button(save_ax, 'Save Marked')
    
    def save_marked_segments(event):
        if marked_segments:
            csv_path = os.path.join(outdir, "marked_abnormal_segments.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['开始时间', '结束时间', '异常通道列表'])
                for start, end, chs in marked_segments:
                    writer.writerow([start, end, chs])
            print(f"✅ 已保存 {len(marked_segments)} 个标记到: {csv_path}")
        else:
            print("没有标记需要保存！")
    
    save_button.on_clicked(save_marked_segments)
    
    # 使用说明
    info_text = "操作说明:\n1. 选择通道(左侧复选框)\n2. 在图上拖动框选时间段\n3. Undo撤销/Clear清空\n4. Save保存到CSV"
    fig.text(0.02, 0.15, info_text, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 初始绘图
    update_plot()
    
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
    
    # 获取boundary段（原始时间）
    boundary_segments = get_boundary_segments(raw)
    
    # 计算原始数据的总时间
    cropped_end_time = raw.times[-1]
    original_total_time = calculate_original_time_from_cropped(cropped_end_time, raw)
    
    # 将所有裁剪后的时间转换为原始时间（用于绘图）
    original_times = convert_all_times_to_original(raw.times, raw)

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
                # 将裁剪后的时间转换为原始时间
                start_original = calculate_original_time_from_cropped(s/sfreq, raw)
                end_original = calculate_original_time_from_cropped(e/sfreq, raw)
                abnormal_intervals.append((start_original, end_original, score))

        merged = merge_intervals(abnormal_intervals, gap=0.5)
        results[ch] = merged

        # 绘图
        # plot_channel_with_bands(raw, ch, sfreq, win_sec, step_sec, ref_stats, merged, outdir, original_times)

    # 输出CSV（原始时间）
    rows = []
    for ch, intervals in results.items():
        for (start, end, score) in intervals:
            rows.append([ch, start, end, score])
    df = pd.DataFrame(rows, columns=["Channel", "Start(s)-Original", "End(s)-Original", "Score"])
    df.to_csv(f"{outdir}/abnormal_intervals_original_time.csv", index=False)

    # 绘制甘特图（原始时间）
    plot_gantt(results, boundary_segments, outdir)
    
    # 创建交互式甘特图（原始时间）
    marked_segments = create_interactive_gantt(results, boundary_segments, original_total_time, outdir)
    
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
    # 批量处理所有文件
    # root_directory = r"E:\DataSet\EEG\EEG dataset_SUAT_processed"  # 修改为你的根目录
    # process_multiple_files(root_directory)

    root_directory = r"E:\DataSet\EEG\EEG dataset_SUAT_processed\补充\刘定治"  # 修改为你的根目录
    process_multiple_files(root_directory)
    
    # # 单个文件处理（取消注释以使用）
    # eeg_file = r"E:\DataSet\EEG\EEG dataset_SUAT_processed\新增10份\高萌\SZ1_preICA_reject_1_postICA.set"
    # analyze_file(eeg_file)

