#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_preictal_channels.py

目的：在 pre-ictal（发作前期）检测“最先出现异常”的电极（channels），并标注这些电极的异常时间区间。
适用：头皮 EEG（scalp EEG），基于用户提供的 baseline 段和发作起始时间。

用法示例：
    python detect_preictal_channels.py --input patient1.edf --seizure_onset 3600 \
        --baseline 0 300 --preictal_minutes 30 --out result.json

依赖：
    pip install mne numpy scipy pandas matplotlib
"""

import argparse
import json
import math
import os
from collections import defaultdict

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, zscore
from scipy.signal import welch
from mne.io import read_raw_eeglab
import plotly.graph_objects as go
import plotly.express as px
import tkinter as tk
from tkinter import ttk, filedialog
import csv

# ---------------------------
# 默认通道列表（用户给定）
# ---------------------------
DEFAULT_CHANNELS = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz','Sph-L','Sph-R'
]


# ---------------------------
# 工具函数 - 特征计算
# ---------------------------
def spectral_entropy(sig, n_bins=50):
    """计算信号的（幅度）熵，用直方图近似概率密度"""
    if np.all(sig == 0):
        return 0.0
    hist, _ = np.histogram(sig, bins=n_bins, density=True)
    p = hist + 1e-12
    ent = -np.sum(p * np.log(p))
    return float(ent)


def bandpower(sig, sfreq, band, method='welch'):
    """简化band power：直接通过时域方差近似（适合短窗）或可扩展为PSD。
    这里为了速度直接用方差作为功率近似；可以替换为 PSD 计算"""
    if method == 'welch':
        f, psd = welch(sig, fs=sfreq, nperseg=min(len(sig), int(sfreq * 2)))
        band_mask = (f >= band[0]) & (f < band[1])
        return np.sum(psd[band_mask]) if np.any(band_mask) else 0.0
    return 0.0

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 70)
}


def line_length(sig):
    """线长（常用于突发性/非平稳检测）"""
    return float(np.sum(np.abs(np.diff(sig))))


def compute_features_for_segment(sig, sfreq):
    """为一个段（1D np.array）计算特征向量"""
    features = {
        'variance': float(np.var(sig)),
        'line_length': line_length(sig),
        'kurtosis': float(kurtosis(sig, fisher=False)),
        'spectral_entropy': spectral_entropy(sig)
    }
    for band_name, band in BANDS.items():
        features[f'{band_name}_power'] = bandpower(sig, sfreq, band)
    return features


# ---------------------------
# 时间段和滑窗工具
# ---------------------------
def sliding_windows_indices(n_samples, win_samples, step_samples):
    idxs = list(range(0, max(1, n_samples - win_samples + 1), step_samples))
    return idxs


def merge_close_intervals(intervals, max_gap):
    """合并相互接近的区间；intervals: list of (start, end) sorted"""
    if not intervals:
        return []
    merged = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s - cur_e <= max_gap:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


# ---------------------------
# 多重比较 - Benjamini-Hochberg
# ---------------------------
def benjamini_hochberg(pvals, alpha=0.05):
    """返回 boolean list: 是否拒绝原假设（即显著）"""
    p = np.array(pvals)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresholds
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    max_k = np.max(np.where(below)[0])
    cutoff = ranked[max_k]
    return p <= cutoff


def compute_permutation_pvalue(test_stat, null_dist):
    return (np.sum(null_dist >= test_stat) + 1) / (len(null_dist) + 1)


# ---------------------------
# 主流程函数
# ---------------------------
def detect_preictal_channels(raw,
                             seizure_onset,
                             baseline_intervals,
                             preictal_minutes=30,
                             channels=DEFAULT_CHANNELS,
                             win_sec=2.0,
                             step_sec=1.0,
                             z_thr=2.0,
                             min_consec=2,
                             merge_gap_sec=2.0,
                             fdr_alpha=0.05,
                             n_permutations=1000,
                             do_plot=True,
                             out_prefix="result",
                             input_file=None,
                             out_dir='.'):
    """
    raw: MNE Raw object
    seizure_onset: float (seconds, absolute time in raw)
    baseline_intervals: list of (tmin, tmax) seconds for baseline (can be multiple)
    preictal_minutes: how many minutes before seizure_onset to analyze
    返回：字典结果并保存 JSON + 图像
    """
    sfreq = raw.info['sfreq']
    win_samples = int(round(win_sec * sfreq))
    step_samples = int(round(step_sec * sfreq))
    merge_gap_samples = int(round(merge_gap_sec * sfreq))

    # 确保通道在 raw 中
    pick_chs = [ch for ch in channels if ch in raw.ch_names]
    if len(pick_chs) == 0:
        raise ValueError("No requested channels found in raw data. Available channels: {}".format(raw.ch_names))
    raw_pick = raw.copy().pick_channels(pick_chs)

    # 定义 baseline 数据（把多个 baseline interval 抽样拼接）
    baseline_data_list = []
    for (t0, t1) in baseline_intervals:
        t0 = float(t0); t1 = float(t1)
        if t1 <= t0:
            continue
        try:
            seg = raw_pick.copy().crop(tmin=t0, tmax=t1).get_data()
            baseline_data_list.append(seg)
        except Exception:
            continue
    if len(baseline_data_list) == 0:
        raise ValueError("No valid baseline intervals extracted. Check baseline_intervals and raw data times.")
    baseline_data = np.concatenate(baseline_data_list, axis=1)  # ch x samples

    # 切出 pre-ictal 段
    preictal_t0 = max(0.0, float(seizure_onset) - float(preictal_minutes) * 60.0)
    preictal_t1 = float(seizure_onset)
    preictal = raw_pick.copy().crop(tmin=preictal_t0, tmax=preictal_t1).get_data()  # ch x samples
    n_pre_samples = preictal.shape[1]

    ch_results = {}

    # 1) 基线特征分布（每个 channel 按滑窗算很多 baseline 特征，记录 mean/std & percentile）
    baseline_feats_per_ch = {}
    feature_names = list(compute_features_for_segment(np.zeros(10), sfreq).keys())
    band_features = [f for f in feature_names if '_power' in f]
    n_features = len(compute_features_for_segment(np.zeros(10), sfreq))  # Get number of features
    for ch_idx, ch in enumerate(pick_chs):
        sig = baseline_data[ch_idx]
        feats = []
        idxs = sliding_windows_indices(len(sig), win_samples, step_samples)
        for start in idxs:
            seg = sig[start:start + win_samples]
            f = compute_features_for_segment(seg, sfreq)
            feats.append(list(f.values()))
        feats = np.array(feats) if len(feats) > 0 else np.zeros((1, n_features))
        mean = feats.mean(axis=0)
        std = feats.std(axis=0, ddof=1)
        std[std < 1e-6] = 1e-6
        baseline_feats_per_ch[ch] = {'mean': mean, 'std': std, 'raw_feats': feats}

    # 2) 在 pre-ictal 滑窗计算特征并转为 z-score
    windows = sliding_windows_indices(n_pre_samples, win_samples, step_samples)
    time_starts = [preictal_t0 + (s / sfreq) for s in windows]
    per_ch_z = {ch: [] for ch in pick_chs}
    per_ch_feat_values = {ch: [] for ch in pick_chs}
    per_ch_z_by_feature = {ch: {feat: [] for feat in feature_names} for ch in pick_chs}

    for ch_idx, ch in enumerate(pick_chs):
        sig = preictal[ch_idx]
        mean = baseline_feats_per_ch[ch]['mean']
        std = baseline_feats_per_ch[ch]['std']
        for start in windows:
            seg = sig[start:start + win_samples]
            f = compute_features_for_segment(seg, sfreq)
            vec = np.array(list(f.values()))
            zvec = (vec - mean) / std
            per_ch_z[ch].append(zvec)
            per_ch_feat_values[ch].append(vec)
            for fid, feat in enumerate(feature_names):
                per_ch_z_by_feature[ch][feat].append(zvec[fid])
        per_ch_z[ch] = np.array(per_ch_z[ch])  # n_windows x n_features
        per_ch_feat_values[ch] = np.array(per_ch_feat_values[ch])
        for feat in feature_names:
            per_ch_z_by_feature[ch][feat] = np.array(per_ch_z_by_feature[ch][feat])

    # 3) 判定每个窗口是否异常（任一 feature z > z_thr）
    per_ch_win_flag = {}
    per_ch_maxz = {}
    for ch in pick_chs:
        zmat = per_ch_z[ch]  # n_windows x features
        maxz = np.max(zmat, axis=1)  # per-window 最大 z
        flag = maxz > z_thr
        per_ch_win_flag[ch] = flag
        per_ch_maxz[ch] = maxz

    # 4) 将连续的异常窗转为时间区间（并要求至少 min_consec 个连续窗）
    results = []
    per_ch_pval = []
    per_ch_interval_z_avgs = {ch: [] for ch in pick_chs}  # 新增：per-channel z avgs
    for ch in pick_chs:
        flags = per_ch_win_flag[ch]
        maxz = per_ch_maxz[ch]
        intervals = []
        n_win = len(flags)
        i = 0
        while i < n_win:
            if flags[i]:
                j = i + 1
                while j < n_win and flags[j]:
                    j += 1
                length = j - i
                if length >= min_consec:
                    start_time = time_starts[i]
                    end_time = time_starts[j - 1] + win_sec
                    intervals.append((start_time, end_time, float(maxz[i:j].mean())))
                i = j
            else:
                i += 1
        intervals_sorted = sorted([(int((s - preictal_t0) * sfreq), int((e - preictal_t0) * sfreq), score) for (s, e, score) in intervals], key=lambda x: x[0])
        intervals_sec = []
        if intervals_sorted:
            simple_intervals = [(samp_s, samp_e) for (samp_s, samp_e, _) in intervals_sorted]
            merged = merge_close_intervals(simple_intervals, max_gap=merge_gap_samples)
            for (ms, me) in merged:
                contrib_scores = []
                contrib_z = {feat: [] for feat in feature_names}
                for idx, (s, e, sc) in enumerate(intervals):
                    samp_s = int((s - preictal_t0) * sfreq)
                    samp_e = int((e - preictal_t0) * sfreq)
                    if not (samp_e <= ms or samp_s >= me):
                        contrib_scores.append(sc)
                        win_start = windows[idx]  # 注意：这里的 windows[idx] 假设 idx 对应窗口索引；如果不准，可调整为实际窗口范围
                        win_end = win_start + length  # length 来自上方；确保正确
                        for feat in feature_names:
                            feat_z = per_ch_z_by_feature[ch][feat][win_start:win_end]
                            contrib_z[feat].extend(feat_z)
                mean_score = float(np.mean(contrib_scores)) if contrib_scores else 0.0
                intervals_sec.append((preictal_t0 + ms / sfreq, preictal_t0 + me / sfreq, mean_score))
                avg_z = {feat: np.mean(vals) if vals else 0.0 for feat, vals in contrib_z.items()}
                per_ch_interval_z_avgs[ch].append(avg_z)  # 填充 per-channel 列表
        if len(intervals_sec) > 0:
            earliest_time = min([s for (s, e, sc) in intervals_sec])
            test_stat = max(maxz)  # Use peak max z as test statistic for permutation
        else:
            earliest_time = None
            test_stat = 0.0
        # Permutation test for p-value
        all_feats = np.vstack((baseline_feats_per_ch[ch]['raw_feats'], per_ch_feat_values[ch]))
        labels = np.array([0] * len(baseline_feats_per_ch[ch]['raw_feats']) + [1] * len(per_ch_feat_values[ch]))
        null_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(labels)
            null_baseline = all_feats[labels == 0]
            null_preictal = all_feats[labels == 1]
            null_mean = null_baseline.mean(axis=0)
            null_std = null_baseline.std(axis=0, ddof=1)
            null_std[null_std < 1e-6] = 1e-6
            null_z = (null_preictal - null_mean) / null_std
            null_maxz = np.max(null_z, axis=1)
            null_stat = np.max(null_maxz) if len(null_maxz) > 0 else 0.0
            null_stats.append(null_stat)
        pval = compute_permutation_pvalue(test_stat, np.array(null_stats))
        per_ch_pval.append(pval)

        ch_results[ch] = {
            'intervals': [{'start_time': float(s), 'end_time': float(e), 'score': float(sc)} for (s,e,sc) in intervals_sec],
            'has_any': len(intervals_sec) > 0,
            'peak_z': float(np.max(per_ch_maxz[ch])) if len(per_ch_maxz[ch])>0 else 0.0,
            'earliest_time': earliest_time,
            'p_value': float(pval),
            'interval_z_avgs': per_ch_interval_z_avgs[ch]  # 新增：存储 per-channel z avgs
        }

    # 5) 多重比较校正（Benjamini-Hochberg），用上面生成的 pseudo-pvals
    #    说明：这里我们没有用确切 p 计算（需 bootstrap 或 permutation），采用了一个保守的伪 p 值转化。
    pvals = per_ch_pval
    signif_mask = benjamini_hochberg(pvals, alpha=fdr_alpha)
    final_list = []
    for i, ch in enumerate(pick_chs):
        entry = ch_results[ch]
        is_signif = bool(signif_mask[i]) and entry['has_any']
        if is_signif:
            final_list.append({
                'channel': ch,
                'intervals': entry['intervals'],
                'peak_z': entry['peak_z'],
                'earliest_time': entry['earliest_time'],
                'p_value': entry['p_value']
            })

    final_list_sorted = sorted(final_list, key=lambda x: x['earliest_time'] if x['earliest_time'] is not None else 1e9)

    output = {
        'input_file': input_file or (str(raw.filenames[0]) if hasattr(raw, 'filenames') and raw.filenames else None),
        'seizure_onset': float(seizure_onset),
        'preictal_window': {'start': preictal_t0, 'end': preictal_t1},
        'parameters': {
            'win_sec': win_sec, 'step_sec': step_sec, 'z_thr': z_thr,
            'min_consec': min_consec, 'merge_gap_sec': merge_gap_sec, 'fdr_alpha': fdr_alpha,
            'n_permutations': n_permutations
        },
        'results': final_list_sorted,
        'all_channels': pick_chs,
        'per_channel_peak_z': {ch: ch_results[ch]['peak_z'] for ch in pick_chs}
    }

    json_path = os.path.join(out_dir, f"{out_prefix}_preictal_channels.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved JSON results to: {json_path}")

    # Additional CSV output
    csv_data = []
    for entry in final_list_sorted:
        ch = entry['channel']
        interval_z_avgs = ch_results[ch].get('interval_z_avgs', [])  # 使用 per-channel 列表
        for int_idx, interval in enumerate(entry['intervals']):
            start, end = interval['start_time'], interval['end_time']
            if int_idx < len(interval_z_avgs):
                avg_z = interval_z_avgs[int_idx]
                abnormal_bands = ','.join([feat.split('_power')[0] for feat in band_features if avg_z.get(feat, 0.0) > z_thr])
            else:
                abnormal_bands = ''  # 边界情况：无 z 数据
            csv_data.append([ch, start, end, abnormal_bands])

    csv_path = os.path.join(out_dir, f"{out_prefix}_abnormal_intervals.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['channel', 'start_time', 'end_time', 'abnormal_bands'])
        writer.writerows(csv_data)
    print(f"Saved CSV to: {csv_path}")

    if do_plot:
        try:
            # Interactive heatmap with Plotly
            all_maxz = np.array([per_ch_maxz[ch] for ch in pick_chs])  # n_ch x n_windows
            times = np.array(time_starts)
            fig = go.Figure(data=go.Heatmap(
                z=all_maxz,
                x=times,
                y=pick_chs,
                colorscale='Viridis'
            ))
            fig.update_layout(
                title='Per-window max z-score (channels x time)',
                xaxis_title='Time (s)',
                yaxis_title='Channels'
            )
            heatpath = os.path.join(out_dir, f"{out_prefix}_zheatmap.html")
            fig.write_html(heatpath)
            print(f"Saved interactive heatmap to: {heatpath}")

            # Per-significant-channel interactive time series
            for entry in final_list_sorted:
                ch = entry['channel']
                tz = per_ch_maxz[ch]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time_starts, y=tz, mode='lines', name='max z per window'))
                fig.add_hline(y=z_thr, line_dash="dash", line_color="red", annotation_text="z threshold")
                for (s, e, sc) in entry['intervals']:
                    fig.add_vrect(x0=s, x1=e, fillcolor="orange", opacity=0.3, line_width=0)
                fig.update_layout(
                    title=f"Channel {ch} — max z over time (highlighted detected intervals)",
                    xaxis_title='time (s)',
                    yaxis_title='max z'
                )
                figpath = os.path.join(out_dir, f"{out_prefix}_{ch}_timeseries.html")
                fig.write_html(figpath)
            print(f"Saved per-channel interactive time series to HTML files in {out_dir}")
        except Exception as e:
            print("Plotting failed:", e)

    return output


# ---------------------------
# CLI 入口
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Detect earliest pre-ictal significant channels")
    p.add_argument('--input', required=True, help="Input EEGLAB .set file")
    p.add_argument('--seizure_onset', type=float, required=True, help="Seizure onset time in seconds (absolute time in the record)")
    p.add_argument('--baseline', nargs=2, type=float, action='append', help="Baseline interval tmin tmax (can provide multiple). Example: --baseline 0 300 --baseline 600 900")
    p.add_argument('--preictal_minutes', type=float, default=30.0, help="Minutes before seizure onset to analyze")
    p.add_argument('--channels', nargs='+', default=DEFAULT_CHANNELS, help="Channel list to analyze")
    p.add_argument('--win_sec', type=float, default=2.0)
    p.add_argument('--step_sec', type=float, default=1.0)
    p.add_argument('--z_thr', type=float, default=2.0)
    p.add_argument('--min_consec', type=int, default=2)
    p.add_argument('--merge_gap_sec', type=float, default=2.0)
    p.add_argument('--fdr_alpha', type=float, default=0.05)
    p.add_argument('--out_prefix', type=str, default='result')
    p.add_argument('--out_dir', type=str, default='.', help="Output directory")
    p.add_argument('--no_plot', action='store_true', help="Disable plotting")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading .set file:", args.input)
    raw = read_raw_eeglab(args.input, preload=True)

    print("Filtering 0.5-70 Hz ...")
    raw.load_data()
    raw.filter(0.5, 70., picks='all', fir_design='firwin', verbose='ERROR')
    if raw.info['sfreq'] >= 100:
        try:
            raw.notch_filter(50., picks='all', verbose='ERROR')
        except Exception:
            pass

    if args.baseline is None:
        raise ValueError("You must specify at least one --baseline tmin tmax interval.")
    baseline_intervals = [(float(a[0]), float(a[1])) for a in args.baseline]

    out = detect_preictal_channels(
        raw=raw,
        seizure_onset=args.seizure_onset,
        baseline_intervals=baseline_intervals,
        preictal_minutes=args.preictal_minutes,
        channels=args.channels,
        win_sec=args.win_sec,
        step_sec=args.step_sec,
        z_thr=args.z_thr,
        min_consec=args.min_consec,
        merge_gap_sec=args.merge_gap_sec,
        fdr_alpha=args.fdr_alpha,
        do_plot=(not args.no_plot),
        out_prefix=args.out_prefix,
        input_file=args.input,
        out_dir=args.out_dir
    )
    print("Done. Results saved to JSON and CSV.")


def gui_main():
    def run_analysis():
        try:
            input_file = entry_input.get()
            seizure_onset = float(entry_seizure.get())
            preictal_minutes = float(entry_preictal.get())
            win_sec = float(entry_win.get())
            step_sec = float(entry_step.get())
            z_thr = float(entry_zthr.get())
            min_consec = int(entry_minconsec.get())
            merge_gap_sec = float(entry_mergegap.get())
            fdr_alpha = float(entry_fdralpha.get())
            out_prefix = entry_outprefix.get()
            no_plot = var_noplot.get()
            out_dir = entry_outdir.get() or '.'

            baseline_intervals = []
            for bl in baseline_list.get(0, tk.END):
                tmin, tmax = map(float, bl.split('-'))
                baseline_intervals.append((tmin, tmax))

            channels = [ch.strip() for ch in entry_channels.get().split(',')]

            raw = read_raw_eeglab(input_file, preload=True, verbose='ERROR')
            raw.load_data()
            raw.filter(0.5, 70., picks='all', fir_design='firwin', verbose='ERROR')
            if raw.info['sfreq'] >= 100:
                try:
                    raw.notch_filter(50., picks='all', verbose='ERROR')
                except Exception:
                    pass

            out = detect_preictal_channels(
                raw=raw,
                seizure_onset=seizure_onset,
                baseline_intervals=baseline_intervals,
                preictal_minutes=preictal_minutes,
                channels=channels,
                win_sec=win_sec,
                step_sec=step_sec,
                z_thr=z_thr,
                min_consec=min_consec,
                merge_gap_sec=merge_gap_sec,
                fdr_alpha=fdr_alpha,
                do_plot=(not no_plot),
                out_prefix=out_prefix,
                input_file=input_file,
                out_dir=out_dir
            )
            result_label.config(text="Analysis completed. Check output files.")
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            result_label.config(text=f"Error: {str(e)}")

    def add_baseline():
        tmin = entry_bl_tmin.get()
        tmax = entry_bl_tmax.get()
        if tmin and tmax:
            baseline_list.insert(tk.END, f"{tmin}-{tmax}")
            entry_bl_tmin.delete(0, tk.END)
            entry_bl_tmax.delete(0, tk.END)

    def browse_file():
        filename = filedialog.askopenfilename()
        entry_input.insert(0, filename)

    def browse_dir():
        dirname = filedialog.askdirectory()
        entry_outdir.insert(0, dirname)

    root = tk.Tk()
    root.title("Preictal Channel Detector")

    ttk.Label(root, text="Input .set File:").grid(row=0, column=0)
    entry_input = ttk.Entry(root, width=50)
    entry_input.grid(row=0, column=1)
    ttk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2)

    ttk.Label(root, text="Seizure Onset (s):").grid(row=1, column=0)
    entry_seizure = ttk.Entry(root)
    entry_seizure.grid(row=1, column=1)
    entry_seizure.insert(0, "3600")

    ttk.Label(root, text="Preictal Minutes:").grid(row=2, column=0)
    entry_preictal = ttk.Entry(root)
    entry_preictal.grid(row=2, column=1)
    entry_preictal.insert(0, "30")

    ttk.Label(root, text="Channels (comma sep):").grid(row=3, column=0)
    entry_channels = ttk.Entry(root, width=50)
    entry_channels.grid(row=3, column=1)
    entry_channels.insert(0, ','.join(DEFAULT_CHANNELS))

    ttk.Label(root, text="Window Sec:").grid(row=4, column=0)
    entry_win = ttk.Entry(root)
    entry_win.grid(row=4, column=1)
    entry_win.insert(0, "2.0")

    ttk.Label(root, text="Step Sec:").grid(row=5, column=0)
    entry_step = ttk.Entry(root)
    entry_step.grid(row=5, column=1)
    entry_step.insert(0, "1.0")

    ttk.Label(root, text="Z Threshold:").grid(row=6, column=0)
    entry_zthr = ttk.Entry(root)
    entry_zthr.grid(row=6, column=1)
    entry_zthr.insert(0, "2.0")

    ttk.Label(root, text="Min Consecutive:").grid(row=7, column=0)
    entry_minconsec = ttk.Entry(root)
    entry_minconsec.grid(row=7, column=1)
    entry_minconsec.insert(0, "2")

    ttk.Label(root, text="Merge Gap Sec:").grid(row=8, column=0)
    entry_mergegap = ttk.Entry(root)
    entry_mergegap.grid(row=8, column=1)
    entry_mergegap.insert(0, "2.0")

    ttk.Label(root, text="FDR Alpha:").grid(row=9, column=0)
    entry_fdralpha = ttk.Entry(root)
    entry_fdralpha.grid(row=9, column=1)
    entry_fdralpha.insert(0, "0.05")

    ttk.Label(root, text="Output Prefix:").grid(row=10, column=0)
    entry_outprefix = ttk.Entry(root)
    entry_outprefix.grid(row=10, column=1)
    entry_outprefix.insert(0, "result")

    var_noplot = tk.BooleanVar()
    ttk.Checkbutton(root, text="No Plot", variable=var_noplot).grid(row=11, column=0)

    ttk.Label(root, text="Output Directory:").grid(row=12, column=0)
    entry_outdir = ttk.Entry(root, width=50)
    entry_outdir.grid(row=12, column=1)
    entry_outdir.insert(0, ".")
    ttk.Button(root, text="Browse", command=browse_dir).grid(row=12, column=2)

    ttk.Label(root, text="Baseline Intervals:").grid(row=13, column=0)
    baseline_list = tk.Listbox(root, height=5)
    baseline_list.grid(row=13, column=1)

    ttk.Label(root, text="Add Baseline (tmin tmax):").grid(row=14, column=0)
    entry_bl_tmin = ttk.Entry(root, width=10)
    entry_bl_tmin.grid(row=14, column=1, sticky='w')
    entry_bl_tmax = ttk.Entry(root, width=10)
    entry_bl_tmax.grid(row=14, column=1, sticky='e')
    ttk.Button(root, text="Add", command=add_baseline).grid(row=14, column=2)

    ttk.Button(root, text="Run Analysis", command=run_analysis).grid(row=15, column=1)

    result_label = ttk.Label(root, text="")
    result_label.grid(row=16, column=1)

    root.mainloop()

if __name__ == "__main__":
    gui_main()
