#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detect the EARLIEST anomalous channels in PRE-ICTAL period
Author : Your Name
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from sklearn.preprocessing import StandardScaler

# ---------- 参数区 ----------
RAW_FILE        = r'E:\DataSet\EEG\EEG dataset_SUAT_processed\头皮数据-6例\江仁坤\SZ2_postICA.set'
PREICTAL_MIN    = 10              # 只分析发作前 N 分钟
WINDOW_S        = 2                # 窗长
OVERLAP         = 0.5
S_FREQ          = 250              # 统一重采样
DEVIATION_TH    = 3.0              # 偏离阈值（单位 σ）
CUSUM_TH        = 3.0              # CUSUM 控制限
CUSUM_DRIFT     = 0.5

# 关注通道（顺序无关，脚本会按文件实际顺序自动子集）
TARGET_CH = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
             'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz','Sph-L','Sph-R']

class Cusum1D:
    def __init__(self, threshold=3.0, drift=0.5):
        self.threshold = threshold
        self.drift = drift
        self.S_pos = 0.0
        self.mu0 = None

    def update(self, x):
        if self.mu0 is None:
            self.mu0 = x
        else:
            self.mu0 = 0.995 * self.mu0 + 0.005 * x
        self.S_pos = max(0., self.S_pos + x - self.mu0 - self.drift)
        if self.S_pos > self.threshold:
            self.S_pos = 0.
            return True
        return False


def cusum_intervals(score_1ch, threshold=3.0, drift=0.5,
                    min_len_s=4, sfreq=256, win_s=8, overlap=0.5):
    """返回最早一个异常窗 [(t0_sec, t1_sec)]，无异常返回 []"""
    detector = Cusum1D(threshold=threshold, drift=drift)
    win_step = int(win_s * (1 - overlap) * sfreq)
    change_pts = []
    for i, val in enumerate(score_1ch):
        if detector.update(val):
            change_pts.append(i)

    # 转成秒级区间
    intervals = [(i * win_s * (1 - overlap),
                  i * win_s * (1 - overlap) + win_s) for i in change_pts]
    # 合并邻近 & 删太短
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    min_len = min_len_s
    merged = [(s, e) for s, e in merged if e - s >= min_len]
    return [merged[0]] if merged else []



# ---------- 1. 读取 EEGLAB .set ----------
def load_preictal(set_path, min_before=120, pick_ch=TARGET_CH, sfreq=S_FREQ):
    """
    输入：经过 ICA/滤波后的 .set 文件（可由 eeglab 导出）
    返回：pre-ictal 段数据 (n_ch, n_sample), sfreq, ch_names
    """
    from mne.io import read_raw_eeglab

    # 1) 载入 .set（同时自动读取同目录 .fdt 若存在）
    raw = read_raw_eeglab(set_path, preload=True, verbose=False)

    # 2）只保留我们关心的 21 导
    raw.pick_channels([ch for ch in pick_ch if ch in raw.ch_names])

    # 3）统一重采样
    raw.resample(sfreq, verbose=False)

    # 4）截取最后 min_before 分钟
    len_pts = int(min_before * 60 * sfreq)  # 整数采样点
    n_total = raw.n_times  # 总采样点
    data, times = raw.get_data(
        return_times=True,
        start=max(0, n_total - len_pts),  # 整数
        stop=n_total  # 整数
    )
    return data, sfreq, raw.ch_names

def sliding_window(data, win_s, overlap, sfreq):
    """返回 (n_win, n_ch, win_len)"""
    win_n = int(win_s * sfreq)
    step  = int(win_n * (1 - overlap))
    n_ch, n_sam = data.shape
    n_win = (n_sam - win_n) // step + 1
    out = np.empty((n_win, n_ch, win_n), dtype=np.float32)
    for i in range(n_win):
        out[i] = data[:, i*step:i*step+win_n]
    return out

def extract_features(sw):
    """最简单但有效的 5 维特征：均值、标准差、偏度、峰度、频谱能量比"""
    from scipy.stats import skew, kurtosis
    n_win, n_ch, _ = sw.shape
    feat = np.empty((n_win, n_ch, 5))
    for w in range(n_win):
        for c in range(n_ch):
            x = sw[w, c]
            feat[w, c, 0] = np.mean(x)
            feat[w, c, 1] = np.std(x)
            feat[w, c, 2] = skew(x)
            feat[w, c, 3] = kurtosis(x)
            # 能量比 0-30 Hz / 30-70 Hz
            psd = np.abs(np.fft.rfft(x))**2
            freqs = np.fft.rfftfreq(len(x), d=1./S_FREQ)
            feat[w, c, 4] = (psd[freqs<=30].sum() + 1e-6) / (psd[freqs>30].sum() + 1e-6)
    # 通道内标准化
    shape_orig = feat.shape
    feat = feat.reshape(n_win, -1)
    feat = StandardScaler().fit_transform(feat)
    feat = feat.reshape(shape_orig)
    return feat   # (n_win, n_ch, 5)

def channel_anomaly_score(feat):
    """无监督：用通道-通道余弦距离偏离均值的程度"""
    n_win, n_ch, _ = feat.shape
    score = np.zeros((n_win, n_ch))
    for w in range(n_win):
        win = feat[w]                    # (n_ch, 5)
        win = win / (np.linalg.norm(win, axis=1, keepdims=True) + 1e-6)
        # 计算通道间距离矩阵
        dist = 1 - win @ win.T           # cosine distance
        # 每个通道对其它通道的平均距离
        avg_dist = dist.mean(axis=1)
        # 偏离度
        score[w] = (avg_dist - avg_dist.mean()) / (avg_dist.std() + 1e-6)
    return score  # (n_win, n_ch)


# ---------- 3. 主流程 ----------
def main():
    print('[1] 加载 pre-ictal 数据 ...')
    data, sfreq, ch_names = load_preictal(RAW_FILE, PREICTAL_MIN, TARGET_CH, S_FREQ)
    print('    通道顺序:', ch_names)
    print('    数据形状:', data.shape)

    print('[2] 滑窗提取特征 ...')
    sw = sliding_window(data, WINDOW_S, OVERLAP, sfreq)
    feat = extract_features(sw)

    print('[3] 计算通道异常得分 ...')
    score = channel_anomaly_score(feat)   # (n_win, n_ch)
    # 转置成 (n_ch, n_win) 方便后面逐通道处理
    score = score.T

    print('[4] 逐通道 CUSUM 找最早异常窗 ...')
    results = []
    for c, ch in enumerate(ch_names):
        intervals = cusum_intervals(score[c],
                                    threshold=CUSUM_TH,
                                    drift=CUSUM_DRIFT,
                                    min_len_s=4,
                                    win_s=WINDOW_S,
                                    overlap=OVERLAP)
        if intervals:
            # 只取最早一个窗
            t0, t1 = intervals[0]
            results.append({'channel': ch,
                            'start_sec': float(t0),
                            'end_sec': float(t1),
                            'max_dev': float(score[c].max())})
    # 按起始时间排序
    results = sorted(results, key=lambda x: x['start_sec'])
    print('    检测到显著通道:', [r['channel'] for r in results])

    print('[5] 保存 json & 绘图 ...')
    out_json = RAW_FILE.replace('.edf', '_salient_channels.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print('    结果已写入:', out_json)

    # 简单可视化：灰度图 + 红框
    plt.figure(figsize=(14, 8))
    n_ch = len(ch_names)
    for c, ch in enumerate(ch_names):
        plt.subplot(n_ch, 1, c+1)
        t = np.arange(data.shape[1]) / sfreq
        plt.plot(t, data[c], 'k', linewidth=0.5)
        plt.ylabel(ch)
        plt.xlim(0, t[-1])
        plt.yticks([]); plt.gca().spines['top'].set_visible(False); plt.gca().spines['right'].set_visible(False)
        # 画异常框
        for r in results:
            if r['channel']==ch:
                s, e = r['start_sec'], r['end_sec']
                plt.axvspan(s, e, color='red', alpha=0.3)
                break
    plt.xlabel('time (sec)')
    plt.suptitle('Pre-ictal (%d min) earliest anomalous windows' % PREICTAL_MIN)
    plt.tight_layout()
    fig_name = RAW_FILE.replace('.edf', '_salient.pdf')
    plt.savefig(fig_name); print('    概览图已保存:', fig_name)

if __name__ == '__main__':
    main()