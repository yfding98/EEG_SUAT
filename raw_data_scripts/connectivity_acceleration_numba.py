#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Numba JIT 编译加速 - 方案2
需要安装: pip install numba
"""

import numpy as np
from scipy.stats import zscore
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Transfer Entropy - Numba JIT加速版本
# ============================================================================
@jit(nopython=True, parallel=True, fastmath=True)
def _discretize_data(data, n_bins):
    """使用Numba加速数据离散化"""
    n_channels, n_samples = data.shape
    discretized = np.zeros((n_channels, n_samples), dtype=np.int32)
    
    for ch in range(n_channels):
        min_val = np.min(data[ch])
        max_val = np.max(data[ch])
        bins = np.linspace(min_val, max_val, n_bins)
        
        for i in range(n_samples):
            for b in range(n_bins - 1):
                if data[ch, i] >= bins[b] and data[ch, i] < bins[b + 1]:
                    discretized[ch, i] = b
                    break
            if data[ch, i] >= bins[-1]:
                discretized[ch, i] = n_bins - 1
                
    return discretized


@jit(nopython=True, fastmath=True)
def _compute_entropy_1d(data, n_bins):
    """计算1维离散数据的熵"""
    counts = np.zeros(n_bins, dtype=np.float64)
    n = len(data)
    
    for i in range(n):
        counts[data[i]] += 1
    
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / n
            entropy -= p * np.log2(p)
    
    return entropy


@jit(nopython=True, fastmath=True)
def _compute_entropy_2d(data1, data2, n_bins):
    """计算2维离散数据的联合熵"""
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    n = len(data1)
    
    for i in range(n):
        counts[data1[i], data2[i]] += 1
    
    entropy = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if counts[i, j] > 0:
                p = counts[i, j] / n
                entropy -= p * np.log2(p)
    
    return entropy


@jit(nopython=True, fastmath=True)
def _compute_entropy_3d(data1, data2, data3, n_bins):
    """计算3维离散数据的联合熵"""
    counts = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    n = len(data1)
    
    for i in range(n):
        counts[data1[i], data2[i], data3[i]] += 1
    
    entropy = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(n_bins):
                if counts[i, j, k] > 0:
                    p = counts[i, j, k] / n
                    entropy -= p * np.log2(p)
    
    return entropy


@jit(nopython=True, parallel=True)
def _compute_te_matrix_numba(discretized, n_channels, n_bins, lag):
    """使用Numba并行计算传递熵矩阵"""
    te_matrix = np.zeros((n_channels, n_channels), dtype=np.float64)
    
    for i in prange(n_channels):
        for j in range(n_channels):
            if i == j:
                continue
            
            X_past = discretized[i, :-lag]
            Y_past = discretized[j, :-lag]
            Y_present = discretized[j, lag:]
            
            # I(Y_present; X_past | Y_past) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
            H_XZ = _compute_entropy_2d(Y_present, X_past, n_bins)
            H_YZ = _compute_entropy_2d(Y_present, Y_past, n_bins)
            H_XYZ = _compute_entropy_3d(Y_present, X_past, Y_past, n_bins)
            H_Z = _compute_entropy_1d(Y_past, n_bins)
            
            te = H_XZ + H_YZ - H_XYZ - H_Z
            te_matrix[i, j] = max(0.0, te)
    
    return te_matrix


def compute_transfer_entropy_pairwise_numba(data, n_bins=10, lag=1):
    """
    使用Numba JIT加速的传递熵计算
    
    参数:
        data: (n_channels, n_samples)
        n_bins: 离散化bins
        lag: 时间延迟
    """
    n_channels = data.shape[0]
    
    # 标准化
    data_normalized = zscore(data, axis=1)
    
    # 离散化（Numba加速）
    discretized = _discretize_data(data_normalized.astype(np.float64), n_bins)
    
    # 计算传递熵（Numba并行）
    te_matrix = _compute_te_matrix_numba(discretized, n_channels, n_bins, lag)
    
    return te_matrix


# ============================================================================
# Granger Causality - 多线程版本（Numba不支持statsmodels）
# ============================================================================
from concurrent.futures import ThreadPoolExecutor
import os


def _compute_gc_single(args):
    """计算单个Granger因果性"""
    i, j, data, max_lag = args
    if i == j:
        return (i, j, 0)
    
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        test_data = np.column_stack([data[j], data[i]])
        result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
        f_values = [result[lag][0]['ssr_ftest'][0] for lag in range(1, max_lag+1)]
        return (i, j, np.mean(f_values))
    except:
        return (i, j, 0)


def compute_granger_causality_pairwise_threaded(data, sfreq, max_lag=None, n_workers=None):
    """
    使用线程池加速Granger因果性计算
    
    参数:
        data: (n_channels, n_samples)
        sfreq: 采样率
        max_lag: 最大滞后
        n_workers: 线程数，None表示使用CPU核心数
    """
    n_channels = data.shape[0]
    
    if max_lag is None:
        max_lag = int(sfreq / 10)
    max_lag = max(1, min(max_lag, 20))
    
    if n_workers is None:
        n_workers = os.cpu_count()
    
    # 准备任务
    tasks = [(i, j, data, max_lag) for i in range(n_channels) for j in range(n_channels)]
    
    # 多线程执行
    gc_matrix = np.zeros((n_channels, n_channels))
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(_compute_gc_single, tasks)
        for i, j, value in results:
            gc_matrix[i, j] = value
    
    return gc_matrix


# ============================================================================
# 替换原函数的代码片段
# ============================================================================
"""
在 extract_connectivity_features.py 中替换：

1. 在文件开头添加导入：
from connectivity_acceleration_numba import (
    compute_granger_causality_pairwise_threaded,
    compute_transfer_entropy_pairwise_numba
)

2. 在 extract_all_features 函数中替换：

# 原代码：
features['granger_causality'] = compute_granger_causality_pairwise(data, sfreq)
features['transfer_entropy'] = compute_transfer_entropy_pairwise(data)

# 新代码：
features['granger_causality'] = compute_granger_causality_pairwise_threaded(data, sfreq)
features['transfer_entropy'] = compute_transfer_entropy_pairwise_numba(data)
"""

