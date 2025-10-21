#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 concurrent.futures 多进程加速 - 方案3
无需额外依赖，使用Python标准库
"""

import numpy as np
from scipy.stats import zscore
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Granger Causality - 多进程版本
# ============================================================================
def _gc_worker(args):
    """单个通道对的Granger因果性计算（用于多进程）"""
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


def compute_granger_causality_pairwise_multiprocess(data, sfreq, max_lag=None, n_workers=None):
    """
    使用多进程加速Granger因果性计算
    
    参数:
        data: (n_channels, n_samples)
        sfreq: 采样率
        max_lag: 最大滞后
        n_workers: 进程数，None表示使用CPU核心数
    """
    n_channels = data.shape[0]
    
    if max_lag is None:
        max_lag = int(sfreq / 10)
    max_lag = max(1, min(max_lag, 20))
    
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)  # 保留一个核心
    
    # 准备任务
    tasks = [(i, j, data, max_lag) for i in range(n_channels) for j in range(n_channels)]
    
    # 多进程执行
    gc_matrix = np.zeros((n_channels, n_channels))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_gc_worker, task): task for task in tasks}
        
        for future in as_completed(futures):
            i, j, value = future.result()
            gc_matrix[i, j] = value
    
    return gc_matrix


# ============================================================================
# Transfer Entropy - 多进程版本
# ============================================================================
def _calculate_entropy_discrete(data, n_bins):
    """计算离散数据的熵"""
    try:
        hist, _ = np.histogramdd(data, bins=n_bins)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return -np.sum(prob * np.log2(prob))
    except:
        return 0


def _calculate_cmi(X, Y, Z, n_bins):
    """计算条件互信息"""
    try:
        XYZ = np.column_stack([X, Y, Z])
        H_XYZ = _calculate_entropy_discrete(XYZ, n_bins**3)
        H_XZ = _calculate_entropy_discrete(np.column_stack([X, Z]), n_bins**2)
        H_YZ = _calculate_entropy_discrete(np.column_stack([Y, Z]), n_bins**2)
        H_Z = _calculate_entropy_discrete(Z.reshape(-1, 1), n_bins)
        return H_XZ + H_YZ - H_XYZ - H_Z
    except:
        return 0


def _te_worker(args):
    """单个通道对的传递熵计算（用于多进程）"""
    i, j, data_normalized, n_bins, lag = args
    if i == j:
        return (i, j, 0)
    
    try:
        X = data_normalized[i]
        Y = data_normalized[j]
        
        X_bins = np.digitize(X, bins=np.linspace(X.min(), X.max(), n_bins))
        Y_bins = np.digitize(Y, bins=np.linspace(Y.min(), Y.max(), n_bins))
        
        Y_present = Y_bins[lag:]
        Y_past = Y_bins[:-lag]
        X_past = X_bins[:-lag]
        
        te = _calculate_cmi(Y_present, X_past, Y_past, n_bins)
        return (i, j, max(0, te))
    except:
        return (i, j, 0)


def compute_transfer_entropy_pairwise_multiprocess(data, n_bins=10, lag=1, n_workers=None):
    """
    使用多进程加速传递熵计算
    
    参数:
        data: (n_channels, n_samples)
        n_bins: 离散化bins
        lag: 时间延迟
        n_workers: 进程数
    """
    n_channels = data.shape[0]
    data_normalized = zscore(data, axis=1)
    
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)
    
    # 准备任务
    tasks = [(i, j, data_normalized, n_bins, lag) for i in range(n_channels) for j in range(n_channels)]
    
    # 多进程执行
    te_matrix = np.zeros((n_channels, n_channels))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_te_worker, task): task for task in tasks}
        
        for future in as_completed(futures):
            i, j, value = future.result()
            te_matrix[i, j] = value
    
    return te_matrix


# ============================================================================
# 批量处理版本（更高效）- 减少进程创建开销
# ============================================================================
def _gc_batch_worker(batch):
    """批量处理多个通道对的Granger因果性"""
    results = []
    for args in batch:
        results.append(_gc_worker(args))
    return results


def compute_granger_causality_pairwise_batch(data, sfreq, max_lag=None, n_workers=None, batch_size=10):
    """
    批量多进程加速Granger因果性（减少进程创建开销）
    
    参数:
        batch_size: 每个进程处理的任务数
    """
    n_channels = data.shape[0]
    
    if max_lag is None:
        max_lag = int(sfreq / 10)
    max_lag = max(1, min(max_lag, 20))
    
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)
    
    # 准备任务并分批
    all_tasks = [(i, j, data, max_lag) for i in range(n_channels) for j in range(n_channels)]
    batches = [all_tasks[i:i+batch_size] for i in range(0, len(all_tasks), batch_size)]
    
    # 多进程执行
    gc_matrix = np.zeros((n_channels, n_channels))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_gc_batch_worker, batch): batch for batch in batches}
        
        for future in as_completed(futures):
            batch_results = future.result()
            for i, j, value in batch_results:
                gc_matrix[i, j] = value
    
    return gc_matrix


def _te_batch_worker(batch):
    """批量处理多个通道对的传递熵"""
    results = []
    for args in batch:
        results.append(_te_worker(args))
    return results


def compute_transfer_entropy_pairwise_batch(data, n_bins=10, lag=1, n_workers=None, batch_size=10):
    """
    批量多进程加速传递熵（减少进程创建开销）
    
    参数:
        batch_size: 每个进程处理的任务数
    """
    n_channels = data.shape[0]
    data_normalized = zscore(data, axis=1)
    
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)
    
    # 准备任务并分批
    all_tasks = [(i, j, data_normalized, n_bins, lag) for i in range(n_channels) for j in range(n_channels)]
    batches = [all_tasks[i:i+batch_size] for i in range(0, len(all_tasks), batch_size)]
    
    # 多进程执行
    te_matrix = np.zeros((n_channels, n_channels))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_te_batch_worker, batch): batch for batch in batches}
        
        for future in as_completed(futures):
            batch_results = future.result()
            for i, j, value in batch_results:
                te_matrix[i, j] = value
    
    return te_matrix


# ============================================================================
# 替换原函数的代码片段
# ============================================================================
"""
在 extract_connectivity_features.py 中替换：

方式1 - 常规多进程：
from connectivity_acceleration_multiprocess import (
    compute_granger_causality_pairwise_multiprocess,
    compute_transfer_entropy_pairwise_multiprocess
)

features['granger_causality'] = compute_granger_causality_pairwise_multiprocess(data, sfreq)
features['transfer_entropy'] = compute_transfer_entropy_pairwise_multiprocess(data)


方式2 - 批量多进程（推荐，更高效）：
from connectivity_acceleration_multiprocess import (
    compute_granger_causality_pairwise_batch,
    compute_transfer_entropy_pairwise_batch
)

features['granger_causality'] = compute_granger_causality_pairwise_batch(data, sfreq, batch_size=10)
features['transfer_entropy'] = compute_transfer_entropy_pairwise_batch(data, batch_size=10)
"""

