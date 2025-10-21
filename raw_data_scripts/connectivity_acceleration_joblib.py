#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 joblib 并行化加速 - 方案1
需要安装: pip install joblib
"""

import numpy as np
from scipy.stats import zscore
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Granger Causality - joblib并行版本
# ============================================================================
def _compute_gc_pair(i, j, data, max_lag):
    """计算单个通道对的Granger因果性"""
    if i == j:
        return 0
    
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        test_data = np.column_stack([data[j], data[i]])
        result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
        f_values = [result[lag][0]['ssr_ftest'][0] for lag in range(1, max_lag+1)]
        return np.mean(f_values)
    except:
        return 0


def compute_granger_causality_pairwise_parallel(data, sfreq, max_lag=None, n_jobs=-1):
    """
    并行计算Granger因果性
    
    参数:
        data: (n_channels, n_samples)
        sfreq: 采样率
        max_lag: 最大滞后
        n_jobs: 并行作业数，-1表示使用所有CPU核心
    """
    n_channels = data.shape[0]
    
    if max_lag is None:
        max_lag = int(sfreq / 10)
    max_lag = max(1, min(max_lag, 20))
    
    # 生成所有通道对
    pairs = [(i, j) for i in range(n_channels) for j in range(n_channels)]
    
    # 并行计算
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_compute_gc_pair)(i, j, data, max_lag) for i, j in pairs
    )
    
    # 重组为矩阵
    gc_matrix = np.array(results).reshape(n_channels, n_channels)
    return gc_matrix


# ============================================================================
# Transfer Entropy - joblib并行版本
# ============================================================================
def _calculate_entropy_discrete_fast(data, n_bins):
    """快速计算离散熵"""
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
        H_XYZ = _calculate_entropy_discrete_fast(XYZ, n_bins**3)
        H_XZ = _calculate_entropy_discrete_fast(np.column_stack([X, Z]), n_bins**2)
        H_YZ = _calculate_entropy_discrete_fast(np.column_stack([Y, Z]), n_bins**2)
        H_Z = _calculate_entropy_discrete_fast(Z.reshape(-1, 1), n_bins)
        return H_XZ + H_YZ - H_XYZ - H_Z
    except:
        return 0


def _compute_te_pair(i, j, data_normalized, n_bins, lag):
    """计算单个通道对的传递熵"""
    if i == j:
        return 0
    
    try:
        X = data_normalized[i]
        Y = data_normalized[j]
        
        X_bins = np.digitize(X, bins=np.linspace(X.min(), X.max(), n_bins))
        Y_bins = np.digitize(Y, bins=np.linspace(Y.min(), Y.max(), n_bins))
        
        Y_present = Y_bins[lag:]
        Y_past = Y_bins[:-lag]
        X_past = X_bins[:-lag]
        
        te = _calculate_cmi(Y_present, X_past, Y_past, n_bins)
        return max(0, te)
    except:
        return 0


def compute_transfer_entropy_pairwise_parallel(data, n_bins=10, lag=1, n_jobs=-1):
    """
    并行计算传递熵
    
    参数:
        data: (n_channels, n_samples)
        n_bins: 离散化bins
        lag: 时间延迟
        n_jobs: 并行作业数
    """
    n_channels = data.shape[0]
    data_normalized = zscore(data, axis=1)
    
    # 生成所有通道对
    pairs = [(i, j) for i in range(n_channels) for j in range(n_channels)]
    
    # 并行计算
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_compute_te_pair)(i, j, data_normalized, n_bins, lag) for i, j in pairs
    )
    
    # 重组为矩阵
    te_matrix = np.array(results).reshape(n_channels, n_channels)
    return te_matrix


# ============================================================================
# 替换原函数的代码片段
# ============================================================================
"""
在 extract_connectivity_features.py 中替换：

1. 在文件开头添加导入：
from connectivity_acceleration_joblib import (
    compute_granger_causality_pairwise_parallel,
    compute_transfer_entropy_pairwise_parallel
)

2. 在 extract_all_features 函数中替换：

# 原代码：
features['granger_causality'] = compute_granger_causality_pairwise(data, sfreq)
features['transfer_entropy'] = compute_transfer_entropy_pairwise(data)

# 新代码：
features['granger_causality'] = compute_granger_causality_pairwise_parallel(data, sfreq, n_jobs=-1)
features['transfer_entropy'] = compute_transfer_entropy_pairwise_parallel(data, n_jobs=-1)
"""

