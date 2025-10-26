#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frequency_band_training

频段分离训练模块
支持delta, theta, alpha, beta, gamma频段的独立训练
"""

__version__ = "1.0.0"
__author__ = "EEG Research Team"

# 频段定义
FREQUENCY_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 100.0)
}

# 频段顺序（按重要性排序）
BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma']

