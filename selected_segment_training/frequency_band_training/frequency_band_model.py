#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frequency_band_model.py

频段特定的模型类
为每个频段设计专门的神经网络架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import math


class FrequencyBandFeatureExtractor(nn.Module):
    """频段特定的特征提取器"""
    
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        frequency_band: str,
        d_model: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.frequency_band = frequency_band
        self.d_model = d_model
        
        # 频段特定的卷积层
        self._build_band_specific_layers()
        
        # 通用特征提取
        self.conv1 = nn.Conv1d(n_channels, d_model // 4, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(d_model // 4, d_model // 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(d_model // 4)
        self.bn2 = nn.BatchNorm1d(d_model // 2)
        self.bn3 = nn.BatchNorm1d(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def _build_band_specific_layers(self):
        """构建频段特定的层"""
        if self.frequency_band == 'delta':
            # Delta波: 低频，需要更大的感受野
            self.band_conv = nn.Conv1d(self.n_channels, self.d_model // 8, kernel_size=15, padding=7)
            self.band_pool = nn.MaxPool1d(4)
        elif self.frequency_band == 'theta':
            # Theta波: 中低频，中等感受野
            self.band_conv = nn.Conv1d(self.n_channels, self.d_model // 8, kernel_size=11, padding=5)
            self.band_pool = nn.MaxPool1d(3)
        elif self.frequency_band == 'alpha':
            # Alpha波: 中频，标准感受野
            self.band_conv = nn.Conv1d(self.n_channels, self.d_model // 8, kernel_size=7, padding=3)
            self.band_pool = nn.MaxPool1d(2)
        elif self.frequency_band == 'beta':
            # Beta波: 中高频，较小感受野
            self.band_conv = nn.Conv1d(self.n_channels, self.d_model // 8, kernel_size=5, padding=2)
            self.band_pool = nn.MaxPool1d(2)
        elif self.frequency_band == 'gamma':
            # Gamma波: 高频，最小感受野
            self.band_conv = nn.Conv1d(self.n_channels, self.d_model // 8, kernel_size=3, padding=1)
            self.band_pool = nn.MaxPool1d(1)
        else:
            raise ValueError(f"不支持的频段: {self.frequency_band}")
        
        self.band_bn = nn.BatchNorm1d(self.d_model // 8)
    
    def forward(self, x):
        """
        x: (batch_size, n_channels, n_samples)
        """
        # 频段特定特征提取
        band_features = F.relu(self.band_bn(self.band_conv(x)))
        band_features = self.band_pool(band_features)
        
        # 通用特征提取
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.dropout(x1)
        
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = self.dropout(x2)
        
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3 = self.dropout(x3)
        
        # 全局平均池化
        global_features = self.pool(x3).squeeze(-1)  # (batch_size, d_model)
        band_features_pooled = self.pool(band_features).squeeze(-1)  # (batch_size, d_model//8)
        
        # 合并特征
        combined_features = torch.cat([global_features, band_features_pooled], dim=1)
        
        return combined_features


class FrequencyBandAttention(nn.Module):
    """频段特定的注意力机制"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.3,
        frequency_band: str = 'alpha'
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.frequency_band = frequency_band
        
        # 频段特定的注意力权重
        self.band_attention = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (batch_size, n_channels, d_model)
        """
        batch_size, n_channels, d_model = x.shape
        
        # 频段特定的注意力权重
        band_weights = torch.sigmoid(self.band_attention)
        x_weighted = x * band_weights
        
        # 自注意力
        x_flat = x_weighted.view(batch_size * n_channels, 1, d_model)
        attn_output, _ = self.multihead_attn(x_flat, x_flat, x_flat)
        attn_output = attn_output.view(batch_size, n_channels, d_model)
        
        # 残差连接和层归一化
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class FrequencyBandClassifier(nn.Module):
    """频段特定的分类器"""
    
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        frequency_band: str,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.frequency_band = frequency_band
        self.d_model = d_model
        
        # 特征提取器
        self.feature_extractor = FrequencyBandFeatureExtractor(
            n_channels=n_channels,
            n_samples=n_samples,
            frequency_band=frequency_band,
            d_model=d_model,
            dropout=dropout
        )
        
        # 注意力层
        self.attention = FrequencyBandAttention(
            d_model=d_model + d_model // 8,  # 特征提取器输出维度
            n_heads=n_heads,
            dropout=dropout,
            frequency_band=frequency_band
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model + d_model // 8, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, n_channels)
        )
        
        # 频段特定的初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """频段特定的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 根据频段调整初始化范围
                if self.frequency_band in ['delta', 'theta']:
                    # 低频段：较小的初始化范围
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif self.frequency_band in ['alpha', 'beta']:
                    # 中频段：标准初始化范围
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                elif self.frequency_band == 'gamma':
                    # 高频段：较大的初始化范围
                    nn.init.normal_(m.weight, mean=0.0, std=0.03)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        x: (batch_size, n_channels, n_samples)
        """
        # 特征提取
        features = self.feature_extractor(x)  # (batch_size, d_model + d_model//8)
        
        # 扩展为通道维度
        features_expanded = features.unsqueeze(1).expand(-1, self.n_channels, -1)
        
        # 注意力机制
        attended_features = self.attention(features_expanded)
        
        # 全局平均池化
        global_features = attended_features.mean(dim=1)  # (batch_size, d_model + d_model//8)
        
        # 分类
        logits = self.classifier(global_features)  # (batch_size, n_channels)
        
        return logits


def create_frequency_band_classifier(
    n_channels: int,
    n_samples: int,
    frequency_band: str,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 2,
    dropout: float = 0.3
) -> FrequencyBandClassifier:
    """创建频段特定的分类器"""
    
    if frequency_band not in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        raise ValueError(f"不支持的频段: {frequency_band}")
    
    model = FrequencyBandClassifier(
        n_channels=n_channels,
        n_samples=n_samples,
        frequency_band=frequency_band,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    )
    
    return model


class FrequencyBandLoss(nn.Module):
    """频段特定的损失函数"""
    
    def __init__(
        self,
        frequency_band: str,
        alpha: float = 1.0,
        gamma: float = 2.0,
        use_class_weights: bool = True
    ):
        super().__init__()
        self.frequency_band = frequency_band
        self.alpha = alpha
        self.gamma = gamma
        self.use_class_weights = use_class_weights
        
        # 频段特定的权重
        self.band_weights = {
            'delta': 1.2,    # 低频段权重较高
            'theta': 1.1,
            'alpha': 1.0,    # 基准权重
            'beta': 0.9,
            'gamma': 0.8     # 高频段权重较低
        }
        
        self.band_weight = self.band_weights.get(frequency_band, 1.0)
    
    def forward(self, logits, targets):
        """
        logits: (batch_size, n_channels)
        targets: (batch_size, n_channels)
        """
        # 基础BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 频段权重
        weighted_loss = bce_loss * self.band_weight
        
        # Focal Loss (如果启用)
        if self.gamma > 0:
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            weighted_loss = weighted_loss * focal_loss
        
        return weighted_loss.mean()


def compute_frequency_band_metrics(logits, targets):
    """计算频段特定的指标"""
    predictions = torch.sigmoid(logits)
    predictions_binary = (predictions > 0.5).float()
    
    # 基本指标
    tp = (predictions_binary * targets).sum(dim=0)
    fp = (predictions_binary * (1 - targets)).sum(dim=0)
    fn = ((1 - predictions_binary) * targets).sum(dim=0)
    tn = ((1 - predictions_binary) * (1 - targets)).sum(dim=0)
    
    # 避免除零
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # 宏平均
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # 微平均
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_precision = micro_tp / (micro_tp + micro_fp + 1e-8)
    micro_recall = micro_tp / (micro_tp + micro_fn + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    
    return {
        'macro_precision': macro_precision.item(),
        'macro_recall': macro_recall.item(),
        'macro_f1': macro_f1.item(),
        'micro_precision': micro_precision.item(),
        'micro_recall': micro_recall.item(),
        'micro_f1': micro_f1.item()
    }

