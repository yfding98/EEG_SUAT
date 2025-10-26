#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basic_eeg_channel_classifier.py

基础EEG通道分类器 - 专门用于发作前期显著通道标记
基于多频段特征提取和通道注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Dict, Optional


class MultiBandFeatureExtractor(nn.Module):
    """
    多频段特征提取器
    针对EEG的不同频段（delta, theta, alpha, beta, gamma, hfo）进行特征提取
    """
    
    def __init__(self, n_channels: int, n_samples: int, d_model: int = 128):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.d_model = d_model
        
        # 每个频段的特征提取器
        self.band_extractors = nn.ModuleList([
            nn.Sequential(
                # 时间卷积
                nn.Conv1d(1, 32, kernel_size=15, padding=7),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                
                nn.Conv1d(32, 64, kernel_size=15, padding=7),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                
                nn.Conv1d(64, 128, kernel_size=15, padding=7),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)  # 全局平均池化
            ) for _ in range(6)  # 6个频段
        ])
        
        # 频段融合层
        self.band_fusion = nn.Sequential(
            nn.Linear(128 * 6, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, bands_data: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            bands_data: List of (batch, n_channels, n_samples) for each band
        Returns:
            features: (batch, n_channels, d_model)
        """
        batch_size = bands_data[0].shape[0]
        n_channels = bands_data[0].shape[1]
        
        # 对每个频段提取特征
        band_features = []
        for i, band_data in enumerate(bands_data):
            # band_data: (batch, n_channels, n_samples)
            batch_size, n_channels, n_samples = band_data.shape
            
            # 对每个通道分别处理
            channel_features = []
            for ch in range(n_channels):
                ch_data = band_data[:, ch:ch+1, :]  # (batch, 1, n_samples)
                ch_feat = self.band_extractors[i](ch_data)  # (batch, 128, 1)
                channel_features.append(ch_feat.squeeze(-1))  # (batch, 128)
            
            # 堆叠所有通道的特征
            band_feat = torch.stack(channel_features, dim=1)  # (batch, n_channels, 128)
            band_features.append(band_feat)
        
        # 融合所有频段特征
        fused_features = torch.cat(band_features, dim=-1)  # (batch, n_channels, 128*6)
        fused_features = self.band_fusion(fused_features)  # (batch, n_channels, d_model)
        
        return fused_features


class ChannelAttention(nn.Module):
    """
    通道注意力机制
    学习通道间的重要性权重，突出异常通道
    """
    
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_channels, d_model)
        Returns:
            output: (batch, n_channels, d_model)
            attention_weights: (batch, n_channels, n_channels)
        """
        # 自注意力
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights


class SpatialTemporalEncoder(nn.Module):
    """
    时空编码器
    结合空间（通道间）和时间信息
    """
    
    def __init__(self, n_channels: int, d_model: int, n_layers: int = 2):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, d_model)
        Returns:
            encoded: (batch, n_channels, d_model)
        """
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        encoded = self.transformer(x)
        
        return encoded


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class BasicEEGChannelClassifier(nn.Module):
    """
    基础EEG通道分类器
    
    架构：
    1. 多频段特征提取
    2. 通道注意力机制
    3. 时空编码
    4. 多标签分类头
    """
    
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.d_model = d_model
        
        # 1. 多频段特征提取
        self.band_extractor = MultiBandFeatureExtractor(n_channels, n_samples, d_model)
        
        # 2. 通道注意力
        self.channel_attention = ChannelAttention(d_model, n_heads)
        
        # 3. 时空编码
        self.spatial_temporal_encoder = SpatialTemporalEncoder(n_channels, d_model, n_layers)
        
        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # 每个通道一个二分类
        )
        
    def forward(self, bands_data: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            bands_data: List of (batch, n_channels, n_samples) for each band
        Returns:
            logits: (batch, n_channels) - 每个通道的显著性logits
        """
        # 1. 多频段特征提取
        features = self.band_extractor(bands_data)  # (batch, n_channels, d_model)
        
        # 2. 通道注意力
        attended_features, attention_weights = self.channel_attention(features)
        
        # 3. 时空编码
        encoded_features = self.spatial_temporal_encoder(attended_features)
        
        # 4. 分类
        logits = self.classifier(encoded_features)  # (batch, n_channels, 1)
        logits = logits.squeeze(-1)  # (batch, n_channels)
        
        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    专门处理EEG通道分类中的类别不平衡问题
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, n_channels) - 预测logits
            targets: (batch, n_channels) - 真实标签
        """
        # 计算BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-bce_loss)
        
        # Focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveClassWeights:
    """
    自适应类别权重计算
    根据每个通道的正样本比例动态调整权重
    """
    
    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        
    def compute_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: (batch, n_channels) - 真实标签
        Returns:
            weights: (n_channels,) - 每个通道的权重
        """
        # 计算每个通道的正样本比例
        positive_ratios = labels.mean(dim=0)  # (n_channels,)
        
        # 计算权重
        weights = torch.ones(self.n_channels)
        
        for i, ratio in enumerate(positive_ratios):
            if ratio == 0.0:
                # 从未出现的通道：极小权重
                weights[i] = 0.01
            elif ratio < 0.05:
                # 极稀有通道：高权重保护
                weights[i] = 10.0
            elif ratio < 0.1:
                # 稀有通道：较高权重
                weights[i] = 5.0
            elif ratio < 0.2:
                # 低频通道：中等权重
                weights[i] = 3.0
            elif ratio < 0.4:
                # 中频通道：正常权重
                weights[i] = 2.0
            else:
                # 高频通道：低权重
                weights[i] = 1.0
        
        # 归一化权重
        weights = weights / weights.mean()
        
        return weights


def create_basic_eeg_classifier(
    n_channels: int,
    n_samples: int,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 2,
    dropout: float = 0.3
) -> BasicEEGChannelClassifier:
    """创建基础EEG通道分类器"""
    return BasicEEGChannelClassifier(
        n_channels=n_channels,
        n_samples=n_samples,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    )


def compute_multilabel_metrics(pred_logits: torch.Tensor, true_labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    计算多标签分类指标
    """
    pred_probs = torch.sigmoid(pred_logits)
    pred_binary = (pred_probs > threshold).float()
    
    batch_size, n_channels = pred_logits.shape
    
    # 每个通道的指标
    channel_precisions = []
    channel_recalls = []
    channel_f1s = []
    
    for ch in range(n_channels):
        pred_ch = pred_binary[:, ch]
        true_ch = true_labels[:, ch]
        
        # 计算TP, FP, FN
        tp = (pred_ch * true_ch).sum().item()
        fp = (pred_ch * (1 - true_ch)).sum().item()
        fn = ((1 - pred_ch) * true_ch).sum().item()
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        channel_precisions.append(precision)
        channel_recalls.append(recall)
        channel_f1s.append(f1)
    
    # 宏平均
    macro_precision = np.mean(channel_precisions) * 100
    macro_recall = np.mean(channel_recalls) * 100
    macro_f1 = np.mean(channel_f1s) * 100
    
    # 微平均
    total_tp = sum([(pred_binary[:, ch] * true_labels[:, ch]).sum().item() for ch in range(n_channels)])
    total_fp = sum([(pred_binary[:, ch] * (1 - true_labels[:, ch])).sum().item() for ch in range(n_channels)])
    total_fn = sum([((1 - pred_binary[:, ch]) * true_labels[:, ch]).sum().item() for ch in range(n_channels)])
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision * 100,
        'micro_recall': micro_recall * 100,
        'micro_f1': micro_f1 * 100
    }


if __name__ == "__main__":
    # 测试模型
    print("测试基础EEG通道分类器...")
    
    # 模拟参数
    batch_size = 4
    n_channels = 21
    n_samples = 1500  # 6秒 * 250Hz
    n_bands = 6
    
    # 创建模型
    model = create_basic_eeg_classifier(
        n_channels=n_channels,
        n_samples=n_samples,
        d_model=128,
        n_heads=8,
        n_layers=2
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 模拟输入数据
    bands_data = [torch.randn(batch_size, n_channels, n_samples) for _ in range(n_bands)]
    
    # 模拟标签（随机2-5个异常通道）
    labels = torch.zeros(batch_size, n_channels)
    for i in range(batch_size):
        n_abnormal = torch.randint(2, 6, (1,)).item()
        abnormal_idx = torch.randperm(n_channels)[:n_abnormal]
        labels[i, abnormal_idx] = 1
    
    # 前向传播
    with torch.no_grad():
        logits = model(bands_data)
        probs = torch.sigmoid(logits)
        pred_binary = (probs > 0.5).float()
    
    print(f"输入形状: {[band.shape for band in bands_data]}")
    print(f"输出形状: {logits.shape}")
    print(f"真实标签: {labels.sum(dim=1).tolist()}")
    print(f"预测标签: {pred_binary.sum(dim=1).tolist()}")
    
    # 计算指标
    metrics = compute_multilabel_metrics(logits, labels)
    print(f"指标: {metrics}")
    
    print("✓ 模型测试完成！")
