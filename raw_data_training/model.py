"""
时空特征提取网络模型
结合CNN和Transformer来提取EEG的时空特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class TemporalConvNet(nn.Module):
    """时间卷积网络 - 用于提取时间维度特征"""
    
    def __init__(self, n_channels: int, n_filters: int = 64, kernel_size: int = 7):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_channels, n_filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(n_filters)
        
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(n_filters*2)
        
        self.conv3 = nn.Conv1d(n_filters*2, n_filters*4, kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(n_filters*4)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_samples)
        Returns:
            (batch, n_filters*4, n_samples//8)
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        return x


class SpatialAttention(nn.Module):
    """空间注意力 - 用于捕获通道间的关系"""
    
    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, features)
        Returns:
            (batch, channels, features)
        """
        # Global average pooling
        gap = x.mean(dim=-1)  # (batch, channels)
        
        # Attention weights
        weights = self.attention(gap)  # (batch, channels)
        weights = weights.unsqueeze(-1)  # (batch, channels, 1)
        
        # Apply attention
        return x * weights


class PositionalEncoding(nn.Module):
    """位置编码 for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class SpatioTemporalEEGNet(nn.Module):
    """
    时空EEG特征提取网络
    结合CNN提取局部时空特征 + Transformer捕获长程依赖
    """
    
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        n_filters: int = 64,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.3
    ):
        """
        Args:
            n_channels: 输入通道数
            n_samples: 每个窗口的采样点数
            n_classes: 分类类别数
            n_filters: CNN滤波器数量
            d_model: Transformer模型维度
            n_heads: Transformer注意力头数
            n_layers: Transformer层数
            dropout: Dropout比例
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        
        # 时间卷积网络
        self.temporal_conv = TemporalConvNet(n_channels, n_filters)
        
        # 空间注意力
        self.spatial_attention = SpatialAttention(n_filters * 4)
        
        # 计算卷积后的序列长度
        self.conv_out_len = n_samples // 8
        
        # 特征投影
        self.feature_proj = nn.Linear(n_filters * 4, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.conv_out_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_samples)
        Returns:
            logits: (batch, n_classes)
        """
        batch_size = x.size(0)
        
        # 时间卷积特征提取
        x = self.temporal_conv(x)  # (batch, n_filters*4, seq_len)
        
        # 空间注意力
        x = self.spatial_attention(x)  # (batch, n_filters*4, seq_len)
        
        # 转置为 (batch, seq_len, n_filters*4)
        x = x.transpose(1, 2)
        
        # 特征投影到Transformer维度
        x = self.feature_proj(x)  # (batch, seq_len, d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (batch, d_model)
        
        # 分类
        logits = self.classifier(x)  # (batch, n_classes)
        
        return logits


class LightweightEEGNet(nn.Module):
    """
    轻量级EEG网络（适用于较少数据的场景）
    使用纯CNN架构
    """
    
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # 第一层：时间卷积
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(16)
        
        # 第二层：深度卷积（通道分离）
        self.conv2 = nn.Conv2d(16, 32, (n_channels, 1), groups=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((1, 4))
        
        # 第三层：可分离卷积
        self.conv3 = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32)
        self.conv4 = nn.Conv2d(32, 64, (1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d((1, 8))
        
        # 计算展平后的大小
        self.flatten_size = self._get_flatten_size(n_channels, n_samples)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def _get_flatten_size(self, n_channels, n_samples):
        """计算展平后的大小"""
        x = torch.zeros(1, 1, n_channels, n_samples)
        x = self.pool2(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        x = self.pool3(F.relu(self.bn3(self.conv4(self.conv3(x)))))
        return x.numel()
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_samples)
        Returns:
            logits: (batch, n_classes)
        """
        # 添加一个维度 (batch, 1, n_channels, n_samples)
        x = x.unsqueeze(1)
        
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Block 2
        x = self.conv3(x)
        x = F.relu(self.bn3(self.conv4(x)))
        x = self.pool3(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def create_model(
    model_type: str,
    n_channels: int,
    n_samples: int,
    n_classes: int,
    **kwargs
) -> nn.Module:
    """
    创建模型
    
    Args:
        model_type: 模型类型 ('spatiotemporal' 或 'lightweight')
        n_channels: 通道数
        n_samples: 采样点数
        n_classes: 类别数
        **kwargs: 其他参数
    
    Returns:
        模型
    """
    if model_type == 'spatiotemporal':
        return SpatioTemporalEEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            **kwargs
        )
    elif model_type == 'lightweight':
        return LightweightEEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 测试模型
    batch_size = 4
    n_channels = 3
    n_samples = 1500  # 6秒 @ 250Hz
    n_classes = 5
    
    # 测试时空模型
    print("Testing SpatioTemporalEEGNet...")
    model1 = SpatioTemporalEEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes
    )
    x = torch.randn(batch_size, n_channels, n_samples)
    out1 = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out1.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    print("\nTesting LightweightEEGNet...")
    model2 = LightweightEEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes
    )
    out2 = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out2.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model2.parameters()):,}")

