"""
通道感知多标签分类模型
支持多频段融合和通道显著性预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BandAttention(nn.Module):
    """
    频段注意力模块
    学习不同频段的重要性权重
    """
    
    def __init__(self, n_bands, d_model):
        super().__init__()
        self.n_bands = n_bands
        self.d_model = d_model
        
        # 频段特征提取
        self.band_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                
                nn.Conv1d(32, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                
                nn.Conv1d(64, d_model, kernel_size=7, padding=3),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)  # 全局平均池化
            )
            for _ in range(n_bands)
        ])
        
        # 注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, bands_tensor):
        """
        Args:
            bands_tensor: (batch, n_bands, n_channels, n_samples)
        Returns:
            fused_data: (batch, n_channels, n_samples)
            attention_weights: (batch, n_bands)
        """
        batch_size, n_bands, n_channels, n_samples = bands_tensor.shape
        
        # 提取每个频段的特征
        band_features = []
        for i in range(n_bands):
            band_data = bands_tensor[:, i, :, :]  # (batch, n_channels, n_samples)
            
            # 对每个通道分别处理
            channel_features = []
            for ch in range(n_channels):
                ch_data = band_data[:, ch:ch+1, :]  # (batch, 1, n_samples)
                ch_feat = self.band_encoders[i](ch_data)  # (batch, d_model, 1)
                channel_features.append(ch_feat.squeeze(-1))  # (batch, d_model)
            
            # 平均所有通道的特征
            band_feat = torch.stack(channel_features, dim=1).mean(dim=1)  # (batch, d_model)
            band_features.append(band_feat)
        
        band_features = torch.stack(band_features, dim=1)  # (batch, n_bands, d_model)
        
        # 计算注意力权重
        attention_scores = self.attention(band_features)  # (batch, n_bands, 1)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)  # (batch, n_bands)
        
        # 加权融合频段数据
        attention_weights_expanded = attention_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, n_bands, 1, 1)
        fused_data = (bands_tensor * attention_weights_expanded).sum(dim=1)  # (batch, n_channels, n_samples)
        
        return fused_data, attention_weights


class ChannelRoleAttention(nn.Module):
    """
    通道角色注意力机制
    区分活跃通道（发作源）和传导通道
    """
    
    def __init__(self, n_channels, d_model):
        super().__init__()
        self.n_channels = n_channels
        
        # 通道角色编码
        self.role_embedding = nn.Embedding(2, d_model)  # 0:非活跃, 1:活跃
        
        # 通道特征变换
        self.channel_transform = nn.Linear(d_model, d_model)
        
        # 注意力权重
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
    def forward(self, x, channel_mask):
        """
        Args:
            x: (batch, n_channels, d_model) 通道特征
            channel_mask: (batch, n_channels) 1表示活跃通道，0表示非活跃
        """
        batch_size = x.size(0)
        
        # 创建角色编码
        role_idx = channel_mask.long()  # (batch, n_channels)
        role_emb = self.role_embedding(role_idx)  # (batch, n_channels, d_model)
        
        # 融合角色信息
        x_with_role = x + role_emb
        
        # 通道间注意力
        # 活跃通道作为query，所有通道作为key/value
        attn_out, attn_weights = self.attention(
            x_with_role, x_with_role, x_with_role
        )
        
        return attn_out, attn_weights


class SpatialPropagationModule(nn.Module):
    """
    空间传播模块
    建模从活跃通道到其他通道的信号传播
    """
    
    def __init__(self, n_channels, d_model):
        super().__init__()
        
        # 源通道特征提取
        self.source_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 传播通道特征提取
        self.propagated_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 源-传播关系学习
        self.relation_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x, channel_mask):
        """
        Args:
            x: (batch, n_channels, d_model)
            channel_mask: (batch, n_channels)
        """
        batch_size, n_channels, d_model = x.shape
        
        # 分离活跃通道和非活跃通道
        mask_expanded = channel_mask.unsqueeze(-1)  # (batch, n_channels, 1)
        
        # 活跃通道特征（求平均）
        source_features = (x * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        source_features = self.source_encoder(source_features)  # (batch, d_model)
        
        # 非活跃通道特征
        propagated_mask = 1 - mask_expanded
        propagated_features = x * propagated_mask
        propagated_features = self.propagated_encoder(propagated_features)  # (batch, n_channels, d_model)
        
        # 源-传播关系
        source_expanded = source_features.unsqueeze(1).expand(-1, n_channels, -1)
        combined = torch.cat([source_expanded, propagated_features], dim=-1)
        relation_features = self.relation_net(combined)
        
        return relation_features, source_features


class ChannelAwareMultilabelNet(nn.Module):
    """
    通道感知多标签分类网络
    
    核心思想：
    1. 使用BandAttention融合多频段数据
    2. 利用活跃通道标记引导特征学习
    3. 建模容积传导的时空传播模式
    4. 输出每个通道的显著性概率
    """
    
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_bands: int = 6,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_bands = n_bands
        self.d_model = d_model
        
        # 1. 频段注意力融合
        self.band_attention = BandAttention(n_bands, d_model)
        
        # 2. 时间卷积 - 提取每个通道的时域特征
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        
        # 计算卷积后的序列长度
        self.conv_out_len = n_samples // 8
        
        # 3. 通道维度转换
        self.channel_projection = nn.Linear(256, d_model)
        
        # 4. 通道角色注意力
        self.channel_role_attention = ChannelRoleAttention(n_channels, d_model)
        
        # 5. 空间传播模块
        self.spatial_propagation = SpatialPropagationModule(n_channels, d_model)
        
        # 6. 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.conv_out_len)
        
        # 7. Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 8. 多标签分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2因为融合了source和全局特征
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_channels)  # 每个通道一个二分类输出
        )
        
    def forward(self, bands_tensor, channel_mask=None):
        """
        Args:
            bands_tensor: (batch, n_bands, n_channels, n_samples) 多频段EEG数据
            channel_mask: (batch, n_channels) 活跃通道掩码，1表示活跃，0表示非活跃
        
        Returns:
            logits: (batch, n_channels) 每个通道的显著性logits
        """
        batch_size = bands_tensor.size(0)
        
        # 如果没有提供mask，创建全1的mask（所有通道都考虑）
        if channel_mask is None:
            channel_mask = torch.ones(batch_size, self.n_channels, device=bands_tensor.device)

        bands_tensor = bands_tensor.permute(1, 0, 2, 3)
        # 1. 频段注意力融合
        fused_data, attention_weights = self.band_attention(bands_tensor)  # (batch, n_channels, n_samples)
        
        # 2. 时间卷积特征提取
        temporal_features = self.temporal_conv(fused_data)  # (batch, 256, seq_len)
        
        # 3. 转置并投影到d_model维度
        temporal_features = temporal_features.transpose(1, 2)  # (batch, seq_len, 256)
        temporal_features = self.channel_projection(temporal_features)  # (batch, seq_len, d_model)
        
        # 4. 位置编码
        temporal_features = self.pos_encoder(temporal_features)
        
        # 5. 通道级特征聚合（对时间维度池化）
        channel_features = temporal_features.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        channel_features = channel_features.expand(-1, self.n_channels, -1)  # (batch, n_channels, d_model)
        
        # 6. 通道角色注意力
        channel_attended, attn_weights = self.channel_role_attention(
            channel_features, channel_mask
        )
        
        # 7. 空间传播建模
        propagation_features, source_features = self.spatial_propagation(
            channel_attended, channel_mask
        )
        
        # 8. 融合通道特征到时间序列
        # 将通道维度的特征广播到时间维度
        channel_info = propagation_features.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        channel_info = channel_info.expand(-1, self.conv_out_len, -1)
        
        # 融合
        fused_features = temporal_features + channel_info
        
        # 9. Transformer编码
        encoded = self.transformer(fused_features)  # (batch, seq_len, d_model)
        
        # 10. 全局池化
        global_features = encoded.mean(dim=1)  # (batch, d_model)
        
        # 11. 融合源通道特征和全局特征
        final_features = torch.cat([source_features, global_features], dim=-1)
        
        # 12. 多标签分类
        logits = self.classifier(final_features)  # (batch, n_channels)
        
        return logits


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
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


def create_channel_aware_multilabel_model(
    n_channels: int,
    n_samples: int,
    n_bands: int = 6,
    **kwargs
):
    """创建通道感知多标签模型"""
    return ChannelAwareMultilabelNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_bands=n_bands,
        **kwargs
    )


if __name__ == "__main__":
    # 测试
    batch_size = 4
    n_channels = 21
    n_samples = 1536  # 6秒 * 256Hz
    n_bands = 6
    
    model = create_channel_aware_multilabel_model(n_channels, n_samples, n_bands)
    
    # 模拟输入
    bands_tensor = torch.randn(batch_size, n_bands, n_channels, n_samples)
    
    # 模拟通道掩码（随机2-5个活跃通道）
    channel_mask = torch.zeros(batch_size, n_channels)
    for i in range(batch_size):
        n_active = torch.randint(2, 6, (1,)).item()
        active_idx = torch.randperm(n_channels)[:n_active]
        channel_mask[i, active_idx] = 1
    
    # 前向传播
    logits = model(bands_tensor, channel_mask)
    
    print(f"输入形状: {bands_tensor.shape}")
    print(f"通道掩码形状: {channel_mask.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 打印每个样本的活跃通道数
    for i in range(batch_size):
        n_active = channel_mask[i].sum().item()
        print(f"样本{i}: {int(n_active)}个活跃通道")
