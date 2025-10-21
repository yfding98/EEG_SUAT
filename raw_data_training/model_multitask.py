"""
多任务学习模型
同时学习：
1. 发作类型分类（主任务）
2. 活跃通道预测（辅助任务）- 显式利用通道组合信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
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


class ChannelFeatureExtractor(nn.Module):
    """通道特征提取器"""
    
    def __init__(self, n_channels, n_samples, d_model=256):
        super().__init__()
        
        # 时间卷积（每个通道独立）
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        
        self.conv_out_len = n_samples // 8
        
    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_samples)
        Returns:
            (batch, d_model, seq_len)
        """
        return self.temporal_conv(x)


class MultiTaskEEGNet(nn.Module):
    """
    多任务EEG网络
    
    任务1（主）：发作类型分类 (SZ1, SZ4, ...)
    任务2（辅）：活跃通道预测 (哪些通道是发作源)
    任务3（辅）：源-传导关系预测
    """
    
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.d_model = d_model
        
        # 共享的特征提取器
        self.feature_extractor = ChannelFeatureExtractor(n_channels, n_samples, d_model)
        self.conv_out_len = self.feature_extractor.conv_out_len
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.conv_out_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 任务1：发作类型分类头
        self.seizure_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        # 任务2：活跃通道预测头（每个通道二分类）
        self.channel_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_channels)  # 每个通道一个logit
        )
        
        # 任务3：通道关系预测（可选）
        self.relation_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_channels * n_channels)
        )
        
    def forward(self, x, return_aux_tasks=True):
        """
        Args:
            x: (batch, n_channels, n_samples)
            return_aux_tasks: 是否返回辅助任务的输出
        
        Returns:
            如果return_aux_tasks=True:
                - seizure_logits: (batch, n_classes) 发作类型
                - channel_logits: (batch, n_channels) 活跃通道预测
                - relation_logits: (batch, n_channels, n_channels) 通道关系
            否则:
                - seizure_logits: (batch, n_classes)
        """
        batch_size = x.size(0)
        
        # 1. 时间特征提取
        temporal_features = self.feature_extractor(x)  # (batch, d_model, seq_len)
        
        # 2. 转置并添加位置编码
        temporal_features = temporal_features.transpose(1, 2)  # (batch, seq_len, d_model)
        temporal_features = self.pos_encoder(temporal_features)
        
        # 3. Transformer编码
        encoded = self.transformer(temporal_features)  # (batch, seq_len, d_model)
        
        # 4. 全局池化
        global_features = encoded.mean(dim=1)  # (batch, d_model)
        
        # 5. 任务1：发作类型分类
        seizure_logits = self.seizure_classifier(global_features)  # (batch, n_classes)
        
        if not return_aux_tasks:
            return seizure_logits
        
        # 6. 任务2：活跃通道预测
        channel_logits = self.channel_predictor(global_features)  # (batch, n_channels)
        
        # 7. 任务3：通道关系预测
        relation_logits = self.relation_predictor(global_features)  # (batch, n_ch*n_ch)
        relation_logits = relation_logits.view(batch_size, self.n_channels, self.n_channels)
        
        return seizure_logits, channel_logits, relation_logits


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    
    L_total = λ1·L_seizure + λ2·L_channel + λ3·L_relation
    """
    
    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        seizure_weight: float = 1.0,
        channel_weight: float = 0.5,
        relation_weight: float = 0.3
    ):
        super().__init__()
        
        self.seizure_weight = seizure_weight
        self.channel_weight = channel_weight
        self.relation_weight = relation_weight
        
        # 任务1：发作类型分类loss
        self.seizure_criterion = nn.CrossEntropyLoss()
        
        # 任务2：活跃通道预测loss（多标签二分类）
        self.channel_criterion = nn.BCEWithLogitsLoss()
        
        # 任务3：通道关系loss
        self.relation_criterion = nn.MSELoss()
        
    def forward(self, 
                seizure_logits, channel_logits, relation_logits,
                seizure_target, channel_mask):
        """
        Args:
            seizure_logits: (batch, n_classes) 发作类型预测
            channel_logits: (batch, n_channels) 活跃通道预测
            relation_logits: (batch, n_channels, n_channels) 通道关系预测
            seizure_target: (batch,) 发作类型真实标签
            channel_mask: (batch, n_channels) 活跃通道真实标签
        """
        # 任务1：发作类型分类
        loss_seizure = self.seizure_criterion(seizure_logits, seizure_target)
        
        # 任务2：活跃通道预测
        loss_channel = self.channel_criterion(channel_logits, channel_mask)
        
        # 任务3：通道关系（从channel_mask构建理想的关系矩阵）
        # 活跃通道之间应该有强关联
        batch_size = channel_mask.size(0)
        ideal_relation = torch.bmm(
            channel_mask.unsqueeze(2),  # (batch, n_channels, 1)
            channel_mask.unsqueeze(1)   # (batch, 1, n_channels)
        )  # (batch, n_channels, n_channels) - 活跃通道间为1，其他为0
        
        loss_relation = self.relation_criterion(
            torch.sigmoid(relation_logits), 
            ideal_relation
        )
        
        # 总损失
        total_loss = (
            self.seizure_weight * loss_seizure +
            self.channel_weight * loss_channel +
            self.relation_weight * loss_relation
        )
        
        return {
            'total': total_loss,
            'seizure': loss_seizure.item(),
            'channel': loss_channel.item(),
            'relation': loss_relation.item()
        }


def create_multitask_model(n_channels, n_samples, n_classes, **kwargs):
    """创建多任务模型"""
    return MultiTaskEEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        **kwargs
    )


if __name__ == "__main__":
    # 测试
    batch_size = 4
    n_channels = 21
    n_samples = 1500
    n_classes = 5
    
    model = create_multitask_model(n_channels, n_samples, n_classes)
    
    # 模拟数据
    x = torch.randn(batch_size, n_channels, n_samples)
    channel_mask = torch.zeros(batch_size, n_channels)
    
    # 模拟活跃通道 [F7, T3] 在索引 [6, 13]
    channel_mask[0, [6, 13]] = 1  # 样本0: F7, T3
    channel_mask[1, [2, 5, 8]] = 1  # 样本1: 3个活跃通道
    channel_mask[2, [10]] = 1  # 样本2: 1个活跃通道
    channel_mask[3, [1, 4, 7, 15]] = 1  # 样本3: 4个活跃通道
    
    seizure_target = torch.tensor([1, 4, 1, 2])  # 发作类型标签
    
    # 前向传播
    seizure_logits, channel_logits, relation_logits = model(x)
    
    print("="*60)
    print("多任务模型测试")
    print("="*60)
    print(f"\n输入:")
    print(f"  数据形状: {x.shape}")
    print(f"  通道掩码形状: {channel_mask.shape}")
    print(f"  发作类型标签: {seizure_target}")
    
    print(f"\n输出:")
    print(f"  发作类型logits: {seizure_logits.shape}")
    print(f"  活跃通道logits: {channel_logits.shape}")
    print(f"  通道关系logits: {relation_logits.shape}")
    
    # 计算loss
    criterion = MultiTaskLoss(n_classes, n_channels)
    losses = criterion(
        seizure_logits, channel_logits, relation_logits,
        seizure_target, channel_mask
    )
    
    print(f"\n损失:")
    print(f"  总损失: {losses['total']:.4f}")
    print(f"  发作类型loss: {losses['seizure']:.4f}")
    print(f"  活跃通道loss: {losses['channel']:.4f}")
    print(f"  通道关系loss: {losses['relation']:.4f}")
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 检查通道预测准确性
    channel_preds = (torch.sigmoid(channel_logits) > 0.5).float()
    channel_acc = (channel_preds == channel_mask).float().mean()
    print(f"\n通道预测准确率: {channel_acc.item()*100:.2f}%")

