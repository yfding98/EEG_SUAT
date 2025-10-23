"""
活跃通道检测模型
任务：从21个通道中识别哪些是活跃的（发作源）
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


class ActiveChannelDetector(nn.Module):
    """
    活跃通道检测网络
    
    输入：(batch, 21, 1500) 所有通道的EEG数据
    输出：(batch, 21) 每个通道的激活概率
    
    核心思想：
    1. 提取每个通道的时空特征
    2. 学习通道间的相互作用（容积传导）
    3. 预测每个通道是否活跃
    """
    
    def __init__(
        self,
        n_channels: int = 21,
        n_samples: int = 1500,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.d_model = d_model
        
        # 1. 逐通道时域特征提取
        # 每个通道单独卷积（不混合通道）
        self.channel_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                
                nn.Conv1d(32, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                
                nn.Conv1d(64, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            for _ in range(n_channels)
        ])
        
        self.conv_out_len = n_samples // 8
        
        # 2. 通道特征投影
        self.channel_projection = nn.Linear(128, d_model)
        
        # 3. 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.conv_out_len)
        
        # 4. 通道间注意力（学习通道间相互作用）
        self.cross_channel_attention = nn.MultiheadAttention(
            d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        
        # 5. Transformer编码器（学习时空模式）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 6. 通道激活预测头
        self.activation_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)  # 每个通道一个输出
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_samples) 例如 (batch, 21, 1500)
        
        Returns:
            channel_logits: (batch, n_channels) 每个通道的激活logit
        """
        batch_size = x.size(0)
        
        # 1. 逐通道提取时域特征
        channel_features = []
        for ch_idx in range(self.n_channels):
            ch_data = x[:, ch_idx:ch_idx+1, :]  # (batch, 1, n_samples)
            ch_feat = self.channel_convs[ch_idx](ch_data)  # (batch, 128, seq_len)
            channel_features.append(ch_feat)
        
        # 堆叠为 (batch, n_channels, 128, seq_len)
        channel_features = torch.stack(channel_features, dim=1)
        
        # 2. 重组为 (batch, n_channels, seq_len, 128)
        channel_features = channel_features.permute(0, 1, 3, 2)
        
        # 3. 投影到d_model
        channel_features = self.channel_projection(channel_features)  # (batch, n_ch, seq_len, d_model)
        
        # 4. 重组为 (batch*n_channels, seq_len, d_model) 用于位置编码和Transformer
        original_shape = channel_features.shape
        channel_features = channel_features.view(-1, self.conv_out_len, self.d_model)
        
        # 5. 位置编码
        channel_features = self.pos_encoder(channel_features)
        
        # 6. Transformer编码（学习每个通道的时序模式）
        encoded = self.transformer(channel_features)  # (batch*n_ch, seq_len, d_model)
        
        # 7. 恢复形状
        encoded = encoded.view(batch_size, self.n_channels, self.conv_out_len, self.d_model)
        
        # 8. 时间池化 - 每个通道一个特征向量
        channel_vectors = encoded.mean(dim=2)  # (batch, n_channels, d_model)
        
        # 9. 通道间注意力（学习通道间相互作用，容积传导）
        channel_attended, attn_weights = self.cross_channel_attention(
            channel_vectors, channel_vectors, channel_vectors
        )  # (batch, n_channels, d_model)
        
        # 10. 预测每个通道的激活
        channel_logits = self.activation_predictor(channel_attended)  # (batch, n_channels, 1)
        channel_logits = channel_logits.squeeze(-1)  # (batch, n_channels)
        
        return channel_logits


class FocalBCELoss(nn.Module):
    """
    Focal BCE Loss for 活跃通道检测
    解决类别不平衡（活跃通道通常只有2-5个，非活跃有16-19个）
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, n_channels)
            targets: (batch, n_channels) binary, 1=活跃, 0=非活跃
        """
        # BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal weight
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight (平衡正负样本)
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal BCE
        loss = alpha_weight * focal_weight * bce
        
        return loss.mean()


def create_channel_detector(n_channels=21, n_samples=1500, **kwargs):
    """创建活跃通道检测器"""
    return ActiveChannelDetector(
        n_channels=n_channels,
        n_samples=n_samples,
        **kwargs
    )


if __name__ == "__main__":
    # 测试
    print("="*70)
    print("活跃通道检测模型测试")
    print("="*70)
    
    batch_size = 4
    n_channels = 21
    n_samples = 1500
    
    model = create_channel_detector(n_channels, n_samples)
    
    # 模拟输入
    x = torch.randn(batch_size, n_channels, n_samples)
    
    # 模拟真实活跃通道
    true_active = torch.zeros(batch_size, n_channels)
    true_active[0, [6, 13]] = 1  # 样本0: F7(6), T3(13)
    true_active[1, [2, 5, 8]] = 1  # 样本1: 3个通道
    true_active[2, [10]] = 1  # 样本2: 1个通道
    true_active[3, [1, 4, 7, 15]] = 1  # 样本3: 4个通道
    
    print(f"\n输入:")
    print(f"  数据形状: {x.shape}")
    print(f"  真实活跃通道:")
    for i in range(batch_size):
        active_idx = torch.where(true_active[i] == 1)[0].tolist()
        print(f"    样本{i}: 通道 {active_idx} ({len(active_idx)}个)")
    
    # 前向传播
    print(f"\n前向传播...")
    channel_logits = model(x)
    
    print(f"\n输出:")
    print(f"  通道logits形状: {channel_logits.shape}")
    
    # 预测
    channel_probs = torch.sigmoid(channel_logits)
    channel_preds = (channel_probs > 0.5).float()
    
    print(f"\n预测结果:")
    for i in range(batch_size):
        pred_idx = torch.where(channel_preds[i] == 1)[0].tolist()
        true_idx = torch.where(true_active[i] == 1)[0].tolist()
        
        print(f"\n  样本{i}:")
        print(f"    真实活跃: {true_idx}")
        print(f"    预测活跃: {pred_idx}")
        
        # 计算准确率
        correct = (channel_preds[i] == true_active[i]).float().mean()
        print(f"    准确率: {correct.item()*100:.1f}%")
    
    # 计算loss
    criterion = FocalBCELoss()
    loss = criterion(channel_logits, true_active)
    
    print(f"\nFocal BCE Loss: {loss.item():.4f}")
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n{'='*70}")
    print("测试完成！")
    print("="*70)

