#!/usr/bin/env python3
"""
多标签分类模型 - 用于通道级别的异常检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GraphConv(nn.Module):
    """图卷积层"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # x: [B, N, in_dim] or [N, in_dim]
        # adj: [B, N, N] or [N, N]
        support = self.linear(x)
        
        if adj.dim() == 2 and support.dim() == 2:
            # Single graph
            out = torch.matmul(adj, support)
        elif adj.dim() == 3 and support.dim() == 3:
            # Batch of graphs
            out = torch.bmm(adj, support)
        else:
            raise ValueError(f"Dimension mismatch: adj.dim={adj.dim()}, support.dim={support.dim()}")
        
        return out + self.bias


class GCNEncoder(nn.Module):
    """GCN编码器"""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList([
            GraphConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        h = x
        for i, conv in enumerate(self.layers):
            h = conv(h, adj)
            if i < len(self.layers) - 1:
                h = self.act(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Global mean pooling
        if h.dim() == 2:
            g = h.mean(dim=0, keepdim=True)  # [1, D]
        else:
            g = h.mean(dim=1)  # [B, D]
        
        return g


class MultiLabelGNNClassifier(nn.Module):
    """
    多标签GNN分类器
    
    输出每个通道的异常概率
    """
    
    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        num_channels: int = 18,
        num_layers: int = 3,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            in_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            num_channels: 通道数量（输出维度）
            num_layers: GCN层数
            dropout: Dropout比例
            use_batch_norm: 是否使用BatchNorm
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        
        # GCN编码器
        self.encoder = GCNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 多标签分类头
        layers = []
        
        # Layer 1
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Layer 2
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim // 2, num_channels))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, in_dim] 节点特征
            adj: [B, N, N] 邻接矩阵
        
        Returns:
            logits: [B, num_channels] 每个通道的logits
        """
        # 编码
        g = self.encoder(x, adj)  # [B, hidden_dim]
        
        # 分类
        logits = self.classifier(g)  # [B, num_channels]
        
        return logits
    
    def predict_proba(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        预测概率
        
        Returns:
            probs: [B, num_channels] 每个通道的异常概率
        """
        logits = self.forward(x, adj)
        probs = torch.sigmoid(logits)
        return probs
    
    def predict(self, x: torch.Tensor, adj: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        预测二值标签
        
        Args:
            threshold: 判定阈值
        
        Returns:
            preds: [B, num_channels] 二值预测
        """
        probs = self.predict_proba(x, adj)
        preds = (probs > threshold).float()
        return preds


class MultiLabelGNNWithAttention(nn.Module):
    """
    带注意力机制的多标签GNN分类器
    
    使用注意力机制来学习每个通道的重要性
    """
    
    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        num_channels: int = 18,
        num_layers: int = 3,
        dropout: float = 0.5,
        num_heads: int = 4,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        
        # GCN编码器
        self.encoder = GCNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 通道嵌入（可学习）
        self.channel_embeddings = nn.Parameter(
            torch.randn(num_channels, hidden_dim)
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, in_dim] 节点特征
            adj: [B, N, N] 邻接矩阵
        
        Returns:
            logits: [B, num_channels] 每个通道的logits
        """
        B = x.size(0)
        
        # 编码图
        g = self.encoder(x, adj)  # [B, hidden_dim]
        
        # 扩展通道嵌入到batch
        channel_emb = self.channel_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, num_channels, hidden_dim]
        
        # 图嵌入作为query，通道嵌入作为key和value
        g_query = g.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 注意力机制
        attn_output, attn_weights = self.attention(
            g_query.expand(-1, self.num_channels, -1),  # query: [B, num_channels, hidden_dim]
            channel_emb,  # key: [B, num_channels, hidden_dim]
            channel_emb   # value: [B, num_channels, hidden_dim]
        )  # attn_output: [B, num_channels, hidden_dim]
        
        # 投影到logits
        logits = self.output_proj(attn_output).squeeze(-1)  # [B, num_channels]
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重（用于可解释性）
        
        Returns:
            attn_weights: [B, num_heads, num_channels, num_channels]
        """
        B = x.size(0)
        g = self.encoder(x, adj)
        
        channel_emb = self.channel_embeddings.unsqueeze(0).expand(B, -1, -1)
        g_query = g.unsqueeze(1).expand(-1, self.num_channels, -1)
        
        _, attn_weights = self.attention(g_query, channel_emb, channel_emb)
        
        return attn_weights


if __name__ == "__main__":
    print("Testing MultiLabel Models...")
    
    # 测试基础模型
    print("\n1. Testing MultiLabelGNNClassifier...")
    model = MultiLabelGNNClassifier(
        in_dim=2,
        hidden_dim=64,
        num_channels=18,
        num_layers=3,
        dropout=0.5
    )
    
    # 创建测试数据
    batch_size = 4
    num_nodes = 20
    x = torch.randn(batch_size, num_nodes, 2)
    adj = torch.randn(batch_size, num_nodes, num_nodes)
    
    # 前向传播
    logits = model(x, adj)
    print(f"  Input: x={x.shape}, adj={adj.shape}")
    print(f"  Output logits: {logits.shape}")
    
    # 预测
    probs = model.predict_proba(x, adj)
    preds = model.predict(x, adj, threshold=0.5)
    print(f"  Probabilities: {probs.shape}, range=[{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  Predictions: {preds.shape}, sum={preds.sum(dim=1)}")
    
    # 测试注意力模型
    print("\n2. Testing MultiLabelGNNWithAttention...")
    model_attn = MultiLabelGNNWithAttention(
        in_dim=2,
        hidden_dim=64,
        num_channels=18,
        num_layers=3,
        num_heads=4
    )
    
    logits_attn = model_attn(x, adj)
    print(f"  Output logits: {logits_attn.shape}")
    
    attn_weights = model_attn.get_attention_weights(x, adj)
    print(f"  Attention weights: {attn_weights.shape}")
    
    print("\n✓ All tests passed!")

