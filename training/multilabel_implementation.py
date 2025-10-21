#!/usr/bin/env python3
"""
多标签分类实现 - 预测每个通道的异常概率

使用场景：
- 输入：EEG连接图
- 输出：每个通道是否异常的概率
- 标签：通道组合（如 "Fp1-F3-C3"）→ 转换为多标签向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    hamming_loss, jaccard_score, classification_report,
    multilabel_confusion_matrix, average_precision_score
)


# ============================================================================
# 标准通道定义
# ============================================================================

# 10-20系统标准通道
STANDARD_CHANNELS = [
    'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

# 可以根据你的数据调整
ALL_CHANNELS = STANDARD_CHANNELS


# ============================================================================
# 数据集类
# ============================================================================

class MultiLabelConnectivityDataset(Dataset):
    """
    多标签连接性数据集
    
    将通道组合转换为多标签向量
    例如: "Fp1-F3-C3" → [1,0,1,1,0,0,...]
    """
    
    def __init__(
        self, 
        npz_paths: List[str], 
        labels_df: pd.DataFrame,
        all_channels: List[str] = None,
        matrix_keys: List[str] = ["plv_alpha"],
        fusion_method: str = "weighted",
        topk_ratio: float = 0.2
    ):
        """
        Args:
            npz_paths: NPZ文件路径列表
            labels_df: 标签DataFrame（包含channel_combination列）
            all_channels: 所有可能的通道列表（默认使用STANDARD_CHANNELS）
            matrix_keys: 使用的矩阵键
            fusion_method: 矩阵融合方法
            topk_ratio: Top-k稀疏化比例
        """
        self.npz_paths = npz_paths
        self.labels_df = labels_df
        self.matrix_keys = matrix_keys if isinstance(matrix_keys, list) else [matrix_keys]
        self.fusion_method = fusion_method
        self.topk_ratio = topk_ratio
        
        # 通道映射
        self.all_channels = sorted(all_channels or ALL_CHANNELS)
        self.channel_to_idx = {ch: idx for idx, ch in enumerate(self.all_channels)}
        self.num_channels = len(self.all_channels)
        
        print(f"MultiLabel Dataset: {self.num_channels} channels as labels")
        print(f"Channels: {', '.join(self.all_channels)}")
    
    def __len__(self):
        return len(self.npz_paths)
    
    def _parse_channel_combination(self, combo_str: str) -> torch.Tensor:
        """
        将通道组合字符串转换为多标签向量
        
        Args:
            combo_str: "Fp1-F3-C3" 或 "Fp1, F3, C3"
        
        Returns:
            label_vector: [num_channels] 的二值向量
        """
        label_vector = np.zeros(self.num_channels, dtype=np.float32)
        
        # 处理不同的分隔符
        for sep in ['-', ',', ';', ' ']:
            if sep in combo_str:
                channels = combo_str.split(sep)
                break
        else:
            channels = [combo_str]  # 单个通道
        
        # 标记存在的通道
        for ch in channels:
            ch = ch.strip()
            if ch in self.channel_to_idx:
                label_vector[self.channel_to_idx[ch]] = 1.0
            else:
                # 通道不在标准列表中，记录警告
                if ch:  # 非空
                    pass  # print(f"Warning: Unknown channel '{ch}'")
        
        return torch.from_numpy(label_vector)
    
    def _get_channel_combination(self, npz_file: str) -> str:
        """从CSV获取通道组合"""
        for _, row in self.labels_df.iterrows():
            features_dir_path = row['features_dir_path']
            if features_dir_path in npz_file:
                return row['channel_combination']
        return ""
    
    def __getitem__(self, idx: int):
        # 这里复用原来的数据加载逻辑
        # 简化版本，实际使用时应该整合到datasets.py
        
        from .datasets import _safe_load_npz, build_graph_from_matrix
        
        npz_file = self.npz_paths[idx]
        arrays = _safe_load_npz(npz_file)
        
        # 加载矩阵（简化版，实际应使用fusion）
        for key in self.matrix_keys:
            if key in arrays:
                matrix = arrays[key]
                break
        else:
            # 使用第一个可用的矩阵
            matrix = arrays[list(arrays.keys())[0]]
        
        adj, node_feat = build_graph_from_matrix(matrix, self.topk_ratio)
        
        # 获取多标签
        channel_combo = self._get_channel_combination(npz_file)
        multi_labels = self._parse_channel_combination(channel_combo)
        
        return {
            "adj": adj,
            "x": node_feat,
            "y": multi_labels,  # [num_channels] 多标签向量
            "n": torch.tensor(adj.shape[0], dtype=torch.long),
            "path": npz_file,
            "combo": channel_combo  # 保留原始字符串用于调试
        }


# ============================================================================
# 模型类
# ============================================================================

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
        dropout: float = 0.5,
        use_attention: bool = False
    ):
        """
        Args:
            in_dim: 节点特征维度
            hidden_dim: 隐藏层维度
            num_channels: 通道数量（输出维度）
            dropout: Dropout比例
            use_attention: 是否使用注意力机制
        """
        super().__init__()
        
        from .models import GCNEncoder
        
        self.encoder = GCNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=3,
            dropout=dropout
        )
        
        # 多标签分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_channels)
        )
        
        # 可选：注意力机制（用于通道重要性）
        if use_attention:
            self.channel_attention = nn.Sequential(
                nn.Linear(hidden_dim, num_channels),
                nn.Softmax(dim=-1)
            )
        else:
            self.channel_attention = None
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor):
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
        
        # 可选：应用注意力
        if self.channel_attention is not None:
            attention = self.channel_attention(g)  # [B, num_channels]
            logits = logits * attention
        
        return logits


# ============================================================================
# 训练和评估
# ============================================================================

def train_epoch(model, loader, optimizer, device, pos_weight=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    
    # BCE Loss with optional pos_weight for imbalanced data
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for batch in loader:
        adj = batch['adj'].to(device)
        x = batch['x'].to(device)
        labels = batch['y'].to(device)  # [B, num_channels]
        
        # 前向传播
        logits = model(x, adj)  # [B, num_channels]
        loss = criterion(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, channel_names):
    """评估模型"""
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    for batch in loader:
        adj = batch['adj'].to(device)
        x = batch['x'].to(device)
        labels = batch['y'].cpu().numpy()  # [B, num_channels]
        
        logits = model(x, adj)
        probs = torch.sigmoid(logits).cpu().numpy()  # [B, num_channels]
        preds = (probs > 0.5).astype(np.float32)
        
        all_labels.append(labels)
        all_preds.append(preds)
        all_probs.append(probs)
    
    # 合并所有批次
    all_labels = np.vstack(all_labels)  # [N, num_channels]
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    
    # 计算指标
    metrics = {}
    
    # 1. Hamming Loss (预测错误的标签比例)
    metrics['hamming_loss'] = hamming_loss(all_labels, all_preds)
    
    # 2. Jaccard Score (交并比)
    metrics['jaccard_macro'] = jaccard_score(all_labels, all_preds, average='macro')
    metrics['jaccard_samples'] = jaccard_score(all_labels, all_preds, average='samples')
    
    # 3. Average Precision (AP)
    metrics['map'] = average_precision_score(all_labels, all_probs, average='macro')
    
    # 4. 每个通道的指标
    print("\nPer-Channel Metrics:")
    print(classification_report(
        all_labels, all_preds,
        target_names=channel_names,
        zero_division=0
    ))
    
    return metrics, all_probs


# ============================================================================
# 可视化和解释
# ============================================================================

def visualize_predictions(probs, channel_names, threshold=0.5):
    """
    可视化预测结果
    
    Args:
        probs: [num_channels] 概率向量
        channel_names: 通道名称列表
        threshold: 判定阈值
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['red' if p > threshold else 'blue' for p in probs]
    bars = ax.bar(range(len(channel_names)), probs, color=colors, alpha=0.7)
    
    ax.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold={threshold}')
    ax.set_xlabel('Channels')
    ax.set_ylabel('Abnormality Probability')
    ax.set_title('Channel-wise Abnormality Prediction')
    ax.set_xticks(range(len(channel_names)))
    ax.set_xticklabels(channel_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def get_top_k_channels(probs, channel_names, k=5):
    """
    获取最可能异常的top-k通道
    
    Args:
        probs: [num_channels] 概率向量
        channel_names: 通道名称列表
        k: 返回top-k
    
    Returns:
        top_channels: [(channel_name, prob), ...]
    """
    indices = np.argsort(probs)[::-1][:k]
    top_channels = [(channel_names[i], probs[i]) for i in indices]
    return top_channels


# ============================================================================
# 示例使用
# ============================================================================

def example_usage():
    """示例：如何使用多标签分类"""
    
    # 1. 准备数据
    # ... (使用实际的数据路径)
    
    # 2. 创建模型
    model = MultiLabelGNNClassifier(
        in_dim=2,
        hidden_dim=128,
        num_channels=len(ALL_CHANNELS),
        dropout=0.5
    )
    
    # 3. 训练
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # for epoch in range(100):
    #     train_loss = train_epoch(model, train_loader, optimizer, device)
    #     metrics, _ = evaluate(model, val_loader, device, ALL_CHANNELS)
    #     print(f"Epoch {epoch}: Loss={train_loss:.4f}, Jaccard={metrics['jaccard_samples']:.4f}")
    
    # 4. 预测
    # model.eval()
    # with torch.no_grad():
    #     logits = model(x, adj)
    #     probs = torch.sigmoid(logits)[0].cpu().numpy()  # 第一个样本
    
    # 5. 解释结果
    # top_channels = get_top_k_channels(probs, ALL_CHANNELS, k=5)
    # print("Top 5 most likely abnormal channels:")
    # for ch, prob in top_channels:
    #     print(f"  {ch}: {prob:.3f}")
    
    pass


if __name__ == "__main__":
    print("Multi-Label Classification for EEG Channel Prediction")
    print("=" * 60)
    print(f"Number of channels: {len(ALL_CHANNELS)}")
    print(f"Channels: {ALL_CHANNELS}")
    print("\nThis module provides:")
    print("  - MultiLabelConnectivityDataset: 数据集类")
    print("  - MultiLabelGNNClassifier: 模型类")
    print("  - train_epoch, evaluate: 训练和评估函数")
    print("  - visualize_predictions: 可视化工具")
