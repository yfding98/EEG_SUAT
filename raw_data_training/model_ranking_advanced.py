"""
高级排序模型 - 集成最新研究成果

改进点:
1. 多频段特征提取 (Lancet Neurology 2023)
2. 图卷积网络 - 空间约束 (Nature Communications 2023)
3. 通道注意力机制 (Nature Medicine 2022)
4. 对比学习损失 (Nature Medicine 2022)
5. 生理约束损失 (Brain 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================================
# 1. 多频段特征提取
# ============================================================================
class BandSpecificEncoder(nn.Module):
    """
    频段特异性编码器
    
    文献依据: Lancet Neurology 2023
    发作前期在不同频段有不同表现:
    - Delta/Theta: 局部功率增加
    - Alpha: 去同步
    - Beta/Gamma: 活动增强
    - HFO (80-250Hz): 最可靠的标志物
    """
    def __init__(self, n_samples, d_model, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, d_model // 4, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(d_model // 4, d_model // 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(d_model // 4)
        self.bn2 = nn.BatchNorm1d(d_model // 2)
        self.bn3 = nn.BatchNorm1d(d_model)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, n_channels, n_samples)
        batch_size, n_channels, n_samples = x.shape
        
        # Process each channel
        x = x.view(batch_size * n_channels, 1, n_samples)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # (batch*n_channels, d_model)
        
        # Reshape back
        x = x.view(batch_size, n_channels, -1)  # (batch, n_channels, d_model)
        
        return x


class MultiBandFeatureExtractor(nn.Module):
    """
    多频段特征提取器
    
    针对不同频段提取特征，然后融合
    """
    def __init__(self, n_samples, d_model, n_bands=6, dropout=0.1):
        super().__init__()
        
        self.band_encoders = nn.ModuleList([
            BandSpecificEncoder(n_samples, d_model, dropout)
            for _ in range(n_bands)
        ])
        
        # Band fusion with attention
        self.band_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.fusion_norm = nn.LayerNorm(d_model)
        
    def forward(self, x_bands):
        """
        Args:
            x_bands: list of (batch, n_channels, n_samples), one per band
        
        Returns:
            features: (batch, n_channels, d_model)
        """
        # Encode each band
        band_features = []
        for i, x_band in enumerate(x_bands):
            feat = self.band_encoders[i](x_band)
            band_features.append(feat)
        
        # Stack: (n_bands, batch, n_channels, d_model)
        band_features = torch.stack(band_features, dim=0)
        n_bands, batch_size, n_channels, d_model = band_features.shape
        
        # Reshape for attention: (batch*n_channels, n_bands, d_model)
        band_features = band_features.permute(1, 2, 0, 3).reshape(
            batch_size * n_channels, n_bands, d_model
        )
        
        # Cross-band attention
        fused, _ = self.band_attention(
            band_features, band_features, band_features
        )
        
        # Average over bands
        fused = fused.mean(dim=1)  # (batch*n_channels, d_model)
        
        # Reshape back
        fused = fused.view(batch_size, n_channels, d_model)
        
        fused = self.fusion_norm(fused)
        
        return fused


# ============================================================================
# 2. 图卷积网络 (GCN) - 空间约束
# ============================================================================
class GraphConvolution(nn.Module):
    """
    图卷积层
    
    文献依据: Nature Communications 2023
    利用通道的空间拓扑结构
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, n_channels, in_features)
            adj: (n_channels, n_channels) adjacency matrix
        
        Returns:
            out: (batch, n_channels, out_features)
        """
        # Linear transformation
        support = torch.matmul(x, self.weight)  # (batch, n_channels, out_features)
        
        # Graph convolution
        output = torch.matmul(adj.unsqueeze(0), support)  # (batch, n_channels, out_features)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class SpatialGCN(nn.Module):
    """
    空间图卷积网络
    
    利用通道的物理位置构建邻接矩阵
    """
    def __init__(self, d_model, n_gcn_layers=2, dropout=0.1):
        super().__init__()
        
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(d_model, d_model)
            for _ in range(n_gcn_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_gcn_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_matrix):
        """
        Args:
            x: (batch, n_channels, d_model)
            adj_matrix: (n_channels, n_channels)
        
        Returns:
            out: (batch, n_channels, d_model)
        """
        for gcn, norm in zip(self.gcn_layers, self.norms):
            residual = x
            x = F.relu(gcn(x, adj_matrix))
            x = self.dropout(x)
            x = norm(x + residual)  # Residual connection
        
        return x


def compute_spatial_adjacency(channel_positions, k=5, method='knn'):
    """
    根据通道位置计算邻接矩阵
    
    Args:
        channel_positions: (n_channels, 3) 通道的3D坐标
        k: K近邻数量
        method: 'knn' or 'threshold'
    
    Returns:
        adj_matrix: (n_channels, n_channels)
    """
    n_channels = channel_positions.shape[0]
    
    # 计算距离矩阵
    dist_matrix = torch.cdist(channel_positions, channel_positions)
    
    if method == 'knn':
        # K近邻
        _, indices = torch.topk(dist_matrix, k + 1, largest=False, dim=1)
        adj_matrix = torch.zeros(n_channels, n_channels)
        for i in range(n_channels):
            adj_matrix[i, indices[i]] = 1.0
    else:
        # 阈值法
        threshold = dist_matrix.median()
        adj_matrix = (dist_matrix < threshold).float()
    
    # 对称化
    adj_matrix = (adj_matrix + adj_matrix.t()) / 2
    
    # 归一化（D^{-1/2} A D^{-1/2}）
    degree = adj_matrix.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    adj_matrix = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    
    return adj_matrix


# ============================================================================
# 3. 通道注意力机制
# ============================================================================
class ChannelAttention(nn.Module):
    """
    通道注意力
    
    文献依据: Nature Medicine 2022
    自动学习哪些通道更重要
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, d_model)
        
        Returns:
            out: (batch, n_channels, d_model)
            attention_weights: (batch, n_heads, n_channels, n_channels)
        """
        # Self-attention
        attn_out, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights


# ============================================================================
# 4. 主模型
# ============================================================================
class AdvancedChannelRankingModel(nn.Module):
    """
    高级通道排序模型
    
    集成:
    1. 多频段特征提取
    2. 空间图卷积
    3. 通道注意力
    """
    def __init__(
        self,
        n_channels,
        n_samples,
        d_model=256,
        n_heads=8,
        n_layers=4,
        n_gcn_layers=2,
        dropout=0.3,
        use_multiband=True,
        use_gcn=True,
        channel_positions=None
    ):
        super().__init__()
        
        self.use_multiband = use_multiband
        self.use_gcn = use_gcn
        
        # 1. Feature extraction
        if use_multiband:
            self.feature_extractor = MultiBandFeatureExtractor(
                n_samples, d_model, n_bands=6, dropout=dropout
            )
        else:
            self.feature_extractor = BandSpecificEncoder(
                n_samples, d_model, dropout
            )
        
        # 2. Spatial GCN (optional)
        if use_gcn and channel_positions is not None:
            self.gcn = SpatialGCN(d_model, n_gcn_layers, dropout)
            self.adj_matrix = compute_spatial_adjacency(
                torch.tensor(channel_positions, dtype=torch.float32)
            )
        else:
            self.gcn = None
            self.adj_matrix = None
        
        # 3. Channel attention layers
        self.attention_layers = nn.ModuleList([
            ChannelAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # 4. Channel scoring head
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Store attention weights for visualization
        self.attention_weights = []
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, n_channels, n_samples) or list of band-specific inputs
        
        Returns:
            scores: (batch, n_channels) - abnormality scores [0, 1]
        """
        # 1. Feature extraction
        if self.use_multiband:
            # x should be a list of band-specific inputs
            features = self.feature_extractor(x)
        else:
            features = self.feature_extractor(x)
        
        # 2. Spatial GCN
        if self.gcn is not None:
            adj = self.adj_matrix.to(features.device)
            features = self.gcn(features, adj)
        
        # 3. Channel attention
        self.attention_weights = []
        for attn_layer in self.attention_layers:
            features, attn_w = attn_layer(features)
            if return_attention:
                self.attention_weights.append(attn_w)
        
        # 4. Scoring
        scores = self.score_head(features).squeeze(-1)  # (batch, n_channels)
        
        if return_attention:
            return scores, self.attention_weights
        
        return scores


# ============================================================================
# 5. 改进的损失函数
# ============================================================================
class AdvancedChannelRankingLoss(nn.Module):
    """
    高级排序损失函数
    
    组成:
    1. Score Loss (BCE)
    2. Margin Loss (ranking)
    3. Top-K Loss (IoU)
    4. Contrastive Loss (对比学习)
    5. Spatial Clustering Loss (空间约束)
    6. Network Coherence Loss (连接性约束)
    """
    def __init__(
        self,
        score_weight=3.0,
        margin_weight=1.0,
        topk_weight=2.0,
        contrastive_weight=1.0,
        spatial_weight=0.5,
        network_weight=0.5,
        margin=0.15,
        temperature=0.07,
        expected_k=(2, 5)
    ):
        super().__init__()
        self.score_weight = score_weight
        self.margin_weight = margin_weight
        self.topk_weight = topk_weight
        self.contrastive_weight = contrastive_weight
        self.spatial_weight = spatial_weight
        self.network_weight = network_weight
        self.margin = margin
        self.temperature = temperature
        self.expected_k = expected_k
        
        self.bce_loss = nn.BCELoss()
    
    def score_loss(self, pred_scores, true_labels):
        """Binary Cross Entropy"""
        return self.bce_loss(pred_scores, true_labels.float())
    
    def margin_loss(self, pred_scores, true_labels):
        """
        Margin Ranking Loss
        活跃通道分数应该比非活跃通道高至少margin
        """
        batch_size = pred_scores.size(0)
        device = pred_scores.device
        
        total_loss = torch.tensor(0.0, device=device)  # 修复：使用tensor
        n_pairs = 0
        
        for i in range(batch_size):
            scores = pred_scores[i]
            labels = true_labels[i]
            
            active_idx = (labels == 1).nonzero(as_tuple=True)[0]
            inactive_idx = (labels == 0).nonzero(as_tuple=True)[0]
            
            if len(active_idx) == 0 or len(inactive_idx) == 0:
                continue
            
            active_scores = scores[active_idx]
            inactive_scores = scores[inactive_idx]
            
            # 计算平均分数差
            score_diff = active_scores.mean() - inactive_scores.mean()
            
            # Hinge loss
            loss = F.relu(self.margin - score_diff)
            
            total_loss += loss
            n_pairs += 1
        
        if n_pairs == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / n_pairs
    
    def topk_loss(self, pred_scores, true_labels):
        """
        Top-K IoU Loss
        预测的top-K应该与真实活跃通道重叠
        """
        batch_size = pred_scores.size(0)
        device = pred_scores.device
        
        total_iou_loss = torch.tensor(0.0, device=device)  # 修复：使用tensor
        
        for i in range(batch_size):
            scores = pred_scores[i]
            labels = true_labels[i]
            
            true_k = (labels == 1).sum().item()
            
            if true_k == 0:
                continue
            
            # 动态K
            k = max(self.expected_k[0], min(true_k, self.expected_k[1]))
            
            # 预测top-K
            topk_idx = scores.topk(k).indices
            pred_mask = torch.zeros_like(labels, dtype=torch.bool)
            pred_mask[topk_idx] = True
            
            true_mask = labels == 1
            
            # IoU
            intersection = (pred_mask & true_mask).sum().float()
            union = (pred_mask | true_mask).sum().float()
            
            iou = intersection / (union + 1e-8)
            
            # IoU loss (maximize IoU)
            total_iou_loss += (1 - iou)
        
        return total_iou_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=device)
    
    def contrastive_loss(self, embeddings, labels):
        """
        对比学习损失
        
        文献依据: Nature Medicine 2022
        拉近异常通道，推远正常通道
        
        Args:
            embeddings: (batch, n_channels, d_model) from model
            labels: (batch, n_channels) binary labels
        """
        batch_size, n_channels, d_model = embeddings.shape
        device = embeddings.device
        
        total_loss = torch.tensor(0.0, device=device)  # 修复：使用tensor
        n_samples = 0
        
        for i in range(batch_size):
            emb = embeddings[i]  # (n_channels, d_model)
            lab = labels[i]  # (n_channels,)
            
            active_idx = (lab == 1).nonzero(as_tuple=True)[0]
            
            if len(active_idx) < 2:
                continue
            
            # Normalize embeddings
            emb_norm = F.normalize(emb, dim=-1)
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(emb_norm, emb_norm.t()) / self.temperature
            
            # Positive pairs: both active
            pos_mask = (lab.unsqueeze(1) == 1) & (lab.unsqueeze(0) == 1)
            pos_mask.fill_diagonal_(False)
            
            # Negative pairs: active vs inactive
            neg_mask = (lab.unsqueeze(1) == 1) & (lab.unsqueeze(0) == 0)
            
            if pos_mask.sum() == 0:
                continue
            
            # InfoNCE loss
            pos_sim = sim_matrix[pos_mask]
            neg_sim = sim_matrix[neg_mask]
            
            # For each positive pair, compute contrastive loss
            logits = torch.cat([pos_sim.unsqueeze(1), 
                               neg_sim.view(-1, neg_sim.size(0) // pos_sim.size(0))], dim=1)
            
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            
            loss = F.cross_entropy(logits, targets)
            
            total_loss += loss
            n_samples += 1
        
        if n_samples == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / n_samples
    
    def spatial_clustering_loss(self, pred_scores, channel_positions):
        """
        空间聚类损失
        
        文献依据: Brain 2023
        异常通道应该在空间上聚集
        
        Args:
            pred_scores: (batch, n_channels)
            channel_positions: (n_channels, 3) or None
        """
        if channel_positions is None:
            return torch.tensor(0.0, device=pred_scores.device)
        
        batch_size = pred_scores.size(0)
        total_loss = torch.tensor(0.0, device=pred_scores.device)  # 修复：使用tensor
        
        threshold = 0.5
        
        for i in range(batch_size):
            scores = pred_scores[i]
            
            # 高分通道
            active_mask = scores > threshold
            n_active = active_mask.sum()
            
            if n_active < 2:
                continue
            
            active_positions = channel_positions[active_mask]
            
            # 计算平均距离（希望小）
            distances = torch.pdist(active_positions)
            avg_distance = distances.mean()
            
            total_loss += avg_distance
        
        return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=pred_scores.device)
    
    def network_coherence_loss(self, pred_scores, connectivity_matrix):
        """
        网络连贯性损失
        
        文献依据: Brain 2023
        异常通道间应该有更强的连接性
        
        Args:
            pred_scores: (batch, n_channels)
            connectivity_matrix: (n_channels, n_channels) or None
        """
        if connectivity_matrix is None:
            return torch.tensor(0.0, device=pred_scores.device)
        
        batch_size = pred_scores.size(0)
        total_loss = torch.tensor(0.0, device=pred_scores.device)  # 修复：使用tensor
        
        for i in range(batch_size):
            scores = pred_scores[i]
            
            # Weighted connectivity
            weights = torch.outer(scores, scores)
            coherence = (connectivity_matrix * weights).sum()
            
            # Normalize by number of edges
            n_channels = scores.size(0)
            coherence = coherence / (n_channels * n_channels)
            
            # Maximize coherence (minimize negative)
            total_loss += -coherence
        
        return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=pred_scores.device)
    
    def forward(
        self, 
        pred_scores, 
        true_labels, 
        embeddings=None,
        channel_positions=None,
        connectivity_matrix=None
    ):
        """
        Total loss
        
        Args:
            pred_scores: (batch, n_channels)
            true_labels: (batch, n_channels)
            embeddings: (batch, n_channels, d_model) optional
            channel_positions: (n_channels, 3) optional
            connectivity_matrix: (n_channels, n_channels) optional
        """
        # Core losses
        loss_score = self.score_loss(pred_scores, true_labels)
        loss_margin = self.margin_loss(pred_scores, true_labels)
        loss_topk = self.topk_loss(pred_scores, true_labels)
        
        total_loss = (
            self.score_weight * loss_score +
            self.margin_weight * loss_margin +
            self.topk_weight * loss_topk
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'score_loss': loss_score.item(),
            'margin_loss': loss_margin.item(),
            'topk_loss': loss_topk.item()
        }
        
        # Optional losses
        if embeddings is not None and self.contrastive_weight > 0:
            loss_contrastive = self.contrastive_loss(embeddings, true_labels)
            total_loss += self.contrastive_weight * loss_contrastive
            loss_dict['contrastive_loss'] = loss_contrastive.item()
        
        if channel_positions is not None and self.spatial_weight > 0:
            loss_spatial = self.spatial_clustering_loss(pred_scores, channel_positions)
            total_loss += self.spatial_weight * loss_spatial
            loss_dict['spatial_loss'] = loss_spatial.item()
        
        if connectivity_matrix is not None and self.network_weight > 0:
            loss_network = self.network_coherence_loss(pred_scores, connectivity_matrix)
            total_loss += self.network_weight * loss_network
            loss_dict['network_loss'] = loss_network.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


# ============================================================================
# Factory Functions
# ============================================================================
def create_advanced_ranking_model(
    n_channels,
    n_samples,
    d_model=256,
    n_heads=8,
    n_layers=4,
    dropout=0.3,
    use_multiband=True,
    use_gcn=True,
    channel_positions=None
):
    """
    创建高级排序模型
    
    Args:
        n_channels: 通道数
        n_samples: 时间样本数
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: Transformer层数
        dropout: Dropout率
        use_multiband: 是否使用多频段特征
        use_gcn: 是否使用图卷积
        channel_positions: (n_channels, 3) 通道位置坐标，用于GCN
    """
    model = AdvancedChannelRankingModel(
        n_channels=n_channels,
        n_samples=n_samples,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_gcn_layers=2,
        dropout=dropout,
        use_multiband=use_multiband,
        use_gcn=use_gcn,
        channel_positions=channel_positions
    )
    
    return model


if __name__ == "__main__":
    # 测试
    batch_size = 4
    n_channels = 19
    n_samples = 1536  # 6秒 @ 256Hz
    
    # 模拟多频段输入
    x_delta = torch.randn(batch_size, n_channels, n_samples)
    x_theta = torch.randn(batch_size, n_channels, n_samples)
    x_alpha = torch.randn(batch_size, n_channels, n_samples)
    x_beta = torch.randn(batch_size, n_channels, n_samples)
    x_gamma = torch.randn(batch_size, n_channels, n_samples)
    x_hfo = torch.randn(batch_size, n_channels, n_samples)
    
    x_bands = [x_delta, x_theta, x_alpha, x_beta, x_gamma, x_hfo]
    
    # 模拟通道位置（10-20系统）
    channel_positions = torch.randn(n_channels, 3)
    
    # 创建模型
    model = create_advanced_ranking_model(
        n_channels=n_channels,
        n_samples=n_samples,
        d_model=256,
        use_multiband=True,
        use_gcn=True,
        channel_positions=channel_positions.numpy()
    )
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward
    scores = model(x_bands)
    print(f"输出形状: {scores.shape}")
    print(f"分数范围: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # 测试损失函数
    labels = torch.zeros(batch_size, n_channels)
    labels[:, :3] = 1  # 前3个通道是异常的
    
    criterion = AdvancedChannelRankingLoss()
    
    loss, loss_dict = criterion(scores, labels)
    
    print(f"\n损失:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

