#!/usr/bin/env python3
"""
高级损失函数 - 处理多层次不平衡问题

包含:
1. ChannelAdaptiveFocalLoss - 每个通道不同参数
2. AdaptiveLoss - 动态调整权重
3. OHEMLoss - 难例挖掘
4. TemporalConsistencyLoss - 时序一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAdaptiveFocalLoss(nn.Module):
    """
    通道自适应 Focal Loss
    
    为每个通道使用不同的 focal 参数:
    - 高频通道: gamma 小（已经容易学习）
    - 低频通道: gamma 大（需要强烈聚焦）
    
    理论:
        频率低的通道 → 样本少 → 难学习 → 需要更大的gamma来聚焦
    """
    
    def __init__(self, channel_frequencies, base_gamma=2.5, base_alpha=0.25):
        """
        Args:
            channel_frequencies: [num_channels] 每个通道的样本数
            base_gamma: 基础gamma值
            base_alpha: 基础alpha值
        """
        super().__init__()
        self.num_channels = len(channel_frequencies)
        
        # 归一化频率
        max_freq = max(channel_frequencies) if max(channel_frequencies) > 0 else 1.0
        min_freq = min([f for f in channel_frequencies if f > 0] + [1.0])
        
        # 计算每个通道的gamma和alpha
        channel_gammas = []
        channel_alphas = []
        
        for freq in channel_frequencies:
            if freq == 0:
                # 没有样本的通道：使用最大gamma
                gamma = 5.0
                alpha = 0.5
            else:
                # gamma: 反比于频率，频率低→gamma高
                # 使用对数缩放，避免极端值
                freq_ratio = np.log(max_freq + 1) / np.log(freq + 1)
                gamma = base_gamma * freq_ratio
                gamma = min(gamma, 8.0)  # 上限8.0
                gamma = max(gamma, 1.0)  # 下限1.0
                
                # alpha: 正类权重，频率低→权重高
                alpha = base_alpha * freq_ratio
                alpha = min(alpha, 0.6)
                alpha = max(alpha, 0.15)
            
            channel_gammas.append(gamma)
            channel_alphas.append(alpha)
        
        self.register_buffer('channel_gammas', torch.tensor(channel_gammas, dtype=torch.float32))
        self.register_buffer('channel_alphas', torch.tensor(channel_alphas, dtype=torch.float32))
        
        print(f"\nChannelAdaptiveFocalLoss initialized:")
        print(f"  Gamma range: [{self.channel_gammas.min():.2f}, {self.channel_gammas.max():.2f}]")
        print(f"  Alpha range: [{self.channel_alphas.min():.2f}, {self.channel_alphas.max():.2f}]")
    
    def forward(self, logits, targets, channel_indices=None):
        """
        Args:
            logits: [B, N] - N可能是所有通道或部分通道
            targets: [B, N]
            channel_indices: 可选，长度为N的通道索引列表
                           用于课程学习时指定使用哪些通道的参数
        """
        probs = torch.sigmoid(logits)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # p_t: 正类用p，负类用1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 每个通道不同的focal weight
        num_active_channels = logits.shape[1]
        
        if channel_indices is not None:
            # 课程学习：使用指定通道的参数
            # channel_indices: [idx1, idx2, ...] → 获取对应的gamma/alpha
            gammas = self.channel_gammas[channel_indices].unsqueeze(0)  # [1, N]
            alphas = self.channel_alphas[channel_indices].unsqueeze(0)  # [1, N]
        elif num_active_channels < len(self.channel_gammas):
            # 没有提供索引但通道数少于总数：假设是前N个（向后兼容）
            gammas = self.channel_gammas[:num_active_channels].unsqueeze(0)  # [1, N]
            alphas = self.channel_alphas[:num_active_channels].unsqueeze(0)  # [1, N]
        else:
            # 全部通道
            gammas = self.channel_gammas.unsqueeze(0)  # [1, num_channels]
            alphas = self.channel_alphas.unsqueeze(0)  # [1, num_channels]
        
        # 确保 gammas 和 alphas 在与 logits 相同的设备上
        gammas = gammas.to(logits.device)
        alphas = alphas.to(logits.device)
        
        focal_weight = (1 - p_t) ** gammas
        
        # 每个通道不同的alpha weight
        alpha_weight = alphas * targets + (1 - alphas) * (1 - targets)
        
        # 最终loss
        loss = alpha_weight * focal_weight * bce_loss
        
        return loss.mean()


class AdaptiveLoss(nn.Module):
    """
    自适应损失 - 根据每个通道的训练表现动态调整权重
    
    表现差的通道在下一个epoch获得更高权重
    """
    
    def __init__(self, num_channels, base_criterion=None):
        super().__init__()
        self.num_channels = num_channels
        
        # 可学习的通道权重
        self.channel_weights = nn.Parameter(torch.ones(num_channels))
        
        # 基础损失函数
        if base_criterion is None:
            from losses import AsymmetricLoss
            self.base_criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
        else:
            self.base_criterion = base_criterion
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, num_channels]
            targets: [B, num_channels]
        """
        # 基础loss（每个样本每个通道的loss）
        base_loss = self.base_criterion(logits, targets)
        
        # 应用可学习的通道权重
        # softmax确保权重为正且和为num_channels
        normalized_weights = F.softmax(self.channel_weights, dim=0) * self.num_channels
        
        # 加权
        if isinstance(base_loss, torch.Tensor) and base_loss.dim() > 0:
            weighted_loss = base_loss * normalized_weights.unsqueeze(0)
        else:
            # 如果base_loss已经是标量
            weighted_loss = base_loss
        
        return weighted_loss.mean()
    
    def update_weights_from_performance(self, channel_f1_scores):
        """
        根据通道表现更新权重
        
        Args:
            channel_f1_scores: [num_channels] F1分数
        """
        with torch.no_grad():
            # 表现差的通道权重更高
            performance_weights = 1.0 - channel_f1_scores
            performance_weights = performance_weights / (performance_weights.sum() + 1e-6)
            
            # 指数移动平均更新
            self.channel_weights.data = (
                0.7 * self.channel_weights.data + 
                0.3 * performance_weights.to(self.channel_weights.device)
            )


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining Loss
    
    只用最难的样本计算loss，忽略简单样本
    """
    
    def __init__(self, base_criterion, keep_ratio=0.7):
        """
        Args:
            base_criterion: 基础损失函数
            keep_ratio: 保留多少比例的难样本
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.keep_ratio = keep_ratio
    
    def forward(self, logits, targets):
        """
        只保留最难的 keep_ratio% 样本
        """
        batch_size = logits.size(0)
        
        # 计算每个样本的loss（不降维）
        if hasattr(self.base_criterion, 'forward'):
            # 先计算完整loss
            with torch.no_grad():
                per_sample_loss = F.binary_cross_entropy_with_logits(
                    logits, targets, reduction='none'
                ).mean(dim=1)  # [B]
        else:
            per_sample_loss = self.base_criterion(logits, targets)
            if per_sample_loss.dim() > 1:
                per_sample_loss = per_sample_loss.mean(dim=1)
        
        # 选择最难的k个样本
        k = max(1, int(batch_size * self.keep_ratio))
        top_k_loss, top_k_indices = torch.topk(per_sample_loss, k)
        
        # 只用这些样本计算真实loss
        hard_logits = logits[top_k_indices]
        hard_targets = targets[top_k_indices]
        
        loss = self.base_criterion(hard_logits, hard_targets)
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    时序一致性损失
    
    鼓励模型在不同epoch对同一样本的预测保持一致
    """
    
    def __init__(self, num_samples, num_channels, alpha=0.6):
        """
        Args:
            num_samples: 训练集样本数
            num_channels: 通道数
            alpha: EMA系数
        """
        super().__init__()
        self.alpha = alpha
        self.num_samples = num_samples
        self.num_channels = num_channels
        
        # 历史预测（不参与梯度）
        self.register_buffer(
            'historical_predictions',
            torch.zeros(num_samples, num_channels)
        )
        self.register_buffer('initialized', torch.zeros(num_samples, dtype=torch.bool))
    
    def update_history(self, indices, predictions):
        """
        更新历史预测
        
        Args:
            indices: [B] batch中样本的索引
            predictions: [B, num_channels] 当前预测概率
        """
        with torch.no_grad():
            for i, idx in enumerate(indices):
                if self.initialized[idx]:
                    # EMA更新
                    self.historical_predictions[idx] = (
                        self.alpha * self.historical_predictions[idx] +
                        (1 - self.alpha) * predictions[i].cpu()
                    )
                else:
                    # 第一次见到这个样本
                    self.historical_predictions[idx] = predictions[i].cpu()
                    self.initialized[idx] = True
    
    def forward(self, indices, current_predictions):
        """
        计算一致性loss
        
        Args:
            indices: [B] batch中样本的索引
            current_predictions: [B, num_channels] 当前预测概率
        """
        # 获取历史预测
        historical = self.historical_predictions[indices].to(current_predictions.device)
        
        # 只对已初始化的样本计算一致性loss
        initialized_mask = self.initialized[indices].to(current_predictions.device)
        
        if initialized_mask.sum() == 0:
            return torch.tensor(0.0).to(current_predictions.device)
        
        # MSE loss（只计算已初始化的）
        diff = (current_predictions - historical) ** 2
        diff = diff[initialized_mask]
        
        return diff.mean()


class CombinedAdvancedLoss(nn.Module):
    """
    组合多个高级损失函数
    """
    
    def __init__(
        self,
        channel_frequencies,
        num_samples,
        focal_weight=0.6,
        ohem_weight=0.3,
        consistency_weight=0.1
    ):
        super().__init__()
        
        # 通道自适应Focal Loss
        self.focal_loss = ChannelAdaptiveFocalLoss(channel_frequencies)
        
        # OHEM Loss
        self.ohem_loss = OHEMLoss(
            AsymmetricLoss(gamma_neg=6.0, gamma_pos=0.0),
            keep_ratio=0.7
        )
        
        # 时序一致性Loss
        self.consistency_loss = TemporalConsistencyLoss(
            num_samples,
            len(channel_frequencies),
            alpha=0.6
        )
        
        self.focal_weight = focal_weight
        self.ohem_weight = ohem_weight
        self.consistency_weight = consistency_weight
    
    def forward(self, logits, targets, sample_indices=None):
        """
        Args:
            logits: [B, num_channels]
            targets: [B, num_channels]
            sample_indices: [B] 样本索引（用于一致性loss）
        """
        # 1. Focal Loss
        loss_focal = self.focal_loss(logits, targets)
        
        # 2. OHEM Loss
        loss_ohem = self.ohem_loss(logits, targets)
        
        # 3. 一致性Loss
        loss_consistency = torch.tensor(0.0).to(logits.device)
        if sample_indices is not None:
            probs = torch.sigmoid(logits)
            loss_consistency = self.consistency_loss(sample_indices, probs)
            # 更新历史
            self.consistency_loss.update_history(sample_indices, probs.detach())
        
        # 组合
        total_loss = (
            self.focal_weight * loss_focal +
            self.ohem_weight * loss_ohem +
            self.consistency_weight * loss_consistency
        )
        
        return total_loss, {
            'focal': loss_focal.item(),
            'ohem': loss_ohem.item(),
            'consistency': loss_consistency.item() if isinstance(loss_consistency, torch.Tensor) else 0.0
        }


# 导入基础loss
from losses import AsymmetricLoss


if __name__ == "__main__":
    print("Testing Advanced Loss Functions...")
    
    # 测试数据
    batch_size = 8
    num_channels = 10
    
    # 模拟频率（一些高频，一些低频）
    channel_freqs = [50, 45, 30, 25, 15, 10, 5, 3, 1, 0]
    
    # 测试ChannelAdaptiveFocalLoss
    print("\n1. ChannelAdaptiveFocalLoss:")
    loss_fn = ChannelAdaptiveFocalLoss(channel_freqs)
    
    logits = torch.randn(batch_size, num_channels)
    targets = torch.randint(0, 2, (batch_size, num_channels)).float()
    
    loss = loss_fn(logits, targets)
    print(f"  Loss: {loss.item():.4f}")
    
    # 测试AdaptiveLoss
    print("\n2. AdaptiveLoss:")
    loss_fn2 = AdaptiveLoss(num_channels)
    loss = loss_fn2(logits, targets)
    print(f"  Loss: {loss.item():.4f}")
    
    # 测试OHEM
    print("\n3. OHEMLoss:")
    loss_fn3 = OHEMLoss(AsymmetricLoss(), keep_ratio=0.7)
    loss = loss_fn3(logits, targets)
    print(f"  Loss: {loss.item():.4f}")
    
    print("\n✓ All advanced loss functions working!")

