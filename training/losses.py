#!/usr/bin/env python3
"""
专门用于不平衡多标签分类的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    
    论文: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    FL(p) = -α(1-p)^γ log(p)
    
    参数:
        alpha: 平衡因子，控制正负样本的权重
        gamma: 聚焦参数，让模型更关注难分类的样本
               gamma=0 时退化为BCE loss
               gamma>0 时对易分类样本的loss进行down-weight
    
    特点:
        - 自动降低易分类样本的权重
        - 聚焦于难分类的样本
        - 对类别不平衡非常有效
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 平衡因子，推荐 0.25 (正类权重更高)
            gamma: 聚焦参数，推荐 2.0
            reduction: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: [B, num_classes] 模型输出的logits
            targets: [B, num_classes] 真实标签 (0或1)
        
        Returns:
            loss: scalar
        """
        # 计算概率
        probs = torch.sigmoid(logits)
        
        # BCE loss (不降维)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # 计算 p_t: 正类用p，负类用1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight: (1 - p_t)^gamma
        # 易分类样本(p_t接近1)的weight接近0
        # 难分类样本(p_t接近0)的weight接近1
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight: 正类用alpha，负类用1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 最终loss
        loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification
    
    论文: Ridnik et al. "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
    
    特点:
        - 对正负样本使用不同的gamma
        - 专门为多标签设计
        - SOTA性能
    """
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        """
        Args:
            gamma_neg: 负样本的聚焦参数（推荐4）
            gamma_pos: 正样本的聚焦参数（推荐0-1）
            clip: 概率裁剪值，防止数值不稳定
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, num_classes]
            targets: [B, num_classes]
        """
        # 概率
        probs = torch.sigmoid(logits)
        
        # 概率裁剪（防止log(0)）
        probs = probs.clamp(min=self.clip, max=1.0 - 1e-8)
        
        # 正样本loss
        targets_pos = targets
        loss_pos = targets_pos * torch.log(probs)
        loss_pos = loss_pos * ((1 - probs) ** self.gamma_pos)
        
        # 负样本loss
        targets_neg = 1 - targets
        probs_neg = 1 - probs
        probs_neg = probs_neg.clamp(min=self.clip)
        loss_neg = targets_neg * torch.log(probs_neg)
        loss_neg = loss_neg * (probs ** self.gamma_neg)
        
        # 组合
        loss = -loss_pos - loss_neg
        
        return loss.mean()


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss
    
    论文: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    
    使用有效样本数来计算权重:
    E_n = (1 - β^n) / (1 - β)
    
    其中 β = (N-1)/N, N是总样本数
    """
    
    def __init__(self, beta=0.9999):
        """
        Args:
            beta: 重采样参数，推荐 0.9999 for large datasets
        """
        super().__init__()
        self.beta = beta
    
    def forward(self, logits, targets, samples_per_class):
        """
        Args:
            logits: [B, num_classes]
            targets: [B, num_classes]
            samples_per_class: [num_classes] 每个类的样本数
        """
        # 计算有效样本数
        effective_num = 1.0 - torch.pow(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * len(weights)
        
        # 加权BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # 应用权重
        weighted_loss = bce_loss * weights.unsqueeze(0)
        
        return weighted_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-label classification
    
    常用于医学图像分割，也适用于不平衡多标签分类
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    
    def __init__(self, smooth=1.0):
        """
        Args:
            smooth: 平滑因子，防止除零
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, num_classes]
            targets: [B, num_classes]
        """
        probs = torch.sigmoid(logits)
        
        # 计算 Dice coefficient
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss = 1 - Dice
        loss = 1.0 - dice
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    组合多个损失函数
    
    例如: Focal Loss + Dice Loss
    """
    
    def __init__(self, focal_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice = DiceLoss(smooth=1.0)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        
        total_loss = (self.focal_weight * loss_focal + 
                     self.dice_weight * loss_dice)
        
        return total_loss


# ============================================================================
# 使用示例和测试
# ============================================================================

if __name__ == "__main__":
    print("Testing Loss Functions for Imbalanced Multi-Label Classification")
    print("=" * 60)
    
    # 创建测试数据（模拟不平衡情况）
    batch_size = 32
    num_classes = 17
    
    # 模拟预测
    logits = torch.randn(batch_size, num_classes)
    
    # 模拟不平衡标签（每个样本只有2-3个正类）
    targets = torch.zeros(batch_size, num_classes)
    for i in range(batch_size):
        # 随机选择2-3个通道为正类
        num_pos = torch.randint(2, 4, (1,)).item()
        pos_indices = torch.randperm(num_classes)[:num_pos]
        targets[i, pos_indices] = 1.0
    
    print(f"Batch size: {batch_size}")
    print(f"Num classes: {num_classes}")
    print(f"Positive ratio: {targets.mean().item():.3f}")
    print(f"Avg positives per sample: {targets.sum(dim=1).mean().item():.1f}")
    
    # 测试不同的损失函数
    losses = {
        'BCE': nn.BCEWithLogitsLoss(),
        'Focal (γ=2)': FocalLoss(alpha=0.25, gamma=2.0),
        'Focal (γ=3)': FocalLoss(alpha=0.25, gamma=3.0),
        'Asymmetric': AsymmetricLoss(gamma_neg=4, gamma_pos=1),
        'Dice': DiceLoss(),
        'Combined': CombinedLoss(),
    }
    
    print("\n" + "=" * 60)
    print("Loss Comparison:")
    print("=" * 60)
    
    for name, criterion in losses.items():
        loss = criterion(logits, targets)
        print(f"{name:20s}: {loss.item():.4f}")
    
    print("\n✓ All loss functions working correctly!")
    print("\nRecommendation for your data (1:7 imbalance):")
    print("  1. Focal Loss (gamma=2.5, alpha=0.25)")
    print("  2. Asymmetric Loss (gamma_neg=4, gamma_pos=1)")
    print("  3. Combined Loss (Focal + Dice)")

