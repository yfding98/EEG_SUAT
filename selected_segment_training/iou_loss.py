#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iou_loss.py

IoU损失函数模块
用于多标签分类中的IoU损失计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    """IoU损失函数"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred_logits, true_labels):
        """
        Args:
            pred_logits: (batch, n_channels) 预测logits
            true_labels: (batch, n_channels) 真实标签
        Returns:
            iou_loss: IoU损失值
        """
        # 将logits转换为概率
        pred_probs = torch.sigmoid(pred_logits)
        
        # 计算交集和并集
        intersection = (pred_probs * true_labels).sum(dim=1)  # (batch,)
        union = pred_probs.sum(dim=1) + true_labels.sum(dim=1) - intersection  # (batch,)
        
        # 计算IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # IoU损失 = 1 - IoU
        iou_loss = 1 - iou.mean()
        
        return iou_loss


class FocalIoULoss(nn.Module):
    """Focal IoU损失函数"""
    
    def __init__(self, alpha=1.0, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred_logits, true_labels):
        """
        Args:
            pred_logits: (batch, n_channels) 预测logits
            true_labels: (batch, n_channels) 真实标签
        Returns:
            focal_iou_loss: Focal IoU损失值
        """
        # 将logits转换为概率
        pred_probs = torch.sigmoid(pred_logits)
        
        # 计算交集和并集
        intersection = (pred_probs * true_labels).sum(dim=1)  # (batch,)
        union = pred_probs.sum(dim=1) + true_labels.sum(dim=1) - intersection  # (batch,)
        
        # 计算IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Focal权重
        focal_weight = self.alpha * (1 - iou) ** self.gamma
        
        # Focal IoU损失
        focal_iou_loss = focal_weight * (1 - iou)
        focal_iou_loss = focal_iou_loss.mean()
        
        return focal_iou_loss


class WeightedIoULoss(nn.Module):
    """加权IoU损失函数"""
    
    def __init__(self, pos_weight=None, smooth=1e-6):
        super().__init__()
        self.pos_weight = pos_weight
        self.smooth = smooth
    
    def forward(self, pred_logits, true_labels):
        """
        Args:
            pred_logits: (batch, n_channels) 预测logits
            true_labels: (batch, n_channels) 真实标签
        Returns:
            weighted_iou_loss: 加权IoU损失值
        """
        # 将logits转换为概率
        pred_probs = torch.sigmoid(pred_logits)
        
        # 计算交集和并集
        intersection = (pred_probs * true_labels).sum(dim=1)  # (batch,)
        union = pred_probs.sum(dim=1) + true_labels.sum(dim=1) - intersection  # (batch,)
        
        # 计算IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # 基础IoU损失
        base_iou_loss = 1 - iou
        
        # 如果有权重，应用权重
        if self.pos_weight is not None:
            # 计算每个样本的权重
            sample_weights = (true_labels * self.pos_weight.unsqueeze(0)).sum(dim=1)
            weighted_iou_loss = (sample_weights * base_iou_loss).mean()
        else:
            weighted_iou_loss = base_iou_loss.mean()
        
        return weighted_iou_loss


class CombinedLoss(nn.Module):
    """组合损失函数：BCE + IoU"""
    
    def __init__(self, bce_weight=1.0, iou_weight=2.0, iou_type='basic', smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.smooth = smooth
        
        # 选择IoU损失类型
        if iou_type == 'basic':
            self.iou_loss_fn = IoULoss(smooth=smooth)
        elif iou_type == 'focal':
            self.iou_loss_fn = FocalIoULoss(smooth=smooth)
        elif iou_type == 'weighted':
            self.iou_loss_fn = WeightedIoULoss(smooth=smooth)
        else:
            raise ValueError(f"Unknown IoU type: {iou_type}")
    
    def forward(self, pred_logits, true_labels, pos_weight=None):
        """
        Args:
            pred_logits: (batch, n_channels) 预测logits
            true_labels: (batch, n_channels) 真实标签
            pos_weight: (n_channels,) 正样本权重
        Returns:
            combined_loss: 组合损失值
            loss_dict: 损失详情字典
        """
        # BCE损失
        if pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_logits, true_labels, pos_weight=pos_weight
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(pred_logits, true_labels)
        
        # IoU损失
        if hasattr(self.iou_loss_fn, 'pos_weight') and pos_weight is not None:
            self.iou_loss_fn.pos_weight = pos_weight
        
        iou_loss = self.iou_loss_fn(pred_logits, true_labels)
        
        # 组合损失
        combined_loss = self.bce_weight * bce_loss + self.iou_weight * iou_loss
        
        # 损失详情
        loss_dict = {
            'bce_loss': bce_loss.item(),
            'iou_loss': iou_loss.item(),
            'combined_loss': combined_loss.item()
        }
        
        return combined_loss, loss_dict


def test_iou_loss():
    """测试IoU损失函数"""
    print("测试IoU损失函数...")
    
    # 创建测试数据
    batch_size = 4
    n_channels = 5
    
    pred_logits = torch.randn(batch_size, n_channels)
    true_labels = torch.randint(0, 2, (batch_size, n_channels)).float()
    
    print(f"预测logits形状: {pred_logits.shape}")
    print(f"真实标签形状: {true_labels.shape}")
    
    # 测试不同的IoU损失
    losses = {
        'IoU': IoULoss(),
        'FocalIoU': FocalIoULoss(),
        'WeightedIoU': WeightedIoULoss(),
        'Combined': CombinedLoss(bce_weight=1.0, iou_weight=2.0, iou_type='basic')
    }
    
    for name, loss_fn in losses.items():
        if name == 'Combined':
            loss, loss_dict = loss_fn(pred_logits, true_labels)
            print(f"{name}损失: {loss:.4f}")
            print(f"  详情: {loss_dict}")
        else:
            loss = loss_fn(pred_logits, true_labels)
            print(f"{name}损失: {loss:.4f}")
    
    print("IoU损失函数测试完成！")


if __name__ == "__main__":
    test_iou_loss()
