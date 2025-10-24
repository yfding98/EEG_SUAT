#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_channel_aware_kfold_conservative.py

保守版K折交叉验证训练脚本
解决过拟合和学习率过高问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import sys
import gc
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

# 添加父目录到路径以导入原始模型
sys.path.append(str(Path(__file__).parent.parent / 'raw_data_training'))

from model_channel_aware_multilabel import ChannelAwareMultilabelNet, create_channel_aware_multilabel_model
from utils import AverageMeter, save_checkpoint, EarlyStopping
from dataset_selected import create_dataloaders
from iou_loss import CombinedLoss, IoULoss, FocalIoULoss, WeightedIoULoss


def compute_conservative_class_weights(labels):
    """计算保守的类别权重"""
    n_samples, n_classes = labels.shape
    
    # 计算每个通道的正负样本数量
    pos_counts = labels.sum(axis=0)
    neg_counts = n_samples - pos_counts
    
    # 计算保守权重
    class_weights = torch.ones(n_classes)  # 从1开始，而不是0
    
    for i in range(n_classes):
        if pos_counts[i] > 0 and neg_counts[i] > 0:
            # 使用更保守的权重计算
            ratio = float(pos_counts[i]) / float(neg_counts[i])
            if ratio < 0.1:  # 极稀有通道
                class_weights[i] = 3.0
            elif ratio < 0.3:  # 稀有通道
                class_weights[i] = 2.0
            elif ratio < 0.7:  # 平衡通道
                class_weights[i] = 1.0
            else:  # 高频通道
                class_weights[i] = 0.8
        elif pos_counts[i] == 0:
            # 从未出现的通道，给予极小权重
            class_weights[i] = 0.01
        else:
            # 只有正样本的通道，给予正常权重
            class_weights[i] = 1.0
    
    return class_weights


def create_balanced_sampler(dataset, labels):
    """创建平衡采样器"""
    # 计算每个样本的权重（基于其标签的稀有程度）
    sample_weights = []
    for i in range(len(labels)):
        # 计算该样本中稀有通道的数量
        rare_channels = (labels[i] == 1).sum().item()
        # 稀有通道越多，权重越高，但不要过高
        weight = 1.0 + rare_channels * 0.3  # 降低权重倍数
        sample_weights.append(weight)
    
    # 创建加权随机采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler


def custom_collate_fn(batch):
    """自定义collate函数处理不同大小的批次"""
    # 分离不同字段
    bands_list = [item['bands'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    files_list = [item['file'] for item in batch]
    
    # 处理bands数据
    if isinstance(bands_list[0], list):
        # 如果bands是列表，需要堆叠
        n_bands = len(bands_list[0])
        processed_bands = []
        
        for band_idx in range(n_bands):
            band_tensors = []
            for item_bands in bands_list:
                if isinstance(item_bands[band_idx], list):
                    band_tensors.append(torch.stack(item_bands[band_idx], dim=0))
                else:
                    band_tensors.append(item_bands[band_idx])
            
            # 堆叠当前频段的所有样本
            processed_bands.append(torch.stack(band_tensors, dim=0))
        
        bands = processed_bands
    else:
        # 如果bands已经是张量，直接堆叠
        bands = torch.stack(bands_list, dim=0)
    
    # 堆叠标签
    labels = torch.stack(labels_list, dim=0)
    
    return {
        'bands': bands,
        'labels': labels,
        'file': files_list
    }


def analyze_channel_distribution_conservative(data_loader):
    """保守版通道分布分析"""
    all_labels = []
    
    print("分析通道标签分布...")
    for batch in data_loader:
        labels = batch['labels']  # (batch, n_channels)
        all_labels.append(labels)
    
    all_labels = torch.cat(all_labels, dim=0)  # (total_samples, n_channels)
    n_channels = all_labels.shape[1]
    
    # 计算每个通道的正样本比例
    positive_ratios = all_labels.mean(dim=0)  # (n_channels,)
    
    print(f"通道标签分布:")
    for i, ratio in enumerate(positive_ratios):
        print(f"  通道{i}: {ratio:.3f} ({ratio*100:.1f}% 正样本)")
    
    # 使用保守权重计算
    class_weights = compute_conservative_class_weights(all_labels)
    
    # 统计不同频率通道
    never_positive = (positive_ratios == 0.0)
    rare_positive = (positive_ratios > 0.0) & (positive_ratios < 0.1)
    medium_positive = (positive_ratios >= 0.1) & (positive_ratios < 0.3)
    frequent_positive = (positive_ratios >= 0.3)
    
    print(f"保守类别权重:")
    print(f"  从未出现通道: {never_positive.sum().item()}个, 平均权重: {class_weights[never_positive].mean():.3f}")
    print(f"  稀有通道: {rare_positive.sum().item()}个, 平均权重: {class_weights[rare_positive].mean():.3f}")
    print(f"  中频通道: {medium_positive.sum().item()}个, 平均权重: {class_weights[medium_positive].mean():.3f}")
    print(f"  高频通道: {frequent_positive.sum().item()}个, 平均权重: {class_weights[frequent_positive].mean():.3f}")
    
    return class_weights, positive_ratios


def compute_multilabel_metrics(pred_logits, true_labels, threshold=0.5):
    """计算多标签分类指标"""
    pred_probs = torch.sigmoid(pred_logits)
    pred_binary = (pred_probs > threshold).float()
    
    batch_size, n_channels = pred_logits.shape
    
    # 每个通道的指标
    channel_precisions = []
    channel_recalls = []
    channel_f1s = []
    
    for ch in range(n_channels):
        pred_ch = pred_binary[:, ch]
        true_ch = true_labels[:, ch]
        
        # 计算TP, FP, FN
        tp = (pred_ch * true_ch).sum().item()
        fp = (pred_ch * (1 - true_ch)).sum().item()
        fn = ((1 - pred_ch) * true_ch).sum().item()
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        channel_precisions.append(precision)
        channel_recalls.append(recall)
        channel_f1s.append(f1)
    
    # 宏平均
    macro_precision = np.mean(channel_precisions) * 100
    macro_recall = np.mean(channel_recalls) * 100
    macro_f1 = np.mean(channel_f1s) * 100
    
    # mAP (Mean Average Precision)
    map_scores = []
    for ch in range(n_channels):
        pred_ch = pred_probs[:, ch]
        true_ch = true_labels[:, ch]
        
        # 按预测概率排序
        sorted_indices = torch.argsort(pred_ch, descending=True)
        sorted_true = true_ch[sorted_indices]
        
        # 计算AP
        tp_cumsum = torch.cumsum(sorted_true, dim=0)
        precision_at_k = tp_cumsum / torch.arange(1, len(sorted_true) + 1, device=pred_ch.device).float()
        
        # 只考虑正样本的precision
        ap = precision_at_k[sorted_true == 1].mean().item() if sorted_true.sum() > 0 else 0
        map_scores.append(ap)
    
    mAP = np.mean(map_scores) * 100
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'mAP': mAP
    }


class ConservativeChannelAwareKFoldTrainer:
    """保守版通道感知模型K折交叉验证训练器"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_dir,
        n_channels,
        class_weights,
        fold_idx,
        early_stopping_patience=5,  # 更早的早停
        gradient_accumulation_steps=1,
        use_iou_loss=False,  # 默认不使用IoU损失
        iou_weight=0.5,  # 更低的IoU权重
        iou_type='basic'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fold_idx = fold_idx
        
        # 多标签分类信息
        self.n_channels = n_channels
        self.class_weights = class_weights
        
        # IoU损失配置
        self.use_iou_loss = use_iou_loss
        self.iou_weight = iou_weight
        self.iou_type = iou_type
        
        # 创建IoU损失函数
        if self.use_iou_loss:
            self.combined_criterion = CombinedLoss(
                bce_weight=1.0,
                iou_weight=self.iou_weight,
                iou_type=self.iou_type
            )
            print(f"  使用组合损失函数 (IoU权重: {self.iou_weight}, 类型: {self.iou_type})")
        else:
            print(f"  使用标准BCE损失函数")
        
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        self.best_val_f1 = 0.0
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        losses = AverageMeter()
        metrics_meter = {
            'macro_f1': AverageMeter(),
            'macro_precision': AverageMeter(),
            'macro_recall': AverageMeter(),
            'mAP': AverageMeter()
        }
        
        pbar = tqdm(self.train_loader, desc=f"Fold {self.fold_idx} Epoch {epoch} [Train]")
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 获取多频段数据
            bands = batch['bands']
            labels = batch['labels'].to(self.device)
            
            # 将多频段数据堆叠为 (batch, n_bands, n_channels, n_samples)
            try:
                if isinstance(bands, list):
                    # 如果bands是列表，需要先转换为张量
                    # 确保每个频段都是张量
                    processed_bands = []
                    for band in bands:
                        if isinstance(band, list):
                            processed_bands.append(torch.stack(band, dim=0))
                        else:
                            processed_bands.append(band)
                    bands_tensor = torch.stack(processed_bands, dim=1).to(self.device)
                    # 转置以得到正确的形状 (batch, n_bands, n_channels, n_samples)
                    bands_tensor = bands_tensor.transpose(0, 1)
                else:
                    # 如果bands已经是张量，直接使用
                    bands_tensor = bands.to(self.device)
            except Exception as e:
                print(f"处理bands数据时出错: {e}")
                print(f"bands类型: {type(bands)}")
                if isinstance(bands, list):
                    print(f"bands长度: {len(bands)}")
                    for i, band in enumerate(bands):
                        print(f"频段{i}类型: {type(band)}")
                        if isinstance(band, list):
                            print(f"频段{i}长度: {len(band)}")
                        else:
                            print(f"频段{i}形状: {band.shape}")
                raise e
            
            # Forward
            logits = self.model(bands_tensor, labels)
            
            # 计算损失
            if self.use_iou_loss:
                loss, loss_dict = self.combined_criterion(logits, labels, pos_weight=self.class_weights)
            else:
                loss = self.criterion(logits, labels)
                loss_dict = {'bce_loss': loss.item(), 'iou_loss': 0.0, 'combined_loss': loss.item()}
            
            # 梯度累积
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # 更严格的梯度裁剪
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 指标
            with torch.no_grad():
                metrics = compute_multilabel_metrics(logits.detach(), labels.detach())
            
            # 更新
            losses.update(loss.item() * self.gradient_accumulation_steps, labels.size(0))
            for key in metrics_meter:
                if key in metrics:
                    metrics_meter[key].update(metrics[key], labels.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'F1': f'{metrics_meter["macro_f1"].avg:.1f}%',
                'mAP': f'{metrics_meter["mAP"].avg:.1f}%',
                'mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
            })
            
            # 显式释放
            del bands, labels, bands_tensor, logits, loss
            
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            'loss': losses.avg,
            'macro_precision': metrics_meter['macro_precision'].avg,
            'macro_recall': metrics_meter['macro_recall'].avg,
            'macro_f1': metrics_meter['macro_f1'].avg,
            'mAP': metrics_meter['mAP'].avg
        }
    
    @torch.no_grad()
    def validate(self, epoch, phase='Val'):
        """验证"""
        self.model.eval()
        
        losses = AverageMeter()
        metrics_meter = {
            'macro_f1': AverageMeter(),
            'macro_precision': AverageMeter(),
            'macro_recall': AverageMeter(),
            'mAP': AverageMeter()
        }
        
        pbar = tqdm(self.val_loader, desc=f"Fold {self.fold_idx} Epoch {epoch} [{phase}]")
        for batch in pbar:
            bands = batch['bands']
            labels = batch['labels'].to(self.device)
            
            # 将多频段数据堆叠为 (batch, n_bands, n_channels, n_samples)
            try:
                if isinstance(bands, list):
                    # 如果bands是列表，需要先转换为张量
                    # 确保每个频段都是张量
                    processed_bands = []
                    for band in bands:
                        if isinstance(band, list):
                            processed_bands.append(torch.stack(band, dim=0))
                        else:
                            processed_bands.append(band)
                    bands_tensor = torch.stack(processed_bands, dim=1).to(self.device)
                    # 转置以得到正确的形状 (batch, n_bands, n_channels, n_samples)
                    bands_tensor = bands_tensor.transpose(0, 1)
                else:
                    # 如果bands已经是张量，直接使用
                    bands_tensor = bands.to(self.device)
            except Exception as e:
                print(f"验证时处理bands数据出错: {e}")
                print(f"bands类型: {type(bands)}")
                if isinstance(bands, list):
                    print(f"bands长度: {len(bands)}")
                    for i, band in enumerate(bands):
                        print(f"频段{i}类型: {type(band)}")
                        if isinstance(band, list):
                            print(f"频段{i}长度: {len(band)}")
                        else:
                            print(f"频段{i}形状: {band.shape}")
                raise e
            
            # Forward
            logits = self.model(bands_tensor, labels)
            
            # 计算损失
            if self.use_iou_loss:
                loss, loss_dict = self.combined_criterion(logits, labels, pos_weight=self.class_weights)
            else:
                loss = self.criterion(logits, labels)
            
            # 指标
            metrics = compute_multilabel_metrics(logits, labels)
            
            # 更新
            losses.update(loss.item(), labels.size(0))
            for key in metrics_meter:
                if key in metrics:
                    metrics_meter[key].update(metrics[key], labels.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'F1': f'{metrics_meter["macro_f1"].avg:.1f}%',
                'mAP': f'{metrics_meter["mAP"].avg:.1f}%'
            })
            
            del bands, labels, bands_tensor, logits, loss
        
        torch.cuda.empty_cache()
        
        return {
            'loss': losses.avg,
            'macro_precision': metrics_meter['macro_precision'].avg,
            'macro_recall': metrics_meter['macro_recall'].avg,
            'macro_f1': metrics_meter['macro_f1'].avg,
            'mAP': metrics_meter['mAP'].avg
        }
    
    def train_fold(self, n_epochs):
        """训练一个fold"""
        print(f"\n{'='*80}")
        print(f"开始训练 Fold {self.fold_idx}")
        print(f"{'='*80}")
        print(f"设备: {self.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"训练集: {len(self.train_loader.dataset)}")
        print(f"验证集: {len(self.val_loader.dataset)}")
        print(f"梯度累积步数: {self.gradient_accumulation_steps}")
        
        if torch.cuda.is_available():
            print(f"初始显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        fold_results = []
        
        for epoch in range(1, n_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch, 'Val')
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印
            print(f"\nFold {self.fold_idx} Epoch {epoch}/{n_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"F1: {train_metrics['macro_f1']:.2f}%, "
                  f"mAP: {train_metrics['mAP']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"F1: {val_metrics['macro_f1']:.2f}%, "
                  f"mAP: {val_metrics['mAP']:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            if torch.cuda.is_available():
                print(f"  显存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB (峰值)")
                torch.cuda.reset_peak_memory_stats()
            
            # 保存最佳模型
            is_best = val_metrics['macro_f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['macro_f1']
                print(f"  -> 新的最佳F1: {val_metrics['macro_f1']:.2f}%")
            
            # 保存检查点
            fold_save_dir = self.save_dir / f'fold_{self.fold_idx}'
            fold_save_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            save_checkpoint(
                {
                    'epoch': epoch,
                    'fold': self.fold_idx,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_f1': self.best_val_f1,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                },
                is_best,
                fold_save_dir
            )
            
            # 记录结果
            fold_results.append({
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'lr': current_lr
            })
            
            # Early stopping
            self.early_stopping(val_metrics['macro_f1'])
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        return {
            'best_val_f1': self.best_val_f1,
            'fold_results': fold_results
        }


def kfold_cross_validation_conservative(
    dataset,
    n_channels,
    n_samples,
    n_bands,
    config,
    device,
    save_dir,
    n_folds=5,
    n_epochs=20,  # 减少训练轮数
    batch_size=8,  # 减少批次大小
    lr=0.0001,  # 降低学习率
    weight_decay=0.01,
    early_stopping_patience=5,  # 更早的早停
    gradient_accumulation_steps=2,  # 增加梯度累积
    use_iou_loss=False,  # 默认不使用IoU损失
    iou_weight=0.5,  # 更低的IoU权重
    iou_type='basic'
):
    """保守版K折交叉验证"""
    
    print(f"\n{'='*80}")
    print(f"开始 {n_folds} 折交叉验证 (保守版)")
    print(f"{'='*80}")
    print(f"总数据量: {len(dataset)}")
    print(f"每折数据量: {len(dataset) // n_folds}")
    
    # 创建K折分割器
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 获取所有样本的索引
    dataset_indices = list(range(len(dataset)))
    
    fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset_indices)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")
        print(f"训练集: {len(train_indices)} 样本")
        print(f"验证集: {len(val_indices)} 样本")
        
        # 创建数据加载器
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        # 动态调整批次大小
        train_batch_size = min(batch_size, max(1, len(train_subset) // 8))
        val_batch_size = min(batch_size, max(1, len(val_subset)))
        
        print(f"  训练批次大小: {train_batch_size}")
        print(f"  验证批次大小: {val_batch_size}")
        
        # 创建平衡采样器
        train_labels = torch.stack([dataset[i]['labels'] for i in train_indices])
        train_sampler = create_balanced_sampler(train_subset, train_labels)
        
        train_loader = DataLoader(
            train_subset, batch_size=train_batch_size, sampler=train_sampler, num_workers=0,
            drop_last=True, collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            val_subset, batch_size=val_batch_size, shuffle=False, num_workers=0,
            drop_last=False, collate_fn=custom_collate_fn
        )
        
        # 分析通道分布
        class_weights, positive_ratios = analyze_channel_distribution_conservative(train_loader)
        
        # 创建模型
        model = create_channel_aware_multilabel_model(
            n_channels=n_channels,
            n_samples=n_samples,
            n_bands=n_bands,
            d_model=config.get('d_model', 64),  # 更小的模型
            n_heads=config.get('n_heads', 2),   # 更少的注意力头
            n_layers=config.get('n_layers', 1), # 更少的层数
            dropout=config.get('dropout', 0.5)  # 更高的dropout
        )
        model = model.to(device)
        
        # 损失函数
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
        
        # 优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-7
        )
        
        # 训练器
        trainer = ConservativeChannelAwareKFoldTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            save_dir=save_dir,
            n_channels=n_channels,
            class_weights=class_weights.to(device),
            fold_idx=fold_idx + 1,
            early_stopping_patience=early_stopping_patience,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_iou_loss=use_iou_loss,
            iou_weight=iou_weight,
            iou_type=iou_type
        )
        
        # 训练
        fold_result = trainer.train_fold(n_epochs)
        fold_results.append(fold_result)
        
        # 清理内存
        del model, criterion, optimizer, scheduler, trainer
        torch.cuda.empty_cache()
        gc.collect()
    
    return fold_results


def main():
    parser = argparse.ArgumentParser(description='保守版通道感知EEG分类模型K折交叉验证训练')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--window_size', type=float, default=2.0)
    parser.add_argument('--window_stride', type=float, default=1.0)
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=64)  # 更小的模型
    parser.add_argument('--n_heads', type=int, default=2)   # 更少的注意力头
    parser.add_argument('--n_layers', type=int, default=1)   # 更少的层数
    parser.add_argument('--dropout', type=float, default=0.5)  # 更高的dropout
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8)  # 更小的批次
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=20)  # 更少的训练轮数
    parser.add_argument('--lr', type=float, default=0.0001)  # 更低的学习率
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    
    # K折参数
    parser.add_argument('--n_folds', type=int, default=5,
                        help='K折交叉验证的折数')
    
    # IoU损失参数
    parser.add_argument('--use_iou_loss', action='store_true', default=False,
                        help='使用IoU损失函数')
    parser.add_argument('--iou_weight', type=float, default=0.5,
                        help='IoU损失权重')
    parser.add_argument('--iou_type', type=str, default='basic',
                        choices=['basic', 'focal', 'weighted'],
                        help='IoU损失类型')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints_channel_aware_kfold_conservative')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f"channel_aware_kfold_conservative_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config = {
        'window_size': args.window_size,
        'window_stride': args.window_stride,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'n_folds': args.n_folds,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay
    }
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # 加载数据
    print("\n准备数据...")
    print(f"  数据路径: {args.data_root}")
    print(f"  窗口大小: {args.window_size}秒")
    print(f"  窗口步长: {args.window_stride}秒")
    
    try:
        # 创建完整数据集（不分割）
        train_loader, _, _, channel_names = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            window_size=args.window_size,
            window_stride=args.window_stride,
            val_split=0.0,  # 不分割，使用K折
            test_split=0.0,
            num_workers=0,
            seed=args.seed
        )
        
        # 获取数据集
        dataset = train_loader.dataset
        sample_batch = next(iter(train_loader))
        n_channels = sample_batch['bands'][0].shape[1]
        n_samples = sample_batch['bands'][0].shape[2]
        n_bands = len(sample_batch['bands'])
        
        print(f"\n数据信息:")
        print(f"  通道数: {n_channels}")
        print(f"  时间点数: {n_samples}")
        print(f"  频段数: {n_bands}")
        print(f"  总样本数: {len(dataset)}")
        
    except Exception as e:
        print(f"\n错误：加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # K折交叉验证
    fold_results = kfold_cross_validation_conservative(
        dataset=dataset,
        n_channels=n_channels,
        n_samples=n_samples,
        n_bands=n_bands,
        config=config,
        device=device,
        save_dir=save_dir,
        n_folds=args.n_folds,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_iou_loss=args.use_iou_loss,
        iou_weight=args.iou_weight,
        iou_type=args.iou_type
    )
    
    # 计算平均结果
    best_f1_scores = [result['best_val_f1'] for result in fold_results]
    mean_f1 = np.mean(best_f1_scores)
    std_f1 = np.std(best_f1_scores)
    
    print(f"\n{'='*80}")
    print("K折交叉验证结果 (保守版)")
    print(f"{'='*80}")
    print(f"各折最佳F1分数:")
    for i, f1 in enumerate(best_f1_scores):
        print(f"  Fold {i+1}: {f1:.2f}%")
    
    print(f"\n平均结果:")
    print(f"  平均F1: {mean_f1:.2f}% ± {std_f1:.2f}%")
    print(f"  最佳F1: {max(best_f1_scores):.2f}%")
    print(f"  最差F1: {min(best_f1_scores):.2f}%")
    
    # 保存结果
    results_summary = {
        'fold_results': fold_results,
        'best_f1_scores': best_f1_scores,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'max_f1': max(best_f1_scores),
        'min_f1': min(best_f1_scores)
    }
    
    with open(save_dir / 'kfold_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n结果已保存到: {save_dir}")
    print(f"训练完成!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--data_root', r'E:\DataSet\EEG\EEG dataset_SUAT_processed_selected',
            '--window_size', '2',
            '--window_stride', '1',
            '--batch_size', '8',
            '--d_model', '64',
            '--n_heads', '2',
            '--n_layers', '1',
            '--dropout', '0.5',
            '--n_folds', '5',
            '--n_epochs', '20',
            '--lr', '0.0001',
            '--weight_decay', '0.01',
            '--early_stopping_patience', '5',
            '--gradient_accumulation_steps', '2',
        ])
    main()
