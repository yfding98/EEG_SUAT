#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_channel_aware_kfold_optimized_fixed.py

整合修复与改进：
 - GroupKFold 按 file 分组，避免窗口泄漏
 - 稳健的 class weight / sampler 实现
 - 可切换 IoU（默认关闭，方便先跑 BCE 基线）
 - per-channel threshold search 工具
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
from sklearn.model_selection import GroupKFold

# 添加父目录到路径以导入原始模型（按你的项目目录）
sys.path.append(str(Path(__file__).parent.parent / 'raw_data_training'))

from model_channel_aware_multilabel import ChannelAwareMultilabelNet, create_channel_aware_multilabel_model
from utils import AverageMeter, save_checkpoint, EarlyStopping
from dataset_selected import create_dataloaders
from iou_loss import CombinedLoss, IoULoss, FocalIoULoss, WeightedIoULoss

# -------------------------
# 辅助函数：class weights
# -------------------------
def compute_balanced_class_weights(labels):
    """
    labels: numpy array or torch tensor of shape (N, C)
    返回: torch.FloatTensor of shape (C,) 用作 BCEWithLogitsLoss pos_weight
    对于从未出现的通道，返回一个很小的正权重（例如 0.01）
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.array(labels)

    n_samples, n_classes = labels_np.shape
    pos_counts = labels_np.sum(axis=0).astype(np.float32)
    neg_counts = n_samples - pos_counts

    pos_weights = np.zeros(n_classes, dtype=np.float32)
    for i in range(n_classes):
        if pos_counts[i] > 0:
            # pos_weight = neg / pos as BCEWithLogitsLoss expects
            pos_weights[i] = float(neg_counts[i] / pos_counts[i])
            # 限制最大权重防止梯度爆炸
            pos_weights[i] = min(pos_weights[i], 100.0)
        else:
            # 从未出现的通道，给个小正权重以避免数值问题
            pos_weights[i] = 0.01

    return torch.from_numpy(pos_weights).float()


# -------------------------
# Balanced sampler
# -------------------------
def create_balanced_sampler(dataset, labels):
    """
    labels: torch.Tensor shape (N, C) or numpy array
    返回 WeightedRandomSampler(weights=num_samples, replacement=True)
    样本权重 = sum(channel_weight for positive channel in sample) + epsilon
    channel_weight = inverse frequency（归一化）
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.array(labels)

    n_samples = labels_np.shape[0]
    pos_counts = labels_np.sum(axis=0).astype(np.float32)
    # 频率
    freq = (pos_counts / max(1.0, n_samples))
    # 通道权重 = 1 / freq
    channel_weights = np.zeros_like(freq)
    for i in range(len(freq)):
        channel_weights[i] = 1.0 / (freq[i] + 1e-6)
    # 归一化以避免极端权重
    channel_weights = channel_weights / (channel_weights.mean() + 1e-9)

    # 每个样本权重 = 包含的通道权重之和 + eps
    sample_weights = (labels_np * channel_weights).sum(axis=1) + 1e-3
    # 标准化
    sample_weights = sample_weights / (sample_weights.mean() + 1e-9)

    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# -------------------------
# 自定义 collate
# -------------------------
def custom_collate_fn(batch):
    """自定义collate函数处理不同大小的批次"""
    bands_list = [item['bands'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    files_list = [item.get('file', None) for item in batch]

    # 处理bands数据
    if isinstance(bands_list[0], list):
        n_bands = len(bands_list[0])
        processed_bands = []
        for band_idx in range(n_bands):
            band_tensors = []
            for item_bands in bands_list:
                if isinstance(item_bands[band_idx], list):
                    band_tensors.append(torch.stack(item_bands[band_idx], dim=0))
                else:
                    band_tensors.append(item_bands[band_idx])
            processed_bands.append(torch.stack(band_tensors, dim=0))
        bands = processed_bands
    else:
        bands = torch.stack(bands_list, dim=0)

    labels = torch.stack(labels_list, dim=0)

    return {
        'bands': bands,
        'labels': labels,
        'file': files_list
    }


# -------------------------
# 分析通道分布（稳健版）
# -------------------------
def analyze_channel_distribution_optimized(data_loader):
    """优化版通道分布分析"""
    all_labels = []

    print("分析通道标签分布...")
    for batch in data_loader:
        labels = batch['labels']  # (batch, n_channels)
        all_labels.append(labels)

    if len(all_labels) == 0:
        raise RuntimeError("train loader 中没有标签样本!")

    all_labels = torch.cat(all_labels, dim=0).cpu()  # (total_samples, n_channels)
    n_channels = all_labels.shape[1]

    # 计算每个通道的正样本比例
    positive_ratios = all_labels.float().mean(dim=0)  # (n_channels,)

    print(f"通道标签分布:")
    for i, ratio in enumerate(positive_ratios):
        print(f"  通道{i}: {ratio:.3f} ({ratio*100:.1f}% 正样本)")

    # 使用平衡权重计算
    class_weights = compute_balanced_class_weights(all_labels)

    # 统计不同频率通道
    never_positive = (positive_ratios == 0.0)
    rare_positive = (positive_ratios > 0.0) & (positive_ratios < 0.1)
    medium_positive = (positive_ratios >= 0.1) & (positive_ratios < 0.3)
    frequent_positive = (positive_ratios >= 0.3)

    def _safe_mean(t):
        try:
            return float(t.mean())
        except Exception:
            return 0.0

    print(f"优化类别权重:")
    print(f"  从未出现通道: {never_positive.sum().item()}个, 平均权重: {_safe_mean(class_weights[never_positive]) :.3f}")
    print(f"  稀有通道: {rare_positive.sum().item()}个, 平均权重: {_safe_mean(class_weights[rare_positive]) :.3f}")
    print(f"  中频通道: {medium_positive.sum().item()}个, 平均权重: {_safe_mean(class_weights[medium_positive]) :.3f}")
    print(f"  高频通道: {frequent_positive.sum().item()}个, 平均权重: {_safe_mean(class_weights[frequent_positive]) :.3f}")

    return class_weights, positive_ratios


# -------------------------
# 指标计算（多标签）
# -------------------------
def compute_multilabel_metrics(pred_logits, true_labels, threshold=0.5):
    """计算多标签分类指标"""
    pred_probs = torch.sigmoid(pred_logits)
    pred_binary = (pred_probs > threshold).float()

    batch_size, n_channels = pred_logits.shape

    channel_precisions = []
    channel_recalls = []
    channel_f1s = []

    for ch in range(n_channels):
        pred_ch = pred_binary[:, ch]
        true_ch = true_labels[:, ch]

        tp = (pred_ch * true_ch).sum().item()
        fp = (pred_ch * (1 - true_ch)).sum().item()
        fn = ((1 - pred_ch) * true_ch).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        channel_precisions.append(precision)
        channel_recalls.append(recall)
        channel_f1s.append(f1)

    macro_precision = np.mean(channel_precisions) * 100
    macro_recall = np.mean(channel_recalls) * 100
    macro_f1 = np.mean(channel_f1s) * 100

    # 简易 mAP 计算（每通道）
    map_scores = []
    for ch in range(n_channels):
        pred_ch = pred_probs[:, ch]
        true_ch = true_labels[:, ch]

        sorted_indices = torch.argsort(pred_ch, descending=True)
        sorted_true = true_ch[sorted_indices]

        tp_cumsum = torch.cumsum(sorted_true, dim=0)
        precision_at_k = tp_cumsum / torch.arange(1, len(sorted_true) + 1, device=pred_ch.device).float()

        ap = precision_at_k[sorted_true == 1].mean().item() if sorted_true.sum() > 0 else 0.0
        map_scores.append(ap)

    mAP = np.mean(map_scores) * 100

    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'mAP': mAP
    }


# -------------------------
# Trainer 类
# -------------------------
class OptimizedChannelAwareKFoldTrainer:
    """优化版通道感知模型K折交叉验证训练器"""

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
        early_stopping_patience=10,
        gradient_accumulation_steps=1,
        use_iou_loss=False,
        iou_weight=1.5,
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

        self.n_channels = n_channels
        self.class_weights = class_weights

        self.use_iou_loss = use_iou_loss
        self.iou_weight = iou_weight
        self.iou_type = iou_type

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

    def _prepare_bands_tensor(self, bands):
        """
        统一将 collate 后的 bands 转为形状 (batch, n_bands, n_channels, n_samples) 的张量
        支持 bands 为 list 或 tensor 的情形
        """
        if isinstance(bands, list):
            processed_bands = []
            for band in bands:
                if isinstance(band, list):
                    # band 里每项已经是张量 (batch, channels, samples) -> stack -> (batch, channels, samples)
                    processed_bands.append(torch.stack(band, dim=0))
                else:
                    processed_bands.append(band)
            bands_tensor = torch.stack(processed_bands, dim=1).to(self.device)
            # bands_tensor 现在形状 (batch, n_bands, channels, samples) 或 (n_bands, batch, channels, samples)
            # 先保证 (batch, n_bands, channels, samples)
            if bands_tensor.shape[0] != processed_bands[0].shape[0]:
                bands_tensor = bands_tensor.transpose(0, 1)
        else:
            bands_tensor = bands.to(self.device)
        return bands_tensor

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
            bands = batch['bands']
            labels = batch['labels'].to(self.device)

            try:
                bands_tensor = self._prepare_bands_tensor(bands)
            except Exception as e:
                print(f"处理bands数据时出错: {e}")
                raise e

            logits = self.model(bands_tensor, labels)

            if self.use_iou_loss:
                loss, loss_dict = self.combined_criterion(logits, labels, pos_weight=self.class_weights)
            else:
                loss = self.criterion(logits, labels)
                loss_dict = {'bce_loss': loss.item(), 'iou_loss': 0.0, 'combined_loss': loss.item()}

            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            with torch.no_grad():
                metrics = compute_multilabel_metrics(logits.detach(), labels.detach())

            losses.update(loss.item() * self.gradient_accumulation_steps, labels.size(0))
            for key in metrics_meter:
                if key in metrics:
                    metrics_meter[key].update(metrics[key], labels.size(0))

            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'F1': f'{metrics_meter["macro_f1"].avg:.1f}%',
                'mAP': f'{metrics_meter["mAP"].avg:.1f}%',
                'mem': f'{torch.cuda.memory_allocated()/1024**3:.2f}GB'
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

            try:
                bands_tensor = self._prepare_bands_tensor(bands)
            except Exception as e:
                print(f"验证时处理bands数据出错: {e}")
                raise e

            logits = self.model(bands_tensor, labels)

            if self.use_iou_loss:
                loss, loss_dict = self.combined_criterion(logits, labels, pos_weight=self.class_weights)
            else:
                loss = self.criterion(logits, labels)

            metrics = compute_multilabel_metrics(logits, labels)

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
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch, 'Val')

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

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

            is_best = val_metrics['macro_f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['macro_f1']
                print(f"  -> 新的最佳F1: {val_metrics['macro_f1']:.2f}%")

            fold_save_dir = self.save_dir / f'fold_{self.fold_idx}'
            fold_save_dir.mkdir(parents=True, exist_ok=True)
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

            fold_results.append({
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'lr': current_lr
            })

            self.early_stopping(val_metrics['macro_f1'])
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        return {
            'best_val_f1': self.best_val_f1,
            'fold_results': fold_results
        }


# -------------------------
# K-Fold 主流程（使用 GroupKFold）
# -------------------------
def kfold_cross_validation_optimized(
    dataset,
    n_channels,
    n_samples,
    n_bands,
    config,
    device,
    save_dir,
    n_folds=5,
    n_epochs=30,
    batch_size=16,
    lr=0.001,
    weight_decay=0.01,
    early_stopping_patience=8,
    gradient_accumulation_steps=1,
    use_iou_loss=False,
    iou_weight=1.5,
    iou_type='basic'
):
    """优化版K折交叉验证（按 file 分组）"""

    print(f"\n{'='*80}")
    print(f"开始 {n_folds} 折交叉验证 (优化版)")
    print(f"{'='*80}")
    print(f"总数据量: {len(dataset)}")
    print(f"每折数据量 (approx): {len(dataset) // n_folds}")

    # groups: 按 dataset 中的 'file' 字段分组，确保同一 file 的 windows 不被分到 train/val 两边
    groups = [dataset[i]['file'] for i in range(len(dataset))]

    all_indices = np.arange(len(dataset))
    kfold = GroupKFold(n_splits=n_folds)

    fold_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(all_indices, all_indices, groups)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")
        print(f"训练集: {len(train_indices)} 样本")
        print(f"验证集: {len(val_indices)} 样本")

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        # 使用固定 batch size（不再过度动态缩小）
        train_batch_size = min(batch_size, max(1, len(train_subset)))
        val_batch_size = min(batch_size, max(1, len(val_subset)))

        print(f"  训练批次大小: {train_batch_size}")
        print(f"  验证批次大小: {val_batch_size}")

        # 计算标签并创建 sampler
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

        # 分析通道分布并获得 class_weights
        class_weights, positive_ratios = analyze_channel_distribution_optimized(train_loader)

        # 创建模型
        model = create_channel_aware_multilabel_model(
            n_channels=n_channels,
            n_samples=n_samples,
            n_bands=n_bands,
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 4),
            n_layers=config.get('n_layers', 2),
            dropout=config.get('dropout', 0.3)
        )
        model = model.to(device)

        # 损失函数（pos_weight 接受 tensor）
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))

        # 优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-6
        )

        trainer = OptimizedChannelAwareKFoldTrainer(
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

        fold_result = trainer.train_fold(n_epochs)
        fold_results.append(fold_result)

        del model, criterion, optimizer, scheduler, trainer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    return fold_results


# -------------------------
# 找最佳阈值（验证集上 per-channel threshold search）
# -------------------------
@torch.no_grad()
def find_best_thresholds(model, val_loader, device, n_channels):
    model.eval()
    probs_list = []
    labels_list = []
    for batch in val_loader:
        bands = batch['bands']
        labels = batch['labels'].to(device)
        # 使用与 Trainer 相同的 bands -> bands_tensor 逻辑（复用）
        # 这里简单处理，假设 bands 的格式与训练/验证一致
        if isinstance(bands, list):
            processed_bands = []
            for band in bands:
                if isinstance(band, list):
                    processed_bands.append(torch.stack(band, dim=0))
                else:
                    processed_bands.append(band)
            bands_tensor = torch.stack(processed_bands, dim=1).to(device)
            if bands_tensor.shape[0] != processed_bands[0].shape[0]:
                bands_tensor = bands_tensor.transpose(0, 1)
        else:
            bands_tensor = bands.to(device)

        logits = model(bands_tensor, labels)
        probs = torch.sigmoid(logits).cpu()
        probs_list.append(probs)
        labels_list.append(labels.cpu())

    probs_all = torch.cat(probs_list, dim=0).numpy()
    labels_all = torch.cat(labels_list, dim=0).numpy()

    best_thresholds = np.zeros(n_channels, dtype=np.float32)
    for ch in range(n_channels):
        best_f1 = -1.0
        best_t = 0.5
        for t in np.linspace(0.01, 0.99, 99):
            pred = (probs_all[:, ch] > t).astype(int)
            tp = ((pred == 1) & (labels_all[:, ch] == 1)).sum()
            fp = ((pred == 1) & (labels_all[:, ch] == 0)).sum()
            fn = ((pred == 0) & (labels_all[:, ch] == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[ch] = best_t
    return best_thresholds


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='优化版通道感知EEG分类模型K折交叉验证训练（修复版）')

    # 数据参数
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--window_size', type=float, default=2.0)
    parser.add_argument('--window_stride', type=float, default=1.0)

    # 模型参数
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--early_stopping_patience', type=int, default=8)

    # K折参数
    parser.add_argument('--n_folds', type=int, default=5,
                        help='K折交叉验证的折数')

    # IoU损失参数：默认不启用（便于先用 BCE 基线），通过 --use_iou_loss 启用
    parser.add_argument('--use_iou_loss', action='store_true', default=False,
                        help='启用 IoU 组合损失（默认: False）')
    parser.add_argument('--iou_weight', type=float, default=1.5,
                        help='IoU损失权重')
    parser.add_argument('--iou_type', type=str, default='basic',
                        choices=['basic', 'focal', 'weighted'],
                        help='IoU损失类型')

    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints_channel_aware_kfold_optimized_fixed')

    args = parser.parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f"channel_aware_kfold_optimized_fixed_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

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

    # 加载数据（使用你现有的 create_dataloaders）
    print("\n准备数据...")
    print(f"  数据路径: {args.data_root}")
    print(f"  窗口大小: {args.window_size}秒")
    print(f"  窗口步长: {args.window_stride}秒")

    try:
        train_loader, _, _, channel_names = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            window_size=args.window_size,
            window_stride=args.window_stride,
            val_split=0.0,
            test_split=0.0,
            num_workers=0,
            seed=args.seed
        )

        dataset = train_loader.dataset
        sample_batch = next(iter(train_loader))
        # sample_batch['bands'] 可能是 list 或 tensor；取第0频段样本结构以读取形状
        if isinstance(sample_batch['bands'], list):
            first_band = sample_batch['bands'][0]
            # first_band shape (batch, channels, samples) 或 (channels, samples)
            if isinstance(first_band, torch.Tensor):
                n_channels = first_band.shape[1]
                n_samples = first_band.shape[2]
            else:
                # fallback: inspect element
                n_channels = sample_batch['labels'].shape[1]
                n_samples = first_band[0].shape[-1] if isinstance(first_band, list) else sample_batch['bands'][0].shape[-1]
            n_bands = len(sample_batch['bands'])
        else:
            # bands is tensor shape (batch, n_bands, channels, samples)
            bands_tensor = sample_batch['bands']
            if bands_tensor.ndim == 4:
                n_bands = bands_tensor.shape[1]
                n_channels = bands_tensor.shape[2]
                n_samples = bands_tensor.shape[3]
            elif bands_tensor.ndim == 3:
                n_bands = 1
                n_channels = bands_tensor.shape[1]
                n_samples = bands_tensor.shape[2]
            else:
                raise RuntimeError("无法识别 sample_batch['bands'] 的形状")

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
    fold_results = kfold_cross_validation_optimized(
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

    best_f1_scores = [result['best_val_f1'] for result in fold_results]
    mean_f1 = np.mean(best_f1_scores)
    std_f1 = np.std(best_f1_scores)

    print(f"\n{'='*80}")
    print("K折交叉验证结果 (优化版, 修复)")
    print(f"{'='*80}")
    print(f"各折最佳F1分数:")
    for i, f1 in enumerate(best_f1_scores):
        print(f"  Fold {i+1}: {f1:.2f}%")

    print(f"\n平均结果:")
    print(f"  平均F1: {mean_f1:.2f}% ± {std_f1:.2f}%")
    print(f"  最佳F1: {max(best_f1_scores):.2f}%")
    print(f"  最差F1: {min(best_f1_scores):.2f}%")

    results_summary = {
        'fold_results': fold_results,
        'best_f1_scores': best_f1_scores,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'max_f1': float(max(best_f1_scores)),
        'min_f1': float(min(best_f1_scores))
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
            '--batch_size', '16',
            '--d_model', '128',
            '--n_heads', '4',
            '--n_layers', '2',
            '--n_folds', '5',
            '--n_epochs', '30',
            '--lr', '0.00001',
            # 默认不启用 IoU, 如果想启用在命令行加 --use_iou_loss
            '--iou_weight', '1.5',
            '--iou_type', 'basic',
        ])
    main()
