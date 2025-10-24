#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_selected.py

针对_selected.set文件的高级通道排序模型训练脚本

基于raw_data_training/train_ranking_advanced.py，适配selected数据集
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import sys

# 添加父目录到路径以导入原始模型
sys.path.append(str(Path(__file__).parent.parent / 'raw_data_training'))

from model_ranking_advanced import (
    AdvancedChannelRankingModel,
    AdvancedChannelRankingLoss,
    compute_spatial_adjacency
)

from extract_multiband_features import (
    get_channel_positions_array
)

from utils import AverageMeter, save_checkpoint, EarlyStopping
from dataset_selected import create_dataloaders


def compute_ranking_metrics(pred_scores, true_labels, k=None):
    """
    计算排序指标
    
    Args:
        pred_scores: (batch, n_channels)
        true_labels: (batch, n_channels)
        k: 固定K，如果None则使用真实K
    """
    batch_size = pred_scores.size(0)
    
    precisions = []
    recalls = []
    f1s = []
    top1_hits = []
    ious = []
    
    for i in range(batch_size):
        scores = pred_scores[i]
        labels = true_labels[i]
        
        true_k = (labels == 1).sum().item()
        if true_k == 0:
            continue
        
        # 选择K
        if k is None:
            use_k = max(1, true_k)
        else:
            use_k = k
        
        # Top-K预测
        topk_idx = scores.topk(use_k).indices
        pred_mask = torch.zeros_like(labels, dtype=torch.bool)
        pred_mask[topk_idx] = True
        
        true_mask = labels == 1
        
        # 指标
        tp = (pred_mask & true_mask).sum().item()
        fp = (pred_mask & ~true_mask).sum().item()
        fn = (~pred_mask & true_mask).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        # Top-1
        top1_idx = scores.argmax().item()
        top1_hit = labels[top1_idx].item()
        top1_hits.append(top1_hit)
        
        # IoU
        intersection = tp
        union = (pred_mask | true_mask).sum().item()
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
    
    return {
        'precision': np.mean(precisions) * 100 if precisions else 0,
        'recall': np.mean(recalls) * 100 if recalls else 0,
        'f1': np.mean(f1s) * 100 if f1s else 0,
        'top1_accuracy': np.mean(top1_hits) * 100 if top1_hits else 0,
        'iou': np.mean(ious) * 100 if ious else 0
    }


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_dir,
        channel_positions=None,
        early_stopping_patience=30
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 通道位置（用于空间损失）
        if channel_positions is not None:
            self.channel_positions = torch.from_numpy(channel_positions).float().to(device)
        else:
            self.channel_positions = None
        
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        self.best_val_f1 = 0.0
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        losses = AverageMeter()
        metrics_meter = {
            'precision': AverageMeter(),
            'recall': AverageMeter(),
            'f1': AverageMeter(),
            'iou': AverageMeter()
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            bands = batch['bands']  # list of tensors
            labels = batch['labels'].to(self.device)
            
            # 将所有频段移到设备
            bands = [b.to(self.device) for b in bands]
            
            # Forward
            scores = self.model(bands)
            
            # Loss
            loss, loss_dict = self.criterion(
                scores, 
                labels,
                channel_positions=self.channel_positions
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 指标
            with torch.no_grad():
                metrics = compute_ranking_metrics(scores, labels)
            
            # 更新
            losses.update(loss.item(), labels.size(0))
            for key in metrics_meter:
                metrics_meter[key].update(metrics[key], labels.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'F1': f'{metrics_meter["f1"].avg:.1f}%',
                'IoU': f'{metrics_meter["iou"].avg:.1f}%'
            })
        
        return {
            'loss': losses.avg,
            'precision': metrics_meter['precision'].avg,
            'recall': metrics_meter['recall'].avg,
            'f1': metrics_meter['f1'].avg,
            'iou': metrics_meter['iou'].avg
        }
    
    @torch.no_grad()
    def validate(self, epoch, phase='Val'):
        """验证"""
        self.model.eval()
        
        losses = AverageMeter()
        metrics_meter = {
            'precision': AverageMeter(),
            'recall': AverageMeter(),
            'f1': AverageMeter(),
            'top1_accuracy': AverageMeter(),
            'iou': AverageMeter()
        }
        
        loader = self.val_loader if phase == 'Val' else self.test_loader
        
        # 如果验证/测试集为空，返回默认值
        if len(loader.dataset) == 0:
            return {
                'loss': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'top1_accuracy': 0.0,
                'iou': 0.0
            }
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
        for batch in pbar:
            bands = batch['bands']
            labels = batch['labels'].to(self.device)
            
            bands = [b.to(self.device) for b in bands]
            
            # Forward
            scores = self.model(bands)
            
            # Loss
            loss, loss_dict = self.criterion(
                scores,
                labels,
                channel_positions=self.channel_positions
            )
            
            # 指标
            metrics = compute_ranking_metrics(scores, labels)
            
            # 更新
            losses.update(loss.item(), labels.size(0))
            for key in metrics_meter:
                if key in metrics:
                    metrics_meter[key].update(metrics[key], labels.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'F1': f'{metrics_meter["f1"].avg:.1f}%'
            })
        
        return {
            'loss': losses.avg,
            'precision': metrics_meter['precision'].avg,
            'recall': metrics_meter['recall'].avg,
            'f1': metrics_meter['f1'].avg,
            'top1_accuracy': metrics_meter['top1_accuracy'].avg,
            'iou': metrics_meter['iou'].avg
        }
    
    def train(self, n_epochs):
        """训练主循环"""
        print(f"\n{'='*80}")
        print("开始训练Selected Segments排序模型")
        print(f"{'='*80}")
        print(f"设备: {self.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"训练集: {len(self.train_loader.dataset)}")
        print(f"验证集: {len(self.val_loader.dataset)}")
        print(f"测试集: {len(self.test_loader.dataset)}")
        
        if len(self.val_loader.dataset) == 0:
            print("\n⚠ 警告：验证集为空，将使用训练指标进行模型选择")
        if len(self.test_loader.dataset) == 0:
            print("⚠ 警告：测试集为空")
        
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
            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"F1: {train_metrics['f1']:.2f}%, "
                  f"IoU: {train_metrics['iou']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"F1: {val_metrics['f1']:.2f}%, "
                  f"IoU: {val_metrics['iou']:.2f}%")
            print(f"          Precision: {val_metrics['precision']:.2f}%, "
                  f"Recall: {val_metrics['recall']:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # 保存最佳模型（如果验证集为空，使用训练F1）
            if len(self.val_loader.dataset) == 0:
                # 验证集为空时，每个epoch都保存为最佳（或使用训练F1判断）
                current_f1 = train_metrics['f1']
                is_best = current_f1 > self.best_val_f1
                if is_best:
                    self.best_val_f1 = current_f1
                    print(f"  -> 新的最佳训练F1: {current_f1:.2f}% (验证集为空)")
            else:
                is_best = val_metrics['f1'] > self.best_val_f1
                if is_best:
                    self.best_val_f1 = val_metrics['f1']
                    print(f"  -> 新的最佳验证F1: {val_metrics['f1']:.2f}%")
            
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_f1': self.best_val_f1,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                },
                is_best,
                self.save_dir
            )
            
            # Early stopping（如果验证集为空，使用训练F1）
            if len(self.val_loader.dataset) == 0:
                self.early_stopping(train_metrics['f1'])
            else:
                self.early_stopping(val_metrics['f1'])
            
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # 测试集（如果存在best_model.pth就加载，否则使用当前模型）
        best_model_path = self.save_dir / 'best_model.pth'
        if best_model_path.exists():
            print("\n加载最佳模型并在测试集上评估...")
            checkpoint = torch.load(best_model_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("\n使用当前模型在测试集上评估...")
            # 手动保存一个best_model（使用最后的checkpoint）
            checkpoint_path = self.save_dir / 'checkpoint.pth'
            if checkpoint_path.exists():
                import shutil
                shutil.copyfile(checkpoint_path, best_model_path)
                print("  (已将最后的checkpoint复制为best_model.pth)")
        
        test_metrics = self.validate(n_epochs, 'Test')
        
        print(f"\n{'='*80}")
        print("测试集结果")
        print(f"{'='*80}")
        print(f"  F1分数: {test_metrics['f1']:.2f}%")
        print(f"  IoU: {test_metrics['iou']:.2f}%")
        print(f"  Precision: {test_metrics['precision']:.2f}%")
        print(f"  Recall: {test_metrics['recall']:.2f}%")
        print(f"  Top-1准确率: {test_metrics['top1_accuracy']:.2f}%")
        
        # 保存结果
        with open(self.save_dir / 'final_results.json', 'w') as f:
            json.dump({
                'best_val_f1': self.best_val_f1,
                'test_metrics': test_metrics
            }, f, indent=2)
        
        return self.best_val_f1, test_metrics['f1']


def main():
    parser = argparse.ArgumentParser(description='Selected Segments通道排序模型训练')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录（包含_selected.set文件）')
    parser.add_argument('--window_size', type=float, default=6.0,
                        help='时间窗口大小（秒）')
    parser.add_argument('--window_stride', type=float, default=3.0,
                        help='窗口步长（秒）')
    parser.add_argument('--target_sfreq', type=float, default=250.0,
                        help='目标采样率（Hz），所有数据会重采样到此频率')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--use_multiband', action='store_true', default=True)
    parser.add_argument('--use_gcn', action='store_true', default=True)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    
    # 损失函数参数
    parser.add_argument('--score_weight', type=float, default=3.0)
    parser.add_argument('--margin_weight', type=float, default=1.0)
    parser.add_argument('--topk_weight', type=float, default=2.0)
    parser.add_argument('--contrastive_weight', type=float, default=0.0)
    parser.add_argument('--spatial_weight', type=float, default=0.5)
    parser.add_argument('--network_weight', type=float, default=0.0)
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints_selected')
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f"selected_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 加载数据
    print("\n准备数据...")
    print(f"  数据路径: {args.data_root}")
    print(f"  窗口大小: {args.window_size}秒")
    print(f"  窗口步长: {args.window_stride}秒")
    
    try:
        train_loader, val_loader, test_loader, channel_names = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            window_size=args.window_size,
            window_stride=args.window_stride,
            val_split=args.val_split,
            test_split=args.test_split,
            num_workers=0,
            seed=args.seed,
            target_sfreq=args.target_sfreq
        )
        
        # 从loader获取样本信息
        sample_batch = next(iter(train_loader))
        n_channels = sample_batch['bands'][0].shape[1]
        n_samples = sample_batch['bands'][0].shape[2]
        
        print(f"\n数据信息:")
        print(f"  通道数: {n_channels}")
        print(f"  时间点数: {n_samples}")
        print(f"  频段数: {len(sample_batch['bands'])}")
        print(f"  训练集: {len(train_loader.dataset)}")
        print(f"  验证集: {len(val_loader.dataset)}")
        print(f"  测试集: {len(test_loader.dataset)}")
        
    except Exception as e:
        print(f"\n错误：加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 获取通道位置
    channel_positions = get_channel_positions_array(channel_names)
    
    # 创建模型
    print("\n创建高级排序模型...")
    model = AdvancedChannelRankingModel(
        n_channels=n_channels,
        n_samples=n_samples,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_gcn_layers=2,
        dropout=args.dropout,
        use_multiband=args.use_multiband,
        use_gcn=args.use_gcn,
        channel_positions=channel_positions if args.use_gcn else None
    )
    model = model.to(device)
    
    print(f"模型特性:")
    print(f"  多频段: {'✓' if args.use_multiband else '✗'}")
    print(f"  图卷积: {'✓' if args.use_gcn else '✗'}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数
    criterion = AdvancedChannelRankingLoss(
        score_weight=args.score_weight,
        margin_weight=args.margin_weight,
        topk_weight=args.topk_weight,
        contrastive_weight=args.contrastive_weight,
        spatial_weight=args.spatial_weight,
        network_weight=args.network_weight
    )
    
    print(f"\n损失函数配置:")
    print(f"  Score: {args.score_weight}")
    print(f"  Margin: {args.margin_weight}")
    print(f"  Top-K: {args.topk_weight}")
    print(f"  Contrastive: {args.contrastive_weight}")
    print(f"  Spatial: {args.spatial_weight}")
    print(f"  Network: {args.network_weight}")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=1e-6
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        channel_positions=channel_positions if args.spatial_weight > 0 else None,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # 训练
    best_val_f1, test_f1 = trainer.train(args.n_epochs)
    
    print(f"\n{'='*80}")
    print("训练完成!")
    print(f"{'='*80}")
    print(f"  最佳验证F1: {best_val_f1:.2f}%")
    print(f"  测试F1: {test_f1:.2f}%")
    print(f"  检查点: {save_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--data_root', r'E:\DataSet\EEG\EEG dataset_SUAT_processed_selected',
            '--window_size', '30',
            '--window_stride', '30',
        ])
    main()

