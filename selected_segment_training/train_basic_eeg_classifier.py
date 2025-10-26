#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_basic_eeg_classifier.py

基础EEG通道分类器训练脚本
专门用于发作前期显著通道标记任务
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
import gc
from typing import Dict, List, Tuple
import torch.nn.functional as F


# 导入基础模型
from basic_eeg_channel_classifier import (
    create_basic_eeg_classifier,
    FocalLoss,
    AdaptiveClassWeights,
    compute_multilabel_metrics
)

# 导入数据集
from dataset_selected import create_dataloaders


class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, mode='max', min_delta=0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
    def _is_better(self, current, best):
        if self.mode == 'max':
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta


def save_checkpoint(state: Dict, is_best: bool, save_dir: Path):
    """保存检查点"""
    checkpoint_path = save_dir / 'checkpoint.pth'
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = save_dir / 'best_model.pth'
        torch.save(state, best_path)


class BasicEEGTrainer:
    """基础EEG分类器训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: torch.device,
        save_dir: Path,
        n_channels: int,
        use_class_weights: bool = True,
        early_stopping_patience: int = 20
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.n_channels = n_channels
        self.use_class_weights = use_class_weights
        
        # 类别权重计算器
        self.class_weight_calculator = AdaptiveClassWeights(n_channels)
        
        # 早停机制
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        self.best_val_f1 = 0.0
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        losses = AverageMeter()
        metrics_meter = {
            'macro_f1': AverageMeter(),
            'macro_precision': AverageMeter(),
            'macro_recall': AverageMeter(),
            'micro_f1': AverageMeter()
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            bands = batch['bands']  # List of tensors
            labels = batch['labels'].to(self.device)  # (batch, n_channels)
            
            # 将多频段数据转换为列表格式
            bands_list = [band.to(self.device) for band in bands]
            
            # 前向传播
            logits = self.model(bands_list)  # (batch, n_channels)
            
            # 计算损失
            if self.use_class_weights:
                # 动态计算类别权重
                with torch.no_grad():
                    class_weights = self.class_weight_calculator.compute_weights(labels)
                    class_weights = class_weights.to(self.device)
                
                # 使用加权BCE损失
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels, 
                    pos_weight=class_weights
                )
            else:
                loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 计算指标
            with torch.no_grad():
                metrics = compute_multilabel_metrics(logits.detach(), labels.detach())
            
            # 更新统计
            losses.update(loss.item(), labels.size(0))
            for key in metrics_meter:
                if key in metrics:
                    metrics_meter[key].update(metrics[key], labels.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'F1': f'{metrics_meter["macro_f1"].avg:.1f}%',
                'Micro_F1': f'{metrics_meter["micro_f1"].avg:.1f}%',
                'mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
            })
            
            # 显式释放内存
            del bands, labels, bands_list, logits, loss
            
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'loss': losses.avg,
            'macro_precision': metrics_meter['macro_precision'].avg,
            'macro_recall': metrics_meter['macro_recall'].avg,
            'macro_f1': metrics_meter['macro_f1'].avg,
            'micro_f1': metrics_meter['micro_f1'].avg
        }
    
    @torch.no_grad()
    def validate(self, epoch: int, phase: str = 'Val') -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        losses = AverageMeter()
        metrics_meter = {
            'macro_f1': AverageMeter(),
            'macro_precision': AverageMeter(),
            'macro_recall': AverageMeter(),
            'micro_f1': AverageMeter()
        }
        
        loader = self.val_loader if phase == 'Val' else self.test_loader
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
        for batch in pbar:
            bands = batch['bands']
            labels = batch['labels'].to(self.device)
            
            # 将多频段数据转换为列表格式
            bands_list = [band.to(self.device) for band in bands]
            
            # 前向传播
            logits = self.model(bands_list)
            
            # 计算损失
            if self.use_class_weights:
                with torch.no_grad():
                    class_weights = self.class_weight_calculator.compute_weights(labels)
                    class_weights = class_weights.to(self.device)
                
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels, 
                    pos_weight=class_weights
                )
            else:
                loss = self.criterion(logits, labels)
            
            # 计算指标
            metrics = compute_multilabel_metrics(logits, labels)
            
            # 更新统计
            losses.update(loss.item(), labels.size(0))
            for key in metrics_meter:
                if key in metrics:
                    metrics_meter[key].update(metrics[key], labels.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'F1': f'{metrics_meter["macro_f1"].avg:.1f}%',
                'Micro_F1': f'{metrics_meter["micro_f1"].avg:.1f}%'
            })
            
            del bands, labels, bands_list, logits, loss
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'loss': losses.avg,
            'macro_precision': metrics_meter['macro_precision'].avg,
            'macro_recall': metrics_meter['macro_recall'].avg,
            'macro_f1': metrics_meter['macro_f1'].avg,
            'micro_f1': metrics_meter['micro_f1'].avg
        }
    
    def train(self, n_epochs: int) -> Tuple[float, float]:
        """训练主循环"""
        print(f"\n{'='*80}")
        print("开始训练基础EEG通道分类器")
        print(f"{'='*80}")
        print(f"设备: {self.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"训练集: {len(self.train_loader.dataset)}")
        print(f"验证集: {len(self.val_loader.dataset)}")
        print(f"测试集: {len(self.test_loader.dataset)}")
        print(f"使用类别权重: {self.use_class_weights}")
        
        if torch.cuda.is_available():
            print(f"初始显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        for epoch in range(1, n_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch, 'Val')
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印结果
            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"F1: {train_metrics['macro_f1']:.2f}%, "
                  f"Micro_F1: {train_metrics['micro_f1']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"F1: {val_metrics['macro_f1']:.2f}%, "
                  f"Micro_F1: {val_metrics['micro_f1']:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            if torch.cuda.is_available():
                print(f"  显存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB (峰值)")
                torch.cuda.reset_peak_memory_stats()
            
            # 保存最佳模型
            is_best = val_metrics['macro_f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['macro_f1']
                print(f"  -> 新的最佳F1: {val_metrics['macro_f1']:.2f}%")
            
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_f1': self.best_val_f1,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                },
                is_best,
                self.save_dir
            )
            
            # 早停检查
            self.early_stopping(val_metrics['macro_f1'])
            if self.early_stopping.early_stop:
                print(f"\n早停于epoch {epoch}")
                break
        
        # 测试集评估
        print("\n在测试集上评估...")
        checkpoint = torch.load(self.save_dir / 'best_model.pth', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.validate(n_epochs, 'Test')
        
        print(f"\n{'='*80}")
        print("测试集结果")
        print(f"{'='*80}")
        print(f"  Macro F1: {test_metrics['macro_f1']:.2f}%")
        print(f"  Macro Precision: {test_metrics['macro_precision']:.2f}%")
        print(f"  Macro Recall: {test_metrics['macro_recall']:.2f}%")
        print(f"  Micro F1: {test_metrics['micro_f1']:.2f}%")
        
        # 保存最终结果
        with open(self.save_dir / 'final_results.json', 'w') as f:
            json.dump({
                'best_val_f1': self.best_val_f1,
                'test_metrics': test_metrics
            }, f, indent=2)
        
        return self.best_val_f1, test_metrics['macro_f1']


def main():
    parser = argparse.ArgumentParser(description='基础EEG通道分类器训练')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录')
    parser.add_argument('--window_size', type=float, default=6.0,
                        help='窗口大小（秒）')
    parser.add_argument('--window_stride', type=float, default=3.0,
                        help='窗口步长（秒）')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128,
                        help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Transformer层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批大小')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='早停耐心值')
    
    # 损失函数参数
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='使用Focal Loss')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='使用类别权重')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                        help='Focal Loss alpha参数')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss gamma参数')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='checkpoints_basic_eeg',
                        help='保存目录')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='验证集比例')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='测试集比例')
    
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
    save_dir = Path(args.save_dir) / f"basic_eeg_{timestamp}"
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
            seed=args.seed
        )
        
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
        print(f"  通道名称: {channel_names}")
        
    except Exception as e:
        print(f"\n错误：加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模型
    print("\n创建基础EEG通道分类器...")
    model = create_basic_eeg_classifier(
        n_channels=n_channels,
        n_samples=n_samples,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    
    print(f"模型特性:")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  模型大小: ~{sum(p.numel() for p in model.parameters()) * 4 / (1024**2):.1f} MB")
    print(f"  输出维度: {n_channels} (每个通道一个二分类)")
    print(f"  架构: 多频段特征提取 + 通道注意力 + 时空编码")
    
    # 损失函数
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"使用Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"使用标准BCE Loss")
    
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
    trainer = BasicEEGTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        n_channels=n_channels,
        use_class_weights=args.use_class_weights,
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
            '--window_size', '6',
            '--window_stride', '3',
            '--batch_size', '8',
            '--d_model', '128',
            '--n_heads', '8',
            '--n_layers', '2',
            '--use_focal_loss',
            '--use_class_weights',
            '--n_epochs', '50'
        ])
    main()
