#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_channel_aware.py

基于通道感知模型的全频段训练脚本
使用ChannelAwareEEGNet进行多频段EEG数据训练
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

# 添加父目录到路径以导入原始模型
sys.path.append(str(Path(__file__).parent.parent / 'raw_data_training'))

from model_channel_aware import ChannelAwareEEGNet, create_channel_aware_model
from utils import AverageMeter, save_checkpoint, EarlyStopping
from dataset_selected import create_dataloaders


def compute_classification_metrics(pred_logits, true_labels):
    """计算分类指标"""
    pred_probs = torch.softmax(pred_logits, dim=-1)
    pred_classes = torch.argmax(pred_probs, dim=-1)
    
    # 准确率
    correct = (pred_classes == true_labels).float()
    accuracy = correct.mean().item() * 100
    
    # 每个类别的精确率、召回率、F1
    n_classes = pred_logits.size(-1)
    class_metrics = {}
    
    for cls in range(n_classes):
        true_positive = ((pred_classes == cls) & (true_labels == cls)).sum().item()
        false_positive = ((pred_classes == cls) & (true_labels != cls)).sum().item()
        false_negative = ((pred_classes != cls) & (true_labels == cls)).sum().item()
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[f'class_{cls}'] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
    
    # 宏平均
    macro_precision = np.mean([class_metrics[f'class_{cls}']['precision'] for cls in range(n_classes)])
    macro_recall = np.mean([class_metrics[f'class_{cls}']['recall'] for cls in range(n_classes)])
    macro_f1 = np.mean([class_metrics[f'class_{cls}']['f1'] for cls in range(n_classes)])
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'class_metrics': class_metrics
    }


class ChannelAwareTrainer:
    """通道感知模型训练器"""
    
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
        early_stopping_patience=30,
        gradient_accumulation_steps=1
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
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        self.best_val_f1 = 0.0
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        losses = AverageMeter()
        metrics_meter = {
            'accuracy': AverageMeter(),
            'macro_f1': AverageMeter(),
            'macro_precision': AverageMeter(),
            'macro_recall': AverageMeter()
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 获取多频段数据
            bands = batch['bands']  # List of tensors, each shape: (batch, n_channels, n_samples)
            labels = batch['labels'].to(self.device)  # (batch, n_channels) - 通道级别的标签
            
            # 将多频段数据融合为单一时域信号
            # 方法1: 简单平均所有频段
            fused_data = torch.stack(bands, dim=1)  # (batch, n_bands, n_channels, n_samples)
            fused_data = fused_data.mean(dim=1)  # (batch, n_channels, n_samples)
            fused_data = fused_data.to(self.device)
            
            # 创建通道掩码（基于标签）
            channel_mask = (labels > 0.5).float()  # (batch, n_channels)
            
            # Forward
            logits = self.model(fused_data, channel_mask)
            
            # 计算损失 - 使用通道级别的二分类损失
            # 将多通道标签转换为单通道分类任务
            # 方法：根据活跃通道数量进行分类
            n_active_channels = channel_mask.sum(dim=1)  # (batch,)
            
            # 创建分类标签：0-无活跃, 1-1个活跃, 2-2个活跃, 3-3个活跃, 4-4个或以上
            class_labels = torch.clamp(n_active_channels.long(), 0, 4)  # (batch,)
            
            # 如果模型输出维度不匹配，调整
            if logits.size(-1) != 5:
                # 重新创建模型以匹配分类数量
                print(f"警告：模型输出维度({logits.size(-1)})与分类数(5)不匹配，跳过此batch")
                continue
            
            loss = self.criterion(logits, class_labels)
            
            # 梯度累积
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 指标
            with torch.no_grad():
                metrics = compute_classification_metrics(logits.detach(), class_labels.detach())
            
            # 更新
            losses.update(loss.item() * self.gradient_accumulation_steps, labels.size(0))
            for key in metrics_meter:
                if key in metrics:
                    metrics_meter[key].update(metrics[key], labels.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{metrics_meter["accuracy"].avg:.1f}%',
                'F1': f'{metrics_meter["macro_f1"].avg:.1f}%',
                'mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
            })
            
            # 显式释放
            del bands, labels, fused_data, channel_mask, logits, loss
            
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            'loss': losses.avg,
            'accuracy': metrics_meter['accuracy'].avg,
            'macro_precision': metrics_meter['macro_precision'].avg,
            'macro_recall': metrics_meter['macro_recall'].avg,
            'macro_f1': metrics_meter['macro_f1'].avg
        }
    
    @torch.no_grad()
    def validate(self, epoch, phase='Val'):
        """验证"""
        self.model.eval()
        
        losses = AverageMeter()
        metrics_meter = {
            'accuracy': AverageMeter(),
            'macro_f1': AverageMeter(),
            'macro_precision': AverageMeter(),
            'macro_recall': AverageMeter()
        }
        
        loader = self.val_loader if phase == 'Val' else self.test_loader
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
        for batch in pbar:
            bands = batch['bands']
            labels = batch['labels'].to(self.device)
            
            # 融合多频段数据
            fused_data = torch.stack(bands, dim=1).mean(dim=1).to(self.device)
            channel_mask = (labels > 0.5).float()
            
            # Forward
            logits = self.model(fused_data, channel_mask)
            
            # 计算分类标签
            n_active_channels = channel_mask.sum(dim=1)
            class_labels = torch.clamp(n_active_channels.long(), 0, 4)
            
            if logits.size(-1) != 5:
                continue
            
            # Loss
            loss = self.criterion(logits, class_labels)
            
            # 指标
            metrics = compute_classification_metrics(logits, class_labels)
            
            # 更新
            losses.update(loss.item(), labels.size(0))
            for key in metrics_meter:
                if key in metrics:
                    metrics_meter[key].update(metrics[key], labels.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{metrics_meter["accuracy"].avg:.1f}%',
                'F1': f'{metrics_meter["macro_f1"].avg:.1f}%'
            })
            
            del bands, labels, fused_data, channel_mask, logits, loss
        
        torch.cuda.empty_cache()
        
        return {
            'loss': losses.avg,
            'accuracy': metrics_meter['accuracy'].avg,
            'macro_precision': metrics_meter['macro_precision'].avg,
            'macro_recall': metrics_meter['macro_recall'].avg,
            'macro_f1': metrics_meter['macro_f1'].avg
        }
    
    def train(self, n_epochs):
        """训练主循环"""
        print(f"\n{'='*80}")
        print("开始训练通道感知EEG分类模型")
        print(f"{'='*80}")
        print(f"设备: {self.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"训练集: {len(self.train_loader.dataset)}")
        print(f"验证集: {len(self.val_loader.dataset)}")
        print(f"测试集: {len(self.test_loader.dataset)}")
        print(f"梯度累积步数: {self.gradient_accumulation_steps}")
        
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
            
            # 打印
            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.2f}%, "
                  f"F1: {train_metrics['macro_f1']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.2f}%, "
                  f"F1: {val_metrics['macro_f1']:.2f}%")
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
                    'best_val_f1': self.best_val_f1,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                },
                is_best,
                self.save_dir
            )
            
            # Early stopping
            self.early_stopping(val_metrics['macro_f1'])
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # 测试集
        print("\n在测试集上评估...")
        checkpoint = torch.load(self.save_dir / 'best_model.pth', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.validate(n_epochs, 'Test')
        
        print(f"\n{'='*80}")
        print("测试集结果")
        print(f"{'='*80}")
        print(f"  准确率: {test_metrics['accuracy']:.2f}%")
        print(f"  F1分数: {test_metrics['macro_f1']:.2f}%")
        print(f"  Precision: {test_metrics['macro_precision']:.2f}%")
        print(f"  Recall: {test_metrics['macro_recall']:.2f}%")
        
        # 保存结果
        with open(self.save_dir / 'final_results.json', 'w') as f:
            json.dump({
                'best_val_f1': self.best_val_f1,
                'test_metrics': test_metrics
            }, f, indent=2)
        
        return self.best_val_f1, test_metrics['macro_f1']


def main():
    parser = argparse.ArgumentParser(description='通道感知EEG分类模型训练')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--window_size', type=float, default=6.0)
    parser.add_argument('--window_stride', type=float, default=3.0)
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--n_classes', type=int, default=5, 
                        help='分类数量：0-无活跃, 1-1个活跃, 2-2个活跃, 3-3个活跃, 4-4个或以上')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints_channel_aware')
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)
    
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
    save_dir = Path(args.save_dir) / f"channel_aware_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 加载数据
    print("\n准备数据...")
    print(f"  数据路径: {args.data_root}")
    print(f"  窗口大小: {args.window_size}秒")
    print(f"  窗口步长: {args.window_stride}秒")
    print(f"  分类数量: {args.n_classes}")
    
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
        
    except Exception as e:
        print(f"\n错误：加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模型
    print("\n创建通道感知模型...")
    model = create_channel_aware_model(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=args.n_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    
    print(f"模型特性:")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  模型大小: ~{sum(p.numel() for p in model.parameters()) * 4 / (1024**2):.1f} MB")
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
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
    trainer = ChannelAwareTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        early_stopping_patience=args.early_stopping_patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps
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
            '--n_heads', '4',
            '--n_layers', '2',
            '--n_classes', '5',
        ])
    main()
