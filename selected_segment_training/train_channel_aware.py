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

from model_channel_aware_multilabel import ChannelAwareMultilabelNet, create_channel_aware_multilabel_model
from utils import AverageMeter, save_checkpoint, EarlyStopping
from dataset_selected import create_dataloaders


def analyze_channel_distribution(data_loader):
    """分析数据集中每个通道的标签分布"""
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
    
    # 高级类别权重计算
    class_weights = torch.ones(n_channels)
    
    for i, ratio in enumerate(positive_ratios):
        if ratio == 0.0:
            # 从未出现的通道：极小权重，几乎完全忽略
            class_weights[i] = 0.01
        elif ratio < 0.05:
            # 极稀有通道：高权重保护
            class_weights[i] = 10.0
        elif ratio < 0.1:
            # 稀有通道：较高权重
            class_weights[i] = 5.0
        elif ratio < 0.2:
            # 低频通道：中等权重
            class_weights[i] = 3.0
        elif ratio < 0.4:
            # 中频通道：正常权重
            class_weights[i] = 2.0
        else:
            # 高频通道：低权重
            class_weights[i] = 1.0
    
    # 归一化权重
    class_weights = class_weights / class_weights.mean()
    
    # 统计不同频率通道
    never_positive = (positive_ratios == 0.0)
    rare_positive = (positive_ratios > 0.0) & (positive_ratios < 0.1)
    medium_positive = (positive_ratios >= 0.1) & (positive_ratios < 0.3)
    frequent_positive = (positive_ratios >= 0.3)
    
    print(f"高级类别权重:")
    print(f"  从未出现通道: {never_positive.sum().item()}个, 平均权重: {class_weights[never_positive].mean():.3f}")
    print(f"  稀有通道: {rare_positive.sum().item()}个, 平均权重: {class_weights[rare_positive].mean():.3f}")
    print(f"  中频通道: {medium_positive.sum().item()}个, 平均权重: {class_weights[medium_positive].mean():.3f}")
    print(f"  高频通道: {frequent_positive.sum().item()}个, 平均权重: {class_weights[frequent_positive].mean():.3f}")
    
    # 详细权重显示
    print(f"详细权重分布:")
    for i, weight in enumerate(class_weights):
        ratio = positive_ratios[i]
        if ratio == 0.0:
            print(f"  通道{i}: {weight:.3f} (从未出现 - 几乎忽略)")
        elif ratio < 0.05:
            print(f"  通道{i}: {weight:.3f} (极稀有 - 高权重保护)")
        elif ratio < 0.1:
            print(f"  通道{i}: {weight:.3f} (稀有 - 较高权重)")
        elif ratio < 0.2:
            print(f"  通道{i}: {weight:.3f} (低频 - 中等权重)")
        elif ratio < 0.4:
            print(f"  通道{i}: {weight:.3f} (中频 - 正常权重)")
        else:
            print(f"  通道{i}: {weight:.3f} (高频 - 低权重)")
    
    return class_weights, positive_ratios


def compute_multilabel_metrics(pred_logits, true_labels, threshold=0.5):
    """计算多标签分类指标"""
    pred_probs = torch.sigmoid(pred_logits)
    pred_binary = (pred_probs > threshold).float()
    
    batch_size, n_channels = pred_logits.shape
    
    # 每个通道的指标
    per_channel_metrics = {}
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
        
        per_channel_metrics[f'channel_{ch}'] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
        
        channel_precisions.append(precision)
        channel_recalls.append(recall)
        channel_f1s.append(f1)
    
    # 宏平均
    macro_precision = np.mean(channel_precisions) * 100
    macro_recall = np.mean(channel_recalls) * 100
    macro_f1 = np.mean(channel_f1s) * 100
    
    # 微平均
    total_tp = sum([(pred_binary[:, ch] * true_labels[:, ch]).sum().item() for ch in range(n_channels)])
    total_fp = sum([(pred_binary[:, ch] * (1 - true_labels[:, ch])).sum().item() for ch in range(n_channels)])
    total_fn = sum([((1 - pred_binary[:, ch]) * true_labels[:, ch]).sum().item() for ch in range(n_channels)])
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
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
        'micro_precision': micro_precision * 100,
        'micro_recall': micro_recall * 100,
        'micro_f1': micro_f1 * 100,
        'mAP': mAP,
        'per_channel_metrics': per_channel_metrics
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
        n_channels,
        class_weights,
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
        
        # 多标签分类信息
        self.n_channels = n_channels
        self.class_weights = class_weights  # 已经在主函数中移动到正确设备
        
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
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 获取多频段数据
            bands = batch['bands']  # List of tensors, each shape: (batch, n_channels, n_samples)
            labels = batch['labels'].to(self.device)  # (batch, n_channels) - 通道级别的标签
            
            # 将多频段数据堆叠为 (batch, n_bands, n_channels, n_samples)
            bands_tensor = torch.stack(bands, dim=1).to(self.device)
            
            # Forward - 模型现在直接处理多频段数据
            logits = self.model(bands_tensor, labels)  # (batch, n_channels)
            
            # 多标签二分类损失
            loss = self.criterion(logits, labels)
            
            # 梯度累积
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
        
        loader = self.val_loader if phase == 'Val' else self.test_loader
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
        for batch in pbar:
            bands = batch['bands']
            labels = batch['labels'].to(self.device)
            
            # 将多频段数据堆叠为 (batch, n_bands, n_channels, n_samples)
            bands_tensor = torch.stack(bands, dim=1).to(self.device)
            
            # Forward
            logits = self.model(bands_tensor, labels)
            
            # Loss
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
        best_model_path = self.save_dir / 'best_model.pth'
        if best_model_path.exists():
            print("加载最佳模型进行测试...")
            checkpoint = torch.load(best_model_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("未找到最佳模型，使用当前模型进行测试...")
            # 尝试加载最后一个checkpoint
            checkpoint_path = self.save_dir / 'checkpoint.pth'
            if checkpoint_path.exists():
                import shutil
                shutil.copyfile(checkpoint_path, best_model_path)
                print("已将最后的checkpoint复制为best_model.pth")
                checkpoint = torch.load(best_model_path, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("警告：未找到任何模型文件，使用当前模型状态")
        
        test_metrics = self.validate(n_epochs, 'Test')
        
        print(f"\n{'='*80}")
        print("测试集结果")
        print(f"{'='*80}")
        print(f"  Macro F1: {test_metrics['macro_f1']:.2f}%")
        print(f"  Macro Precision: {test_metrics['macro_precision']:.2f}%")
        print(f"  Macro Recall: {test_metrics['macro_recall']:.2f}%")
        print(f"  mAP: {test_metrics['mAP']:.2f}%")
        
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
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='使用类别权重处理不平衡问题')
    
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
        
        # 分析通道标签分布
        class_weights, positive_ratios = analyze_channel_distribution(train_loader)
        n_channels = sample_batch['bands'][0].shape[1]
        
    except Exception as e:
        print(f"\n错误：加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模型
    print("\n创建通道感知多标签分类模型...")
    model = create_channel_aware_multilabel_model(
        n_channels=n_channels,
        n_samples=n_samples,
        n_bands=len(sample_batch['bands']),  # 频段数量
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
    print(f"  频段融合: 使用可学习注意力机制")
    
    # 损失函数 - 多标签二分类
    if args.use_class_weights:
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
        print(f"使用类别权重处理不平衡问题")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"不使用类别权重")
    
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
        n_channels=n_channels,
        class_weights=class_weights.to(device),
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
            '--use_class_weights',  # 使用类别权重
        ])
    main()
