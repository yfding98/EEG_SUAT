"""
活跃通道检测训练脚本
纯粹的通道检测任务，不做发作类型分类
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from dataset_channel_aware import create_channel_aware_dataloaders
from model_channel_detection import (
    create_channel_detector, 
    FocalBCELoss, 
    BalancedBCEWithCardinality
)
from utils import AverageMeter, save_checkpoint, load_checkpoint, EarlyStopping


def compute_channel_metrics(pred_probs, true_mask, threshold=0.5):
    """
    计算通道检测的评估指标
    
    Args:
        pred_probs: (batch, n_channels) 预测概率
        true_mask: (batch, n_channels) 真实标签
        threshold: 二值化阈值
    
    Returns:
        dict of metrics
    """
    pred_binary = (pred_probs > threshold).float()
    
    # 逐样本计算
    batch_size = pred_probs.size(0)
    precisions = []
    recalls = []
    f1s = []
    
    # 统计预测分布
    total_pred_positive = 0
    total_true_positive = 0
    total_samples_pred_all_zero = 0
    
    for i in range(batch_size):
        pred = pred_binary[i]
        true = true_mask[i]
        
        tp = (pred * true).sum().item()
        fp = (pred * (1 - true)).sum().item()
        fn = ((1 - pred) * true).sum().item()
        
        # 检查是否预测全0
        if pred.sum().item() == 0:
            total_samples_pred_all_zero += 1
        
        total_pred_positive += pred.sum().item()
        total_true_positive += true.sum().item()
        
        # 处理边界情况：如果模型预测全0，precision为0但不是NaN
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    # 整体准确率（注意：这个指标在不平衡数据中意义不大）
    accuracy = (pred_binary == true_mask).float().mean().item()
    
    # 平均预测的活跃通道数
    avg_pred_active = total_pred_positive / batch_size
    avg_true_active = total_true_positive / batch_size
    
    return {
        'accuracy': accuracy * 100,
        'precision': np.mean(precisions) * 100,
        'recall': np.mean(recalls) * 100,
        'f1': np.mean(f1s) * 100,
        'avg_pred_active': avg_pred_active,
        'avg_true_active': avg_true_active,
        'samples_pred_all_zero': total_samples_pred_all_zero,
        'pred_all_zero_rate': total_samples_pred_all_zero / batch_size * 100
    }


def check_model_collapse(metrics, threshold=0.8):
    """
    检测模型是否退化为预测全0
    
    Args:
        metrics: compute_channel_metrics返回的指标
        threshold: 如果超过这个比例的样本预测全0，则认为模型崩溃
    
    Returns:
        bool: True表示模型可能崩溃
    """
    return metrics['pred_all_zero_rate'] > threshold * 100


class ChannelDetectionTrainer:
    """活跃通道检测训练器"""
    
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
        
        self.writer = SummaryWriter(log_dir=self.save_dir / "logs")
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        
        self.best_val_f1 = 0.0
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        f1s = AverageMeter()
        avg_pred_actives = AverageMeter()
        avg_true_actives = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            data = batch['data'].to(self.device)
            channel_mask = batch['channel_mask'].to(self.device)
            
            # Forward
            channel_logits = self.model(data)
            loss = self.criterion(channel_logits, channel_mask)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 计算指标
            with torch.no_grad():
                metrics = compute_channel_metrics(
                    torch.sigmoid(channel_logits), channel_mask
                )
            
            losses.update(loss.item(), data.size(0))
            accuracies.update(metrics['accuracy'], data.size(0))
            precisions.update(metrics['precision'], data.size(0))
            recalls.update(metrics['recall'], data.size(0))
            f1s.update(metrics['f1'], data.size(0))
            avg_pred_actives.update(metrics['avg_pred_active'], data.size(0))
            avg_true_actives.update(metrics['avg_true_active'], data.size(0))
            
            # 检测模型崩溃（预测全0）
            if batch_idx % 50 == 0 and check_model_collapse(metrics, threshold=0.8):
                print(f"\n⚠️  警告: {metrics['pred_all_zero_rate']:.1f}% 样本预测全0！模型可能陷入局部最优。")
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'F1': f'{f1s.avg:.2f}%',
                'R': f'{recalls.avg:.2f}%',
                'PredAvg': f'{avg_pred_actives.avg:.1f}'
            })
        
        return {
            'loss': losses.avg,
            'accuracy': accuracies.avg,
            'precision': precisions.avg,
            'recall': recalls.avg,
            'f1': f1s.avg,
            'avg_pred_active': avg_pred_actives.avg,
            'avg_true_active': avg_true_actives.avg
        }
    
    def validate(self, epoch, phase='Val'):
        """验证"""
        self.model.eval()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        f1s = AverageMeter()
        avg_pred_actives = AverageMeter()
        avg_true_actives = AverageMeter()
        all_zero_count = 0
        total_count = 0
        
        loader = self.val_loader if phase == 'Val' else self.test_loader
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
            for batch in pbar:
                data = batch['data'].to(self.device)
                channel_mask = batch['channel_mask'].to(self.device)
                
                # Forward
                channel_logits = self.model(data)
                loss = self.criterion(channel_logits, channel_mask)
                
                # 计算指标
                metrics = compute_channel_metrics(
                    torch.sigmoid(channel_logits), channel_mask
                )
                
                losses.update(loss.item(), data.size(0))
                accuracies.update(metrics['accuracy'], data.size(0))
                precisions.update(metrics['precision'], data.size(0))
                recalls.update(metrics['recall'], data.size(0))
                f1s.update(metrics['f1'], data.size(0))
                avg_pred_actives.update(metrics['avg_pred_active'], data.size(0))
                avg_true_actives.update(metrics['avg_true_active'], data.size(0))
                
                all_zero_count += metrics['samples_pred_all_zero']
                total_count += data.size(0)
                
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'F1': f'{f1s.avg:.2f}%',
                    'R': f'{recalls.avg:.2f}%',
                    'PredAvg': f'{avg_pred_actives.avg:.1f}'
                })
        
        pred_all_zero_rate = all_zero_count / total_count * 100 if total_count > 0 else 0
        
        return {
            'loss': losses.avg,
            'accuracy': accuracies.avg,
            'precision': precisions.avg,
            'recall': recalls.avg,
            'f1': f1s.avg,
            'avg_pred_active': avg_pred_actives.avg,
            'avg_true_active': avg_true_actives.avg,
            'pred_all_zero_rate': pred_all_zero_rate
        }
    
    def train(self, n_epochs):
        """训练主循环"""
        print(f"\n开始训练活跃通道检测器 {n_epochs} epochs...")
        print(f"设备: {self.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"\n任务：从21个通道中识别2-5个活跃通道（发作源）")
        
        for epoch in range(1, n_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch, 'Val')
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # TensorBoard
            self.writer.add_scalars('Loss', {
                'train': train_metrics['loss'],
                'val': val_metrics['loss']
            }, epoch)
            
            for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
                self.writer.add_scalars(f'Metrics/{metric_name}', {
                    'train': train_metrics[metric_name],
                    'val': val_metrics[metric_name]
                }, epoch)
            
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # 打印信息
            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"F1: {train_metrics['f1']:.2f}%, "
                  f"Recall: {train_metrics['recall']:.2f}%")
            print(f"          PredAvg: {train_metrics['avg_pred_active']:.2f}, "
                  f"TrueAvg: {train_metrics['avg_true_active']:.2f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"F1: {val_metrics['f1']:.2f}%, "
                  f"Recall: {val_metrics['recall']:.2f}%")
            print(f"          Precision: {val_metrics['precision']:.2f}%, "
                  f"PredAvg: {val_metrics['avg_pred_active']:.2f}")
            print(f"          预测全0率: {val_metrics['pred_all_zero_rate']:.1f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # 警告：模型可能崩溃
            if val_metrics['pred_all_zero_rate'] > 50:
                print(f"  ⚠️  警告: 超过50%的验证样本预测全0，模型可能陷入局部最优！")
                print(f"      建议: 1) 增加pos_weight, 2) 降低学习率, 3) 检查数据")
            
            # 保存最佳模型（基于F1分数）
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
                print(f"  -> 新的最佳F1分数: {val_metrics['f1']:.2f}%")
            
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
            self.early_stopping(val_metrics['f1'])
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # 测试集评估
        print("\n在测试集上评估...")
        load_checkpoint(self.save_dir / 'best_model.pth', self.model)
        test_metrics = self.validate(n_epochs, 'Test')
        
        print(f"\n测试集结果:")
        print(f"  F1分数: {test_metrics['f1']:.2f}%")
        print(f"  准确率: {test_metrics['accuracy']:.2f}%")
        print(f"  精确率: {test_metrics['precision']:.2f}%")
        print(f"  召回率: {test_metrics['recall']:.2f}%")
        
        # 保存结果
        with open(self.save_dir / 'final_results.json', 'w') as f:
            json.dump({
                'best_val_f1': self.best_val_f1,
                'test_metrics': test_metrics
            }, f, indent=2)
        
        self.writer.close()
        
        return self.best_val_f1, test_metrics['f1']


def main():
    parser = argparse.ArgumentParser(description='活跃通道检测训练')
    
    # 数据参数
    parser.add_argument('--data_root', type=str,
                        default=r'E:\DataSet\EEG\EEG dataset_SUAT_processed')
    parser.add_argument('--labels_csv', type=str,
                        default=r'E:\output\connectivity_features\labels.csv')
    parser.add_argument('--window_size', type=float, default=6.0)
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--early_stopping_patience', type=int, default=30)
    
    # Loss参数
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['focal', 'balanced_bce'],
                        help='损失函数类型: focal或balanced_bce')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss alpha (正样本权重)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma (难样本权重)')
    parser.add_argument('--pos_weight', type=float, default=8,
                        help='正样本权重（用于处理类别不平衡），推荐10-20')
    parser.add_argument('--cardinality_weight', type=float, default=0.5,
                        help='基数损失权重（鼓励预测合理数量的活跃通道）')
    
    # 其他
    parser.add_argument('--normalization', type=str, default='window_robust')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='checkpoints_channel_detection')
    
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
    save_dir = Path(args.save_dir) / f"detector_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 创建数据加载器
    print("创建数据加载器...")
    print(f"归一化方法: {args.normalization}")
    train_loader, val_loader, test_loader = create_channel_aware_dataloaders(
        data_root=args.data_root,
        labels_csv=args.labels_csv,
        batch_size=args.batch_size,
        window_size=args.window_size,
        num_workers=args.num_workers,
        seed=args.seed,
        normalization=args.normalization
    )
    
    # 获取数据形状和统计信息
    sample_batch = next(iter(train_loader))
    n_channels = sample_batch['data'].shape[1]
    n_samples = sample_batch['data'].shape[2]
    
    # 统计数据集中的正负样本比例
    print("\n统计数据集中的活跃通道分布...")
    total_positive = 0
    total_negative = 0
    total_samples = 0
    active_counts = []
    
    for batch in train_loader:
        channel_mask = batch['channel_mask']
        total_positive += channel_mask.sum().item()
        total_negative += (1 - channel_mask).sum().item()
        total_samples += channel_mask.size(0)
        active_counts.extend(channel_mask.sum(dim=1).tolist())
    
    pos_ratio = total_positive / (total_positive + total_negative)
    neg_ratio = total_negative / (total_positive + total_negative)
    avg_active = total_positive / total_samples
    
    # 计算推荐的pos_weight（负样本/正样本）
    recommended_pos_weight = neg_ratio / pos_ratio if pos_ratio > 0 else 10.0
    
    print(f"\n数据信息:")
    print(f"  数据形状: channels={n_channels}, samples={n_samples}")
    print(f"  任务：从{n_channels}个通道中识别活跃通道（通常2-5个）")
    print(f"\n数据集统计:")
    print(f"  总样本数: {total_samples}")
    print(f"  平均活跃通道数: {avg_active:.2f}")
    print(f"  活跃通道比例: {pos_ratio*100:.2f}%")
    print(f"  非活跃通道比例: {neg_ratio*100:.2f}%")
    print(f"  推荐pos_weight: {recommended_pos_weight:.1f}")
    print(f"  实际使用pos_weight: {args.pos_weight}")
    
    # 创建模型
    print(f"\n创建活跃通道检测器...")
    model = create_channel_detector(
        n_channels=n_channels,
        n_samples=n_samples,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # 损失函数
    print(f"\n损失函数: {args.loss_type}")
    if args.loss_type == 'focal':
        pos_weight_tensor = torch.tensor([args.pos_weight], device=device)
        criterion = FocalBCELoss(
            alpha=args.focal_alpha, 
            gamma=args.focal_gamma,
            pos_weight=pos_weight_tensor
        )
        print(f"  Focal Loss - alpha={args.focal_alpha}, gamma={args.focal_gamma}, pos_weight={args.pos_weight}")
    else:  # balanced_bce
        criterion = BalancedBCEWithCardinality(
            pos_weight=args.pos_weight,
            cardinality_weight=args.cardinality_weight,
            expected_active=(2, 5)
        )
        print(f"  Balanced BCE - pos_weight={args.pos_weight}, cardinality_weight={args.cardinality_weight}")
        print(f"  这个loss会严重惩罚漏检（FN），并鼓励预测2-5个活跃通道")
    
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
    
    # 创建训练器
    trainer = ChannelDetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # 训练
    best_val_f1, test_f1 = trainer.train(args.n_epochs)
    
    print(f"\n最终结果:")
    print(f"  最佳验证F1: {best_val_f1:.2f}%")
    print(f"  测试F1: {test_f1:.2f}%")
    print(f"\n检查点保存到: {save_dir}")


if __name__ == "__main__":
    main()

