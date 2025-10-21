"""
多任务训练脚本
显式利用通道组合信息进行训练
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
from model_multitask import create_multitask_model, MultiTaskLoss
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint, EarlyStopping


class MultiTaskTrainer:
    """多任务训练器"""
    
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
        
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        losses_total = AverageMeter()
        losses_seizure = AverageMeter()
        losses_channel = AverageMeter()
        losses_relation = AverageMeter()
        
        seizure_acc = AverageMeter()
        channel_acc = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            data = batch['data'].to(self.device)
            seizure_target = batch['label'].to(self.device)
            channel_mask = batch['channel_mask'].to(self.device)
            
            # Forward - 多任务输出
            seizure_logits, channel_logits, relation_logits = self.model(data)
            
            # 计算多任务loss
            losses = self.criterion(
                seizure_logits, channel_logits, relation_logits,
                seizure_target, channel_mask
            )
            
            total_loss = losses['total']
            
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 统计
            acc_sz = accuracy(seizure_logits, seizure_target)[0]
            channel_preds = (torch.sigmoid(channel_logits) > 0.5).float()
            acc_ch = (channel_preds == channel_mask).float().mean() * 100
            
            losses_total.update(total_loss.item(), data.size(0))
            losses_seizure.update(losses['seizure'], data.size(0))
            losses_channel.update(losses['channel'], data.size(0))
            losses_relation.update(losses['relation'], data.size(0))
            seizure_acc.update(acc_sz.item(), data.size(0))
            channel_acc.update(acc_ch.item(), data.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses_total.avg:.4f}',
                'sz_acc': f'{seizure_acc.avg:.1f}%',
                'ch_acc': f'{channel_acc.avg:.1f}%'
            })
        
        return {
            'total_loss': losses_total.avg,
            'seizure_loss': losses_seizure.avg,
            'channel_loss': losses_channel.avg,
            'relation_loss': losses_relation.avg,
            'seizure_acc': seizure_acc.avg,
            'channel_acc': channel_acc.avg
        }
    
    def validate(self, epoch, phase='Val'):
        """验证"""
        self.model.eval()
        
        losses_total = AverageMeter()
        seizure_acc = AverageMeter()
        channel_acc = AverageMeter()
        
        all_preds = []
        all_targets = []
        
        loader = self.val_loader if phase == 'Val' else self.test_loader
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
            for batch in pbar:
                data = batch['data'].to(self.device)
                seizure_target = batch['label'].to(self.device)
                channel_mask = batch['channel_mask'].to(self.device)
                
                # Forward
                seizure_logits, channel_logits, relation_logits = self.model(data)
                
                losses = self.criterion(
                    seizure_logits, channel_logits, relation_logits,
                    seizure_target, channel_mask
                )
                
                # 统计
                acc_sz = accuracy(seizure_logits, seizure_target)[0]
                channel_preds = (torch.sigmoid(channel_logits) > 0.5).float()
                acc_ch = (channel_preds == channel_mask).float().mean() * 100
                
                losses_total.update(losses['total'].item(), data.size(0))
                seizure_acc.update(acc_sz.item(), data.size(0))
                channel_acc.update(acc_ch.item(), data.size(0))
                
                _, pred = seizure_logits.max(1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(seizure_target.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{losses_total.avg:.4f}',
                    'sz_acc': f'{seizure_acc.avg:.1f}%',
                    'ch_acc': f'{channel_acc.avg:.1f}%'
                })
        
        return {
            'total_loss': losses_total.avg,
            'seizure_acc': seizure_acc.avg,
            'channel_acc': channel_acc.avg,
            'predictions': np.array(all_preds),
            'targets': np.array(all_targets)
        }
    
    def train(self, n_epochs):
        """训练主循环"""
        print(f"\n开始多任务训练 {n_epochs} epochs...")
        print(f"设备: {self.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"\n任务说明:")
        print(f"  主任务: 发作类型分类 (权重1.0)")
        print(f"  辅助任务1: 活跃通道预测 (权重0.5)")
        print(f"  辅助任务2: 通道关系学习 (权重0.3)")
        
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
            self.writer.add_scalars('Loss/Total', {
                'train': train_metrics['total_loss'],
                'val': val_metrics['total_loss']
            }, epoch)
            self.writer.add_scalars('Loss/Seizure', {
                'train': train_metrics['seizure_loss']
            }, epoch)
            self.writer.add_scalars('Loss/Channel', {
                'train': train_metrics['channel_loss']
            }, epoch)
            self.writer.add_scalars('Accuracy/Seizure', {
                'train': train_metrics['seizure_acc'],
                'val': val_metrics['seizure_acc']
            }, epoch)
            self.writer.add_scalars('Accuracy/Channel', {
                'train': train_metrics['channel_acc'],
                'val': val_metrics['channel_acc']
            }, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # 打印信息
            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"  Train:")
            print(f"    总Loss: {train_metrics['total_loss']:.4f}")
            print(f"    发作类型Acc: {train_metrics['seizure_acc']:.2f}%")
            print(f"    通道预测Acc: {train_metrics['channel_acc']:.2f}%")
            print(f"  Val:")
            print(f"    总Loss: {val_metrics['total_loss']:.4f}")
            print(f"    发作类型Acc: {val_metrics['seizure_acc']:.2f}%")
            print(f"    通道预测Acc: {val_metrics['channel_acc']:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # 保存最佳模型（基于发作类型准确率）
            is_best = val_metrics['seizure_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['seizure_acc']
                print(f"  -> 新的最佳发作类型准确率: {val_metrics['seizure_acc']:.2f}%")
            
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                },
                is_best,
                self.save_dir
            )
            
            # Early stopping
            self.early_stopping(val_metrics['seizure_acc'])
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # 测试集评估
        print("\n在测试集上评估...")
        load_checkpoint(self.save_dir / 'best_model.pth', self.model)
        test_metrics = self.validate(n_epochs, 'Test')
        
        print(f"\n测试集结果:")
        print(f"  发作类型准确率: {test_metrics['seizure_acc']:.2f}%")
        print(f"  通道预测准确率: {test_metrics['channel_acc']:.2f}%")
        
        # 保存结果
        np.savez(
            self.save_dir / 'test_results.npz',
            predictions=test_metrics['predictions'],
            targets=test_metrics['targets'],
            seizure_accuracy=test_metrics['seizure_acc'],
            channel_accuracy=test_metrics['channel_acc']
        )
        
        self.writer.close()
        
        return self.best_val_acc, test_metrics['seizure_acc']


def main():
    parser = argparse.ArgumentParser(description='多任务EEG分类训练')
    
    # 数据参数
    parser.add_argument('--data_root', type=str,
                        default=r'E:\DataSet\EEG\EEG dataset_SUAT_processed')
    parser.add_argument('--labels_csv', type=str,
                        default=r'E:\output\connectivity_features\labels.csv')
    parser.add_argument('--window_size', type=float, default=6.0)
    
    # 模型参数
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # 多任务权重
    parser.add_argument('--seizure_weight', type=float, default=1.0)
    parser.add_argument('--channel_weight', type=float, default=0.5)
    parser.add_argument('--relation_weight', type=float, default=0.3)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--early_stopping_patience', type=int, default=30)
    
    # 其他
    parser.add_argument('--normalization', type=str, default='window_robust')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='checkpoints_multitask')
    
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
    save_dir = Path(args.save_dir) / f"multitask_{timestamp}"
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
    
    # 获取数据形状
    sample_batch = next(iter(train_loader))
    n_channels = sample_batch['data'].shape[1]
    n_samples = sample_batch['data'].shape[2]
    
    print(f"\n数据信息:")
    print(f"  数据形状: channels={n_channels}, samples={n_samples}")
    print(f"  通道掩码形状: {sample_batch['channel_mask'].shape}")
    print(f"  示例活跃通道数: {sample_batch['channel_mask'][0].sum().item():.0f}")
    
    # 创建模型
    print(f"\n创建多任务模型...")
    model = create_multitask_model(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=args.n_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # 多任务损失函数
    criterion = MultiTaskLoss(
        n_classes=args.n_classes,
        n_channels=n_channels,
        seizure_weight=args.seizure_weight,
        channel_weight=args.channel_weight,
        relation_weight=args.relation_weight
    )
    
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
    trainer = MultiTaskTrainer(
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
    best_val_acc, test_acc = trainer.train(args.n_epochs)
    
    print(f"\n最终结果:")
    print(f"  最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  测试准确率: {test_acc:.2f}%")
    print(f"\n检查点保存到: {save_dir}")


if __name__ == "__main__":
    main()

