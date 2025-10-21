"""
通道感知训练脚本
使用活跃通道信息引导模型学习
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
from model_channel_aware import create_channel_aware_model
from utils import (
    AverageMeter, accuracy, save_checkpoint, load_checkpoint,
    EarlyStopping
)


class ChannelAwareTrainer:
    """通道感知训练器"""
    
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
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            data = batch['data'].to(self.device)
            target = batch['label'].to(self.device)
            channel_mask = batch['channel_mask'].to(self.device)
            
            # Forward - 传入通道掩码
            output = self.model(data, channel_mask)
            loss = self.criterion(output, target)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 统计
            acc = accuracy(output, target)[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc.item(), data.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{top1.avg:.2f}%'
            })
        
        return losses.avg, top1.avg
    
    def validate(self, epoch, phase='Val'):
        """验证"""
        self.model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        all_preds = []
        all_targets = []
        
        loader = self.val_loader if phase == 'Val' else self.test_loader
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
            for batch in pbar:
                data = batch['data'].to(self.device)
                target = batch['label'].to(self.device)
                channel_mask = batch['channel_mask'].to(self.device)
                
                # Forward
                output = self.model(data, channel_mask)
                loss = self.criterion(output, target)
                
                # 统计
                acc = accuracy(output, target)[0]
                losses.update(loss.item(), data.size(0))
                top1.update(acc.item(), data.size(0))
                
                _, pred = output.max(1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'acc': f'{top1.avg:.2f}%'
                })
        
        return losses.avg, top1.avg, np.array(all_preds), np.array(all_targets)
    
    def train(self, n_epochs):
        """训练主循环"""
        print(f"\n开始训练 {n_epochs} epochs...")
        print(f"设备: {self.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, n_epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 验证
            val_loss, val_acc, _, _ = self.validate(epoch, 'Val')
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # TensorBoard
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # 打印信息
            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # 保存最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                print(f"  -> 新的最佳验证准确率: {val_acc:.2f}%")
            
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc
                },
                is_best,
                self.save_dir
            )
            
            # Early stopping
            self.early_stopping(val_acc)
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # 测试集评估
        print("\n在测试集上评估...")
        load_checkpoint(self.save_dir / 'best_model.pth', self.model)
        test_loss, test_acc, test_preds, test_targets = self.validate(n_epochs, 'Test')
        print(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # 保存结果
        np.savez(
            self.save_dir / 'test_results.npz',
            predictions=test_preds,
            targets=test_targets,
            accuracy=test_acc
        )
        
        self.writer.close()
        
        return self.best_val_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='通道感知EEG分类训练')
    
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
    parser.add_argument('--save_dir', type=str, default='checkpoints_channel_aware')
    
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
    save_dir = Path(args.save_dir) / f"channel_aware_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 创建数据加载器
    print("创建通道感知数据加载器...")
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
    
    print(f"数据形状: channels={n_channels}, samples={n_samples}")
    print(f"通道掩码形状: {sample_batch['channel_mask'].shape}")
    
    # 创建模型
    print(f"创建通道感知模型...")
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
    
    # 损失函数
    from utils import LabelSmoothing
    criterion = LabelSmoothing(n_classes=args.n_classes, smoothing=0.1)
    
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

