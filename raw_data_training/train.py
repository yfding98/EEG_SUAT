"""
训练脚本
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
import os

from dataset import create_dataloaders
from model import create_model
from utils import (
    AverageMeter, accuracy, save_checkpoint, load_checkpoint,
    EarlyStopping, get_lr_scheduler
)


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_dir: str,
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
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.save_dir / "logs")
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        
        # 统计信息
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            data = batch['data'].to(self.device)
            target = batch['label'].to(self.device)
            
            # Forward
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            acc = accuracy(output, target)[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc.item(), data.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{top1.avg:.2f}%'
            })
        
        return losses.avg, top1.avg
    
    def validate(self, epoch: int, phase: str = 'Val'):
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
                
                # Forward
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # 统计
                acc = accuracy(output, target)[0]
                losses.update(loss.item(), data.size(0))
                top1.update(acc.item(), data.size(0))
                
                # 保存预测结果
                _, pred = output.max(1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'acc': f'{top1.avg:.2f}%'
                })
        
        return losses.avg, top1.avg, np.array(all_preds), np.array(all_targets)
    
    def train(self, n_epochs: int):
        """训练主循环"""
        print(f"Starting training for {n_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, n_epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 验证
            val_loss, val_acc, val_preds, val_targets = self.validate(epoch, 'Val')
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # TensorBoard记录
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # 打印信息
            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # 保存最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                print(f"  -> New best validation accuracy: {val_acc:.2f}%")
            
            # 保存检查点
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'best_val_acc': self.best_val_acc
                },
                is_best,
                self.save_dir
            )
            
            # Early stopping
            self.early_stopping(val_acc)
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # 训练完成
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc
        }
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # 在测试集上评估
        print("\nEvaluating on test set...")
        checkpoint = load_checkpoint(self.save_dir / 'best_model.pth', self.model)
        test_loss, test_acc, test_preds, test_targets = self.validate(epoch, 'Test')
        print(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # 保存测试结果
        np.savez(
            self.save_dir / 'test_results.npz',
            predictions=test_preds,
            targets=test_targets,
            accuracy=test_acc
        )
        
        self.writer.close()
        
        return self.best_val_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='Train EEG Classification Model')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, 
                        default=r'E:\DataSet\EEG\EEG dataset_SUAT_processed',
                        help='Data root directory')
    parser.add_argument('--labels_csv', type=str,
                        default=r'E:\output\connectivity_features\labels.csv',
                        help='Labels CSV file')
    parser.add_argument('--window_size', type=float, default=6.0,
                        help='Window size in seconds')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='lightweight',
                        choices=['spatiotemporal', 'lightweight'],
                        help='Model architecture')
    parser.add_argument('--n_classes', type=int, default=5,
                        help='Number of classes')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Early stopping patience')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f"{args.model_type}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 创建数据加载器
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        labels_csv=args.labels_csv,
        batch_size=args.batch_size,
        window_size=args.window_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # 获取数据形状
    sample_batch = next(iter(train_loader))
    n_channels = sample_batch['data'].shape[1]
    n_samples = sample_batch['data'].shape[2]
    
    print(f"Data shape: channels={n_channels}, samples={n_samples}")
    
    # 创建模型
    print(f"Creating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=args.n_classes
    )
    model = model.to(device)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = get_lr_scheduler(
        optimizer,
        scheduler_type='cosine',
        n_epochs=args.n_epochs
    )
    
    # 创建训练器
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
        early_stopping_patience=args.early_stopping_patience
    )
    
    # 训练
    best_val_acc, test_acc = trainer.train(args.n_epochs)
    
    print(f"\nFinal Results:")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Test Acc: {test_acc:.2f}%")
    print(f"\nCheckpoints saved to: {save_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.argv.extend([
            '--data_root', r'E:\DataSet\EEG\EEG dataset_SUAT_processed',
            '--labels_csv', r'E:\output\connectivity_features\labels.csv'
        ])

    sys.exit(main())

