"""
改进的训练脚本
- 数据增强
- 更好的正则化
- 学习率warmup
- 梯度裁剪
- 支持badcase过滤
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

from dataset_filtered import create_filtered_dataloaders
from model import create_model
from utils import (
    AverageMeter, accuracy, save_checkpoint, load_checkpoint,
    EarlyStopping, get_lr_scheduler
)


class DataAugmentation:
    """EEG数据增强"""
    
    def __init__(self, noise_std=0.05, scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.scale_range = scale_range
    
    def add_noise(self, data):
        """添加高斯噪声"""
        noise = torch.randn_like(data) * self.noise_std
        return data + noise
    
    def scale(self, data):
        """幅度缩放"""
        scale = torch.FloatTensor(1).uniform_(*self.scale_range).to(data.device)
        return data * scale
    
    def time_shift(self, data, max_shift=50):
        """时间平移"""
        shift = np.random.randint(-max_shift, max_shift)
        if shift > 0:
            data = torch.cat([data[:, shift:], data[:, :shift]], dim=1)
        elif shift < 0:
            data = torch.cat([data[:, shift:], data[:, :shift]], dim=1)
        return data
    
    def __call__(self, data):
        """随机应用增强"""
        if torch.rand(1) < 0.5:
            data = self.add_noise(data)
        if torch.rand(1) < 0.5:
            data = self.scale(data)
        if torch.rand(1) < 0.3:
            data = self.time_shift(data)
        return data


class ImprovedTrainer:
    """改进的训练器"""
    
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
        early_stopping_patience: int = 20,
        use_augmentation: bool = True,
        gradient_clip: float = 1.0,
        warmup_epochs: int = 5
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
        self.gradient_clip = gradient_clip
        self.warmup_epochs = warmup_epochs
        
        # 数据增强
        self.augmentation = DataAugmentation() if use_augmentation else None
        
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
    
    def get_lr_scale(self, epoch):
        """学习率warmup"""
        if epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs
        return 1.0
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        # Warmup学习率
        lr_scale = self.get_lr_scale(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_scale
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            data = batch['data'].to(self.device)
            target = batch['label'].to(self.device)
            
            # 数据增强
            if self.augmentation is not None:
                data = self.augmentation(data)
            
            # Forward
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
            
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
        
        # 恢复学习率
        if lr_scale != 1.0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / lr_scale
        
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
        print(f"开始训练 {n_epochs} epochs...")
        print(f"设备: {self.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"数据增强: {'启用' if self.augmentation else '禁用'}")
        print(f"梯度裁剪: {self.gradient_clip}")
        print(f"Warmup轮数: {self.warmup_epochs}")
        
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
                print(f"  -> 新的最佳验证准确率: {val_acc:.2f}%")
            
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
                print(f"\nEarly stopping触发于epoch {epoch}")
                break
        
        # 训练完成
        print(f"\n训练完成!")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        
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
        print("\n在测试集上评估...")
        checkpoint = load_checkpoint(self.save_dir / 'best_model.pth', self.model)
        test_loss, test_acc, test_preds, test_targets = self.validate(n_epochs, 'Test')
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
    parser = argparse.ArgumentParser(description='改进的EEG分类训练')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, 
                        default=r'E:\DataSet\EEG\EEG dataset_SUAT_processed')
    parser.add_argument('--labels_csv', type=str,
                        default=r'E:\output\connectivity_features\labels.csv')
    parser.add_argument('--window_size', type=float, default=6.0)
    parser.add_argument('--bad_windows_file', type=str, default=None,
                        help='Badcase过滤文件 (filtered_dataset.json)')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='spatiotemporal',
                        choices=['spatiotemporal', 'lightweight'])
    parser.add_argument('--n_classes', type=int, default=5)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--early_stopping_patience', type=int, default=30)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    
    # 增强和正则化
    parser.add_argument('--use_augmentation', action='store_true', default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalization', type=str, default='window_robust',
                        choices=['none', 'window_zscore', 'window_robust', 
                                'channel_zscore', 'channel_robust'],
                        help='数据归一化方法')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='checkpoints_improved')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f"{args.model_type}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 创建数据加载器（带过滤）
    print("创建数据加载器...")
    print(f"归一化方法: {args.normalization}")
    train_loader, val_loader, test_loader = create_filtered_dataloaders(
        data_root=args.data_root,
        labels_csv=args.labels_csv,
        batch_size=args.batch_size,
        window_size=args.window_size,
        num_workers=args.num_workers,
        seed=args.seed,
        bad_windows_file=args.bad_windows_file,
        normalization=args.normalization
    )
    
    # 获取数据形状
    sample_batch = next(iter(train_loader))
    n_channels = sample_batch['data'].shape[1]
    n_samples = sample_batch['data'].shape[2]
    
    print(f"数据形状: channels={n_channels}, samples={n_samples}")
    
    # 创建模型
    print(f"创建 {args.model_type} 模型...")
    model = create_model(
        model_type=args.model_type,
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=args.n_classes,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # 损失函数（带标签平滑）
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
    trainer = ImprovedTrainer(
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
        use_augmentation=args.use_augmentation,
        gradient_clip=args.gradient_clip,
        warmup_epochs=args.warmup_epochs
    )
    
    # 训练
    best_val_acc, test_acc = trainer.train(args.n_epochs)
    
    print(f"\n最终结果:")
    print(f"  最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  测试准确率: {test_acc:.2f}%")
    print(f"\n检查点保存到: {save_dir}")


if __name__ == "__main__":
    main()

