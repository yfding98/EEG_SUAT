#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_band_specific.py

按频段分别训练的脚本
分别训练δ、θ、α、β、γ波对应的模型
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

from model_ranking_advanced import AdvancedChannelRankingModel, AdvancedChannelRankingLoss
from utils import AverageMeter, save_checkpoint, EarlyStopping
from dataset_selected import create_dataloaders


# 频段定义
BAND_NAMES = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
BAND_INDICES = [0, 1, 2, 3, 4]  # 对应6个频段中的前5个（排除HFO）


class BandSpecificDataset:
    """单频段数据集适配器"""
    
    def __init__(self, original_dataset, band_index):
        self.original_dataset = original_dataset
        self.band_index = band_index
        self.band_name = BAND_NAMES[band_index] if band_index < len(BAND_NAMES) else f'Band_{band_index}'
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        item = self.original_dataset[idx]
        
        # 只取指定频段的数据
        band_data = item['bands'][self.band_index]  # (n_channels, n_samples)
        
        return {
            'data': band_data,
            'labels': item['labels'],
            'fs': item['fs'],
            'file': item['file'],
            'abnormal_channels': item['abnormal_channels'],
            'band_name': self.band_name
        }


def create_band_specific_dataloaders(
    data_root,
    band_index,
    batch_size=8,
    window_size=6.0,
    window_stride=3.0,
    val_split=0.15,
    test_split=0.15,
    num_workers=0,
    seed=42
):
    """创建单频段数据加载器"""
    
    # 先创建原始的多频段数据加载器
    train_loader, val_loader, test_loader, channel_names = create_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        window_size=window_size,
        window_stride=window_stride,
        val_split=val_split,
        test_split=test_split,
        num_workers=num_workers,
        seed=seed
    )
    
    # 转换为单频段数据集
    train_dataset = BandSpecificDataset(train_loader.dataset, band_index)
    val_dataset = BandSpecificDataset(val_loader.dataset, band_index)
    test_dataset = BandSpecificDataset(test_loader.dataset, band_index)
    
    # 创建新的数据加载器
    train_loader_band = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader_band = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader_band = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader_band, val_loader_band, test_loader_band, channel_names


def compute_ranking_metrics(pred_scores, true_labels, k=None):
    """计算排序指标"""
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
        
        if k is None:
            use_k = max(1, true_k)
        else:
            use_k = k
        
        topk_idx = scores.topk(use_k).indices
        pred_mask = torch.zeros_like(labels, dtype=torch.bool)
        pred_mask[topk_idx] = True
        
        true_mask = labels == 1
        
        tp = (pred_mask & true_mask).sum().item()
        fp = (pred_mask & ~true_mask).sum().item()
        fn = (~pred_mask & true_mask).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        top1_idx = scores.argmax().item()
        top1_hit = labels[top1_idx].item()
        top1_hits.append(top1_hit)
        
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


class BandSpecificTrainer:
    """单频段训练器"""
    
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
        band_name,
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
        self.band_name = band_name
        
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
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [{self.band_name} Train]")
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            data = batch['data'].to(self.device)  # (batch, n_channels, n_samples)
            labels = batch['labels'].to(self.device)  # (batch, n_channels)
            
            # 将单频段数据包装成多频段格式（只有一个频段）
            bands = [data]  # List with single element
            
            # Forward
            scores = self.model(bands)
            
            # Loss
            loss, loss_dict = self.criterion(
                scores, 
                labels,
                channel_positions=None  # 不使用空间信息以节省显存
            )
            
            # 梯度累积
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 指标
            with torch.no_grad():
                metrics = compute_ranking_metrics(scores.detach(), labels.detach())
            
            # 更新
            losses.update(loss.item() * self.gradient_accumulation_steps, labels.size(0))
            for key in metrics_meter:
                metrics_meter[key].update(metrics[key], labels.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'F1': f'{metrics_meter["f1"].avg:.1f}%',
                'mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
            })
            
            # 显式释放
            del data, labels, bands, scores, loss
            
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        gc.collect()
        torch.cuda.empty_cache()
        
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
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{self.band_name} {phase}]")
        for batch in pbar:
            data = batch['data'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            bands = [data]
            
            # Forward
            scores = self.model(bands)
            
            # Loss
            loss, loss_dict = self.criterion(
                scores,
                labels,
                channel_positions=None
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
            
            del data, labels, bands, scores, loss
        
        torch.cuda.empty_cache()
        
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
        print(f"开始训练{self.band_name}频段排序模型")
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
            print(f"\nEpoch {epoch}/{n_epochs} [{self.band_name}]")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"F1: {train_metrics['f1']:.2f}%, "
                  f"IoU: {train_metrics['iou']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"F1: {val_metrics['f1']:.2f}%, "
                  f"IoU: {val_metrics['iou']:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            if torch.cuda.is_available():
                print(f"  显存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB (峰值)")
                torch.cuda.reset_peak_memory_stats()
            
            # 保存最佳模型
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
                print(f"  -> 新的最佳F1: {val_metrics['f1']:.2f}%")
            
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_f1': self.best_val_f1,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'band_name': self.band_name
                },
                is_best,
                self.save_dir
            )
            
            # Early stopping
            self.early_stopping(val_metrics['f1'])
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # 测试集
        print(f"\n在测试集上评估{self.band_name}模型...")
        checkpoint = torch.load(self.save_dir / 'best_model.pth', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.validate(n_epochs, 'Test')
        
        print(f"\n{'='*80}")
        print(f"{self.band_name}频段测试集结果")
        print(f"{'='*80}")
        print(f"  F1分数: {test_metrics['f1']:.2f}%")
        print(f"  IoU: {test_metrics['iou']:.2f}%")
        print(f"  Precision: {test_metrics['precision']:.2f}%")
        print(f"  Recall: {test_metrics['recall']:.2f}%")
        print(f"  Top-1准确率: {test_metrics['top1_accuracy']:.2f}%")
        
        # 保存结果
        with open(self.save_dir / 'final_results.json', 'w') as f:
            json.dump({
                'band_name': self.band_name,
                'best_val_f1': self.best_val_f1,
                'test_metrics': test_metrics
            }, f, indent=2)
        
        return self.best_val_f1, test_metrics['f1']


def train_single_band(
    band_index,
    data_root,
    save_base_dir,
    args
):
    """训练单个频段的模型"""
    
    band_name = BAND_NAMES[band_index] if band_index < len(BAND_NAMES) else f'Band_{band_index}'
    
    print(f"\n{'='*100}")
    print(f"开始训练 {band_name} 频段模型 (索引: {band_index})")
    print(f"{'='*100}")
    
    # 设置随机种子
    torch.manual_seed(args.seed + band_index)  # 每个频段使用不同的种子
    np.random.seed(args.seed + band_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + band_index)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(save_base_dir) / f"{band_name.lower()}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config = vars(args).copy()
    config['band_index'] = band_index
    config['band_name'] = band_name
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # 加载单频段数据
    print(f"\n准备{band_name}频段数据...")
    try:
        train_loader, val_loader, test_loader, channel_names = create_band_specific_dataloaders(
            data_root=data_root,
            band_index=band_index,
            batch_size=args.batch_size,
            window_size=args.window_size,
            window_stride=args.window_stride,
            val_split=args.val_split,
            test_split=args.test_split,
            num_workers=0,
            seed=args.seed + band_index
        )
        
        sample_batch = next(iter(train_loader))
        n_channels = sample_batch['data'].shape[1]
        n_samples = sample_batch['data'].shape[2]
        
        print(f"\n{band_name}频段数据信息:")
        print(f"  通道数: {n_channels}")
        print(f"  时间点数: {n_samples}")
        print(f"  训练集: {len(train_loader.dataset)}")
        print(f"  验证集: {len(val_loader.dataset)}")
        print(f"  测试集: {len(test_loader.dataset)}")
        
    except Exception as e:
        print(f"\n错误：加载{band_name}频段数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # 创建模型
    print(f"\n创建{band_name}频段排序模型...")
    model = AdvancedChannelRankingModel(
        n_channels=n_channels,
        n_samples=n_samples,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_gcn_layers=0,  # 不使用GCN以节省显存
        dropout=args.dropout,
        use_multiband=False,  # 单频段模式
        use_gcn=False,
        channel_positions=None
    )
    model = model.to(device)
    
    print(f"{band_name}模型特性:")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  模型大小: ~{sum(p.numel() for p in model.parameters()) * 4 / (1024**2):.1f} MB")
    
    # 损失函数
    criterion = AdvancedChannelRankingLoss(
        score_weight=args.score_weight,
        margin_weight=args.margin_weight,
        topk_weight=args.topk_weight,
        contrastive_weight=0.0,  # 关闭对比学习
        spatial_weight=0.0,      # 关闭空间损失
        network_weight=0.0       # 关闭网络损失
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
    
    # 训练器
    trainer = BandSpecificTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        band_name=band_name,
        early_stopping_patience=args.early_stopping_patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # 训练
    best_val_f1, test_f1 = trainer.train(args.n_epochs)
    
    print(f"\n{band_name}频段训练完成!")
    print(f"  最佳验证F1: {best_val_f1:.2f}%")
    print(f"  测试F1: {test_f1:.2f}%")
    print(f"  检查点: {save_dir}")
    
    return best_val_f1, test_f1


def main():
    parser = argparse.ArgumentParser(description='按频段分别训练EEG通道排序模型')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--window_size', type=float, default=6.0)
    parser.add_argument('--window_stride', type=float, default=3.0)
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    
    # 损失函数参数
    parser.add_argument('--score_weight', type=float, default=3.0)
    parser.add_argument('--margin_weight', type=float, default=1.0)
    parser.add_argument('--topk_weight', type=float, default=2.0)
    
    # 频段选择
    parser.add_argument('--bands', type=str, nargs='+', default=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                        help='要训练的频段列表')
    parser.add_argument('--band_indices', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='对应的频段索引')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints_band_specific')
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)
    
    args = parser.parse_args()
    
    # 频段名称到索引的映射
    band_name_to_index = {
        'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3, 'gamma': 4
    }
    
    # 解析频段参数
    if args.bands and args.band_indices:
        band_indices = []
        for band_name in args.bands:
            if band_name.lower() in band_name_to_index:
                band_indices.append(band_name_to_index[band_name.lower()])
            else:
                print(f"警告：未知频段名称 {band_name}")
        args.band_indices = band_indices
    
    print(f"\n{'='*100}")
    print("按频段分别训练EEG通道排序模型")
    print(f"{'='*100}")
    print(f"要训练的频段: {[BAND_NAMES[i] for i in args.band_indices]}")
    print(f"频段索引: {args.band_indices}")
    print(f"数据路径: {args.data_root}")
    print(f"保存目录: {args.save_dir}")
    
    # 训练结果汇总
    results = {}
    
    # 逐个训练每个频段
    for band_index in args.band_indices:
        try:
            best_val_f1, test_f1 = train_single_band(
                band_index=band_index,
                data_root=args.data_root,
                save_base_dir=args.save_dir,
                args=args
            )
            
            band_name = BAND_NAMES[band_index] if band_index < len(BAND_NAMES) else f'Band_{band_index}'
            results[band_name] = {
                'best_val_f1': best_val_f1,
                'test_f1': test_f1
            }
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
        except Exception as e:
            print(f"\n错误：训练频段{band_index}失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印最终结果汇总
    print(f"\n{'='*100}")
    print("所有频段训练结果汇总")
    print(f"{'='*100}")
    
    for band_name, metrics in results.items():
        print(f"{band_name:>8}: 验证F1={metrics['best_val_f1']:6.2f}%, 测试F1={metrics['test_f1']:6.2f}%")
    
    # 保存汇总结果
    summary_path = Path(args.save_dir) / 'training_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n训练汇总已保存到: {summary_path}")


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
            '--bands', 'delta', 'theta', 'alpha', 'beta', 'gamma',
        ])
    main()
