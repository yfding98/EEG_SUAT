#!/usr/bin/env python3
"""
干净的基线训练 - 实施所有诊断性改进

改进：
1. ✅ 关闭 SMOTE 和 Mixup
2. ✅ 保守的采样权重 (weight_per_positive=3.0)
3. ✅ Per-matrix 归一化
4. ✅ 自动阈值搜索
5. ✅ 单度量基线选项
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, hamming_loss, jaccard_score,
    average_precision_score
)
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from datasets_multilabel_filtered import FilteredMultiLabelDataset
from models_multilabel import MultiLabelGNNClassifier
from losses import FocalLoss
from samplers_advanced import create_weighted_sampler


def find_optimal_threshold(probs, labels, metric='f1'):
    """
    在验证集上搜索最优阈值
    
    Args:
        probs: [N, C] 预测概率
        labels: [N, C] 真实标签
        metric: 优化目标 ('f1', 'jaccard')
    
    Returns:
        best_threshold: float
        best_score: float
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0.0
    
    for thresh in thresholds:
        preds = (probs > thresh).astype(np.float32)
        
        if metric == 'f1':
            # 使用 micro F1
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            score = f1
        else:  # jaccard
            score = jaccard_score(labels, preds, average='samples', zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


@torch.no_grad()
def evaluate(model, loader, device, channel_names, threshold=0.5):
    """评估模型"""
    model.eval()
    
    all_labels, all_preds, all_probs = [], [], []
    
    if len(loader) == 0:
        return {
            'hamming_loss': 0,
            'jaccard_macro': 0,
            'jaccard_samples': 0,
            'map': 0
        }, np.array([]), np.array([]), np.array([])
    
    for batch in tqdm(loader, desc='Evaluating', leave=False):
        adj = batch['adj'].to(device)
        x = batch['x'].to(device)
        labels = batch['y'].cpu().numpy()
        
        logits = model(x, adj)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > threshold).astype(np.float32)
        
        all_labels.append(labels)
        all_preds.append(preds)
        all_probs.append(probs)
    
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    
    metrics = {
        'hamming_loss': hamming_loss(all_labels, all_preds),
        'jaccard_macro': jaccard_score(all_labels, all_preds, average='macro', zero_division=0),
        'jaccard_samples': jaccard_score(all_labels, all_preds, average='samples', zero_division=0),
        'map': average_precision_score(all_labels, all_probs, average='macro'),
    }
    
    return metrics, all_probs, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description='Clean Baseline Training')
    
    # 数据
    parser.add_argument('--features_root', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--matrix_keys', nargs='+', default=['pearson'])
    parser.add_argument('--min_channel_samples', type=int, default=15)
    
    # 模型（保守设置）
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # 训练
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    
    # 不平衡处理（保守）
    parser.add_argument('--weight_per_positive', type=float, default=3.0)
    parser.add_argument('--base_gamma', type=float, default=2.0)
    parser.add_argument('--threshold', type=float, default=0.40)
    parser.add_argument('--auto_threshold', action='store_true',
                       help='Automatically find best threshold on validation set')
    
    # 其他
    parser.add_argument('--save_dir', type=str, default='checkpoints_baseline_clean')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*80)
    print("Clean Baseline Training")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Features: {args.features_root}")
    print(f"  Matrix Keys: {args.matrix_keys}")
    print(f"  Model: {args.hidden_dim}d, {args.num_layers} layers")
    print(f"  Sampling Weight: {args.weight_per_positive}")
    print(f"  Focal Gamma: {args.base_gamma}")
    print(f"  Threshold: {'Auto' if args.auto_threshold else args.threshold}")
    print(f"  Device: {args.device}")
    
    # 加载数据
    print(f"\n{'='*80}\nLoading Data\n{'='*80}")
    
    labels_df = pd.read_csv(args.labels_csv, encoding='utf-8')
    
    # 收集所有NPZ文件
    npz_files = []
    for _, row in labels_df.iterrows():
        features_dir = Path(args.features_root) / row['features_dir_path']
        if features_dir.exists():
            npz_files.extend(list(features_dir.glob('*.npz')))
    
    print(f"Found {len(npz_files)} NPZ files")
    
    # 创建完整数据集
    full_dataset = FilteredMultiLabelDataset(
        npz_paths=[str(f) for f in npz_files],
        labels_df=labels_df,
        matrix_keys=args.matrix_keys,
        min_samples=args.min_channel_samples,
        fusion_method='average',
        topk_ratio=0.2
    )
    
    channel_names = full_dataset.all_channels
    num_channels = len(channel_names)
    
    print(f"\nDataset: {len(full_dataset)} samples, {num_channels} channels")
    
    # 数据分割
    from datasets_multilabel import make_patient_splits
    
    patient_to_files = {}
    for npz_file in npz_files:
        # 提取患者名（假设在路径中）
        parts = Path(npz_file).parts
        if len(parts) >= 3:
            patient_name = parts[-3]  # 通常是 test_name/patient_name/features_dir
        else:
            patient_name = 'unknown'
        
        if patient_name not in patient_to_files:
            patient_to_files[patient_name] = []
        patient_to_files[patient_name].append(str(npz_file))
    
    splits = make_patient_splits(patient_to_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # 创建数据集
    train_dataset = FilteredMultiLabelDataset(
        npz_paths=splits['train'],
        labels_df=labels_df,
        matrix_keys=args.matrix_keys,
        min_samples=args.min_channel_samples,
        fusion_method='average'
    )
    
    val_dataset = FilteredMultiLabelDataset(
        npz_paths=splits['val'] if splits['val'] else splits['test'][:len(splits['test'])//2],
        labels_df=labels_df,
        matrix_keys=args.matrix_keys,
        all_channels=channel_names,
        fusion_method='average'
    )
    
    test_dataset = FilteredMultiLabelDataset(
        npz_paths=splits['test'],
        labels_df=labels_df,
        matrix_keys=args.matrix_keys,
        all_channels=channel_names,
        fusion_method='average'
    )
    
    print(f"\nSplit: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # 创建加载器（保守的采样权重）
    from datasets_multilabel import collate_graph_multilabel
    
    train_sampler = create_weighted_sampler(
        train_dataset,
        sampler_type='basic',  # 使用基础采样
        weight_per_positive=args.weight_per_positive
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_graph_multilabel,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_multilabel,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_multilabel,
        num_workers=0
    )
    
    # 创建模型
    print(f"\n{'='*80}\nCreating Model\n{'='*80}")
    
    model = MultiLabelGNNClassifier(
        in_dim=2,
        hidden_dim=args.hidden_dim,
        num_channels=num_channels,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(args.device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # 损失函数（简单的Focal Loss）
    criterion = FocalLoss(gamma=args.base_gamma, alpha=0.25)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 学习率调度
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=args.warmup_epochs
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # 训练
    print(f"\n{'='*80}\nTraining\n{'='*80}")
    
    best_jaccard = 0.0
    best_threshold = args.threshold
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch in pbar:
            adj = batch['adj'].to(args.device)
            x = batch['x'].to(args.device)
            y = batch['y'].to(args.device)
            
            # 前向
            logits = model(x, adj)
            loss = criterion(logits, y)
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 学习率调度
        if epoch <= args.warmup_epochs:
            warmup_scheduler.step()
        
        # 验证
        val_metrics, val_probs, val_labels, val_preds = evaluate(
            model, val_loader, args.device, channel_names, args.threshold
        )
        
        # 自动阈值搜索（每10个epoch或最后）
        if args.auto_threshold and (epoch % 10 == 0 or epoch == args.epochs):
            print(f"\n  Searching for optimal threshold...")
            optimal_thresh, optimal_score = find_optimal_threshold(
                val_probs, val_labels, metric='jaccard'
            )
            print(f"  Found: threshold={optimal_thresh:.2f}, score={optimal_score:.4f}")
            
            # 使用最优阈值重新评估
            val_preds_opt = (val_probs > optimal_thresh).astype(np.float32)
            jaccard_opt = jaccard_score(val_labels, val_preds_opt, average='samples', zero_division=0)
            
            if jaccard_opt > val_metrics['jaccard_samples']:
                print(f"  Optimal threshold improved Jaccard: {val_metrics['jaccard_samples']:.4f} → {jaccard_opt:.4f}")
                best_threshold = optimal_thresh
                val_metrics['jaccard_samples'] = jaccard_opt
        
        if epoch > args.warmup_epochs:
            plateau_scheduler.step(val_metrics['jaccard_samples'])
        
        # 打印
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Jaccard: {val_metrics['jaccard_samples']:.4f}, mAP: {val_metrics['map']:.4f}")
        print(f"  Hamming: {val_metrics['hamming_loss']:.4f}")
        print(f"  Current Threshold: {best_threshold:.2f}")
        
        # 保存最佳
        if val_metrics['jaccard_samples'] > best_jaccard:
            best_jaccard = val_metrics['jaccard_samples']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_jaccard': best_jaccard,
                'best_threshold': best_threshold,
                'channel_names': channel_names,
                'args': args
            }, os.path.join(args.save_dir, 'best.pt'))
            print(f"  ✓ Saved best (Jaccard={best_jaccard:.4f}, Threshold={best_threshold:.2f})")
    
    # 最终测试
    print(f"\n{'='*80}\nFinal Test Evaluation\n{'='*80}")
    
    ckpt = torch.load(os.path.join(args.save_dir, 'best.pt'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    best_threshold = ckpt.get('best_threshold', args.threshold)
    
    print(f"Using threshold: {best_threshold:.2f}")
    
    test_metrics, test_probs, test_labels, test_preds = evaluate(
        model, test_loader, args.device, channel_names, best_threshold
    )
    
    print(f"\nTest Results (threshold={best_threshold:.2f}):")
    print(f"  Jaccard (Samples): {test_metrics['jaccard_samples']:.4f}")
    print(f"  Jaccard (Macro): {test_metrics['jaccard_macro']:.4f}")
    print(f"  mAP: {test_metrics['map']:.4f}")
    print(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    
    # Per-channel报告
    print(f"\n{'='*80}\nPer-Channel Results:\n{'='*80}")
    print(classification_report(
        test_labels, test_preds,
        target_names=channel_names,
        zero_division=0
    ))
    
    # 保存结果
    np.savez(
        os.path.join(args.save_dir, 'test_results.npz'),
        probs=test_probs,
        labels=test_labels,
        preds=test_preds,
        channel_names=channel_names,
        threshold=best_threshold
    )
    
    print(f"\n✓ Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()

