#!/usr/bin/env python3
"""
多标签分类训练脚本 - 通道级别异常检测
"""

import os
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    hamming_loss, jaccard_score, average_precision_score,
    classification_report, multilabel_confusion_matrix
)


from datasets_multilabel import (
    MultiLabelConnectivityDataset,
    load_labels_csv,
    discover_patient_segments_from_csv,
    make_patient_splits,
    collate_graph_multilabel
)
from models_multilabel import MultiLabelGNNClassifier, MultiLabelGNNWithAttention


def seed_everything(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, channel_names, threshold=0.5):
    """评估模型"""
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    # 检查loader是否为空
    if len(loader) == 0:
        print("Warning: Empty data loader, returning zero metrics")
        return {
            'hamming_loss': 0.0,
            'jaccard_macro': 0.0,
            'jaccard_samples': 0.0,
            'map': 0.0
        }, np.array([]), np.array([]), np.array([])
    
    for batch in tqdm(loader, desc='Evaluating'):
        adj = batch['adj'].to(device)
        x = batch['x'].to(device)
        labels = batch['y'].cpu().numpy()  # [B, num_channels]
        
        logits = model(x, adj)
        probs = torch.sigmoid(logits).cpu().numpy()  # [B, num_channels]
        preds = (probs > threshold).astype(np.float32)
        
        all_labels.append(labels)
        all_preds.append(preds)
        all_probs.append(probs)
    
    # 检查是否有数据
    if not all_labels:
        print("Warning: No samples evaluated, returning zero metrics")
        return {
            'hamming_loss': 0.0,
            'jaccard_macro': 0.0,
            'jaccard_samples': 0.0,
            'map': 0.0
        }, np.array([]), np.array([]), np.array([])
    
    # 合并所有批次
    all_labels = np.vstack(all_labels)  # [N, num_channels]
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    
    # 计算指标
    metrics = {}
    
    # 1. Hamming Loss (预测错误的标签比例)
    metrics['hamming_loss'] = hamming_loss(all_labels, all_preds)
    
    # 2. Jaccard Score (交并比)
    metrics['jaccard_macro'] = jaccard_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['jaccard_samples'] = jaccard_score(all_labels, all_preds, average='samples', zero_division=0)
    
    # 3. Average Precision (AP)
    metrics['map'] = average_precision_score(all_labels, all_probs, average='macro')
    
    # 4. 每个通道的指标
    print("\n" + "="*60)
    print("Per-Channel Classification Report:")
    print("="*60)
    print(classification_report(
        all_labels, all_preds,
        target_names=channel_names,
        zero_division=0
    ))
    
    return metrics, all_probs, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description='Multi-Label Channel Classification Training')
    
    # 数据参数
    parser.add_argument('--features_root', type=str, required=True,
                       help='Root directory containing connectivity features')
    parser.add_argument('--labels_csv', type=str, required=True,
                       help='CSV file with patient and channel combination information')
    parser.add_argument('--matrix_keys', nargs='+', default=['plv_alpha'],
                       help='List of matrix keys to use')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='basic', choices=['basic', 'attention'],
                       help='Model type: basic or attention')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads (for attention model)')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--use_pos_weight', action='store_true',
                       help='Use positive class weights for imbalanced data')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--save_dir', type=str, default='checkpoints_multilabel',
                       help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*60)
    print("Multi-Label Channel Classification Training")
    print("="*60)
    print(f"Features root: {args.features_root}")
    print(f"Labels CSV: {args.labels_csv}")
    print(f"Matrix keys: {args.matrix_keys}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # 加载数据
    print("\nLoading data...")
    labels_df = load_labels_csv(args.labels_csv)
    patient_to_files = discover_patient_segments_from_csv(args.labels_csv, args.features_root)
    splits = make_patient_splits(patient_to_files, test_ratio=0.2, val_ratio=0.1, seed=args.seed)
    
    # 检查数据分割
    print(f"\nData split details:")
    print(f"  Train files: {len(splits['train'])}")
    print(f"  Val files: {len(splits['val'])}")
    print(f"  Test files: {len(splits['test'])}")
    
    if len(splits['val']) == 0:
        print("\n⚠ WARNING: Validation set is empty!")
        print("  This usually happens when:")
        print("  1. Total patients < 10 (need at least 10 for proper split)")
        print("  2. Some patients have no valid NPZ files")
        print("  Adjusting split ratios...")
        
        # 如果数据太少，使用更小的验证集比例
        total_files = len(splits['train']) + len(splits['val']) + len(splits['test'])
        if total_files < 50:
            print(f"  Total files ({total_files}) < 50, using train/test split only")
            # 重新分割：80% train, 20% test, 0% val

            splits = make_patient_splits(patient_to_files, test_ratio=0.2, val_ratio=0.0, seed=args.seed)
    
    # 创建数据集
    print("\nCreating datasets...")
    train_dataset = MultiLabelConnectivityDataset(
        splits['train'], labels_df,
        matrix_keys=args.matrix_keys,
        fusion_method='weighted'
    )
    
    # 如果没有验证集，使用测试集的一部分作为验证集
    if len(splits['val']) > 0:
        val_dataset = MultiLabelConnectivityDataset(
            splits['val'], labels_df,
            all_channels=train_dataset.all_channels,
            matrix_keys=args.matrix_keys,
            fusion_method='weighted'
        )
    else:
        print("  Using a portion of test set as validation set")
        # 使用测试集的前一半作为验证集
        mid = len(splits['test']) // 2
        val_files = splits['test'][:mid]
        test_files = splits['test'][mid:]
        
        val_dataset = MultiLabelConnectivityDataset(
            val_files, labels_df,
            all_channels=train_dataset.all_channels,
            matrix_keys=args.matrix_keys,
            fusion_method='weighted'
        )
        
        # 更新splits
        splits['val'] = val_files
        splits['test'] = test_files
    
    test_dataset = MultiLabelConnectivityDataset(
        splits['test'], labels_df,
        all_channels=train_dataset.all_channels,
        matrix_keys=args.matrix_keys,
        fusion_method='weighted'
    )
    
    channel_names = train_dataset.all_channels
    num_channels = train_dataset.num_channels
    
    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"\nNumber of channels: {num_channels}")
    print(f"Channel names: {', '.join(channel_names)}")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_graph_multilabel
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_graph_multilabel
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_graph_multilabel
    )
    
    # 创建模型
    print(f"\nCreating {args.model_type} model...")
    if args.model_type == 'basic':
        model = MultiLabelGNNClassifier(
            in_dim=2,
            hidden_dim=args.hidden_dim,
            num_channels=num_channels,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:  # attention
        model = MultiLabelGNNWithAttention(
            in_dim=2,
            hidden_dim=args.hidden_dim,
            num_channels=num_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_heads=args.num_heads
        )
    
    model = model.to(args.device)
    
    # 损失函数
    if args.use_pos_weight:
        pos_weight = train_dataset.get_pos_weight().to(args.device)
        print(f"\nUsing positive class weights")
        print(f"  Weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # 训练循环
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    best_jaccard = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, batch in enumerate(pbar):
            adj = batch['adj'].to(args.device)
            x = batch['x'].to(args.device)
            labels = batch['y'].to(args.device)  # [B, num_channels]
            
            # 前向传播
            logits = model(x, adj)
            loss = criterion(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{train_loss / (batch_idx + 1):.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        val_metrics, _, _, _ = evaluate(
            model, val_loader, args.device,
            channel_names, threshold=args.threshold
        )
        
        # 学习率调度
        scheduler.step(val_metrics['jaccard_samples'])
        
        # 打印总结
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Hamming Loss: {val_metrics['hamming_loss']:.4f}")
        print(f"  Val Jaccard (Macro): {val_metrics['jaccard_macro']:.4f}")
        print(f"  Val Jaccard (Samples): {val_metrics['jaccard_samples']:.4f}")
        print(f"  Val mAP: {val_metrics['map']:.4f}")
        
        # 保存最佳模型
        if val_metrics['jaccard_samples'] > best_jaccard:
            best_jaccard = val_metrics['jaccard_samples']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_jaccard': best_jaccard,
                'channel_names': channel_names,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best.pt'))
            print(f"  ✓ Saved best model (Jaccard={best_jaccard:.4f})")
        
        # 保存最后一个模型
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'channel_names': channel_names,
            'args': args
        }
        torch.save(checkpoint, os.path.join(args.save_dir, 'last.pt'))
    
    # 最终测试
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.save_dir, 'best.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_probs, test_labels, test_preds = evaluate(
        model, test_loader, args.device,
        channel_names, threshold=args.threshold
    )
    
    print(f"\nTest Results:")
    print(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    print(f"  Jaccard (Macro): {test_metrics['jaccard_macro']:.4f}")
    print(f"  Jaccard (Samples): {test_metrics['jaccard_samples']:.4f}")
    print(f"  mAP: {test_metrics['map']:.4f}")
    
    # 保存预测结果
    np.savez(
        os.path.join(args.save_dir, 'test_results.npz'),
        probs=test_probs,
        labels=test_labels,
        preds=test_preds,
        channel_names=channel_names
    )
    print(f"\n✓ Saved test results to {args.save_dir}/test_results.npz")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        # 默认参数用于测试
        sys.argv.extend([
            '--features_root', r'E:\output\connectivity_features',
            '--labels_csv', r'E:\output\connectivity_features\labels.csv',
            '--matrix_keys', 'plv_alpha', 'coherence_alpha', 'wpli_alpha',
            '--model_type', 'basic',
            '--batch_size', '16',
            '--epochs', '50',
            '--device', 'cuda'
        ])
    main()

