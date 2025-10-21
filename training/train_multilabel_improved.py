#!/usr/bin/env python3
"""
改进的多标签分类训练脚本 - 专门针对不平衡数据优化

包含优化:
1. Focal Loss - 聚焦难分类样本
2. 加权采样 - 过采样正样本
3. 动态阈值 - 根据数据调整
4. 数据增强 - 增加正样本多样性
5. 两阶段训练 - 先学习有正样本的数据
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import (
    hamming_loss, jaccard_score, average_precision_score,
    classification_report
)

from datasets_multilabel import (
    MultiLabelConnectivityDataset,
    load_labels_csv,
    discover_patient_segments_from_csv,
    make_patient_splits,
    collate_graph_multilabel
)
from models_multilabel import MultiLabelGNNClassifier
from losses import FocalLoss, AsymmetricLoss, CombinedLoss


def seed_everything(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_sample_weights(dataset):
    """
    计算样本权重，用于加权采样
    有更多正类的样本权重更高
    """
    weights = []
    for i in range(len(dataset)):
        sample = dataset[i]
        num_pos = sample['y'].sum().item()
        
        # 权重策略：基础权重1.0，每个正类增加5.0
        # 这样有3个正类的样本权重是16.0，没有正类的是1.0
        weight = 1.0 + num_pos * 5.0
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def find_best_threshold(probs, labels):
    """
    在验证集上寻找最佳分类阈值
    
    Args:
        probs: [N, num_channels] 预测概率
        labels: [N, num_channels] 真实标签
    
    Returns:
        best_threshold: float
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_jaccard = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        preds = (probs > thresh).astype(np.float32)
        try:
            jaccard = jaccard_score(labels, preds, average='samples', zero_division=0)
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_thresh = thresh
        except:
            continue
    
    return best_thresh


@torch.no_grad()
def evaluate(model, loader, device, channel_names, threshold=0.5):
    """评估模型"""
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    if len(loader) == 0:
        print("Warning: Empty data loader")
        return {
            'hamming_loss': 0.0,
            'jaccard_macro': 0.0,
            'jaccard_samples': 0.0,
            'map': 0.0
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
    
    if not all_labels:
        return {
            'hamming_loss': 0.0,
            'jaccard_macro': 0.0,
            'jaccard_samples': 0.0,
            'map': 0.0
        }, np.array([]), np.array([]), np.array([])
    
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    
    # 计算指标
    metrics = {}
    metrics['hamming_loss'] = hamming_loss(all_labels, all_preds)
    metrics['jaccard_macro'] = jaccard_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['jaccard_samples'] = jaccard_score(all_labels, all_preds, average='samples', zero_division=0)
    metrics['map'] = average_precision_score(all_labels, all_probs, average='macro')
    
    return metrics, all_probs, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description='Improved Multi-Label Training for Imbalanced Data')
    
    # 数据参数
    parser.add_argument('--features_root', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--matrix_keys', nargs='+', default=['plv_alpha'])
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=256, help='Increased from 128')
    parser.add_argument('--num_layers', type=int, default=4, help='Increased from 3')
    parser.add_argument('--dropout', type=float, default=0.3, help='Reduced from 0.5')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='Smaller batch for imbalanced data')
    parser.add_argument('--epochs', type=int, default=150, help='More epochs needed')
    parser.add_argument('--lr', type=float, default=0.0005, help='Lower LR for stability')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # 不平衡处理策略
    parser.add_argument('--loss_type', type=str, default='focal',
                       choices=['bce', 'focal', 'asymmetric', 'combined'],
                       help='Loss function type')
    parser.add_argument('--focal_gamma', type=float, default=2.5, help='Focal loss gamma')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Focal loss alpha')
    parser.add_argument('--use_weighted_sampler', action='store_true',
                       help='Use weighted sampling to oversample positive samples')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Lower threshold for imbalanced data')
    parser.add_argument('--find_best_threshold', action='store_true',
                       help='Find best threshold on validation set')
    
    # 两阶段训练
    parser.add_argument('--two_stage', action='store_true',
                       help='Two-stage training: first on positive samples only')
    parser.add_argument('--stage1_epochs', type=int, default=30,
                       help='Epochs for stage 1')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoints_multilabel_improved')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 80)
    print("Improved Multi-Label Training for Imbalanced Data")
    print("=" * 80)
    print(f"Loss type: {args.loss_type}")
    print(f"Weighted sampler: {args.use_weighted_sampler}")
    print(f"Two-stage training: {args.two_stage}")
    print(f"Threshold: {args.threshold}")
    print("=" * 80)
    
    # 加载数据
    print("\nLoading data...")
    labels_df = load_labels_csv(args.labels_csv)
    patient_to_files = discover_patient_segments_from_csv(args.labels_csv, args.features_root)
    splits = make_patient_splits(patient_to_files, test_ratio=0.2, val_ratio=0.1, seed=args.seed)
    
    # 创建数据集
    print("\nCreating datasets...")
    train_dataset = MultiLabelConnectivityDataset(
        splits['train'], labels_df,
        matrix_keys=args.matrix_keys,
        fusion_method='weighted'
    )
    
    val_dataset = MultiLabelConnectivityDataset(
        splits['val'] if len(splits['val']) > 0 else splits['test'][:len(splits['test'])//2],
        labels_df,
        all_channels=train_dataset.all_channels,
        matrix_keys=args.matrix_keys,
        fusion_method='weighted'
    )
    
    test_dataset = MultiLabelConnectivityDataset(
        splits['test'] if len(splits['val']) > 0 else splits['test'][len(splits['test'])//2:],
        labels_df,
        all_channels=train_dataset.all_channels,
        matrix_keys=args.matrix_keys,
        fusion_method='weighted'
    )
    
    channel_names = train_dataset.all_channels
    num_channels = train_dataset.num_channels
    
    print(f"\nDataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Channels ({num_channels}): {', '.join(channel_names)}")
    
    # 创建DataLoader（可选加权采样）
    if args.use_weighted_sampler:
        print("\n✓ Using weighted sampler to oversample positive samples")
        weights = get_sample_weights(train_dataset)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=sampler, num_workers=0,
            collate_fn=collate_graph_multilabel
        )
    else:
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
    
    # 创建模型（更大的网络）
    print(f"\nCreating model (hidden_dim={args.hidden_dim}, layers={args.num_layers})...")
    model = MultiLabelGNNClassifier(
        in_dim=2,
        hidden_dim=args.hidden_dim,
        num_channels=num_channels,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(args.device)
    
    # 选择损失函数
    print(f"\nUsing {args.loss_type} loss...")
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"  Focal parameters: alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    elif args.loss_type == 'asymmetric':
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1)
        print(f"  Asymmetric parameters: gamma_neg=4, gamma_pos=1")
    elif args.loss_type == 'combined':
        criterion = CombinedLoss(focal_weight=0.7, dice_weight=0.3)
        print(f"  Combined: 70% Focal + 30% Dice")
    else:
        # BCE with pos_weight
        pos_weight = train_dataset.get_pos_weight().to(args.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  BCE with pos_weight, range=[{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器（更激进）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # 两阶段训练
    if args.two_stage:
        print(f"\n{'='*80}")
        print("Stage 1: Training on positive samples only")
        print(f"{'='*80}\n")
        
        # 找出所有有正类的样本
        positive_indices = [i for i in range(len(train_dataset)) 
                           if train_dataset[i]['y'].sum() > 0]
        
        print(f"Positive samples: {len(positive_indices)}/{len(train_dataset)}")
        
        # 创建子集
        from torch.utils.data import Subset
        positive_dataset = Subset(train_dataset, positive_indices)
        positive_loader = DataLoader(
            positive_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_graph_multilabel
        )
        
        # Stage 1 训练
        for epoch in range(1, args.stage1_epochs + 1):
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(positive_loader, desc=f'Stage1 Epoch {epoch}/{args.stage1_epochs}')
            for batch_idx, batch in enumerate(pbar):
                adj = batch['adj'].to(args.device)
                x = batch['x'].to(args.device)
                labels = batch['y'].to(args.device)
                
                logits = model(x, adj)
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = train_loss / len(positive_loader)
            print(f"  Stage1 Epoch {epoch}: loss={avg_loss:.4f}")
        
        print(f"\n✓ Stage 1 completed\n")
    
    # Stage 2 或正常训练
    print(f"\n{'='*80}")
    if args.two_stage:
        print(f"Stage 2: Fine-tuning on all data")
    else:
        print("Training on all data")
    print(f"{'='*80}\n")
    
    best_jaccard = 0.0
    best_threshold = args.threshold
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, batch in enumerate(pbar):
            adj = batch['adj'].to(args.device)
            x = batch['x'].to(args.device)
            labels = batch['y'].to(args.device)
            
            logits = model(x, adj)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{train_loss / (batch_idx + 1):.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        val_metrics, val_probs, val_labels, val_preds = evaluate(
            model, val_loader, args.device,
            channel_names, threshold=best_threshold
        )
        
        # 动态寻找最佳阈值（每10个epoch或当性能提升时）
        if args.find_best_threshold and epoch % 10 == 0 and len(val_probs) > 0:
            new_threshold = find_best_threshold(val_probs, val_labels)
            print(f"\n  Found best threshold: {new_threshold:.3f} (was {best_threshold:.3f})")
            best_threshold = new_threshold
            
            # 用新阈值重新评估
            val_preds = (val_probs > best_threshold).astype(np.float32)
            val_metrics['jaccard_samples'] = jaccard_score(
                val_labels, val_preds, average='samples', zero_division=0
            )
        
        # 学习率调度
        scheduler.step(val_metrics['jaccard_samples'])
        
        # 打印总结
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Hamming: {val_metrics['hamming_loss']:.4f}")
        print(f"  Val Jaccard (Macro): {val_metrics['jaccard_macro']:.4f}")
        print(f"  Val Jaccard (Samples): {val_metrics['jaccard_samples']:.4f}")
        print(f"  Val mAP: {val_metrics['map']:.4f}")
        print(f"  Threshold: {best_threshold:.3f}")
        
        # 保存最佳模型
        if val_metrics['jaccard_samples'] > best_jaccard:
            best_jaccard = val_metrics['jaccard_samples']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_jaccard': best_jaccard,
                'best_threshold': best_threshold,
                'channel_names': channel_names,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best.pt'))
            print(f"  ✓ Saved best model (Jaccard={best_jaccard:.4f})")
    
    # 最终测试
    print(f"\n{'='*80}")
    print("Final Test Evaluation")
    print(f"{'='*80}")
    
    checkpoint = torch.load(os.path.join(args.save_dir, 'best.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_threshold = checkpoint.get('best_threshold', args.threshold)
    
    test_metrics, test_probs, test_labels, test_preds = evaluate(
        model, test_loader, args.device,
        channel_names, threshold=best_threshold
    )
    
    print(f"\nTest Results (threshold={best_threshold:.3f}):")
    print(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    print(f"  Jaccard (Macro): {test_metrics['jaccard_macro']:.4f}")
    print(f"  Jaccard (Samples): {test_metrics['jaccard_samples']:.4f}")
    print(f"  mAP: {test_metrics['map']:.4f}")
    
    # 详细报告
    if len(test_labels) > 0:
        print("\n" + "=" * 80)
        print("Per-Channel Test Results:")
        print("=" * 80)
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
    print(f"\n✓ Results saved to {args.save_dir}/test_results.npz")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--features_root', r'E:\output\connectivity_features',
            '--labels_csv', r'E:\output\connectivity_features\labels.csv',
            '--matrix_keys', 'coherence_beta', 'plv_alpha', 'coherence_delta', 'wpli_gamma','phase_angle_beta','icoh_alpha','transfer_entropy','partial_corr','coherence_theta','plv_beta','phase_angle_gamma',
            '--loss_type', 'focal',
            '--use_weighted_sampler',
            '--find_best_threshold',
            '--two_stage',
            '--batch_size', '16',
            '--epochs', '100',
            '--device', 'cuda'
        ])
    main()

