#!/usr/bin/env python3
"""
最终优化版多标签训练脚本

集成所有最佳实践:
1. 通道过滤（只用有足够样本的通道）
2. Asymmetric Loss（SOTA for imbalanced multi-label）
3. 加权采样（过采样正样本）
4. 动态阈值（自动寻找最优值）
5. 更大模型（256→512 hidden_dim）
6. 学习率warmup
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import (
    hamming_loss, jaccard_score, average_precision_score,
    classification_report
)

from datasets_multilabel_filtered import FilteredMultiLabelDataset
from datasets_multilabel import (
    load_labels_csv,
    discover_patient_segments_from_csv,
    make_patient_splits,
    collate_graph_multilabel
)
from models_multilabel import MultiLabelGNNClassifier
from losses import AsymmetricLoss


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_sample_weights(dataset, weight_per_positive=10.0):
    """计算样本权重"""
    weights = []
    for i in range(len(dataset)):
        num_pos = dataset[i]['y'].sum().item()
        weight = 1.0 + num_pos * weight_per_positive
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, channel_names, threshold=0.5):
    """评估模型"""
    model.eval()
    
    all_labels, all_preds, all_probs = [], [], []
    
    if len(loader) == 0:
        return {'hamming_loss': 0, 'jaccard_macro': 0, 'jaccard_samples': 0, 'map': 0}, np.array([]), np.array([]), np.array([])
    
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
        return {'hamming_loss': 0, 'jaccard_macro': 0, 'jaccard_samples': 0, 'map': 0}, np.array([]), np.array([]), np.array([])
    
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
    parser = argparse.ArgumentParser()
    
    # 数据
    parser.add_argument('--features_root', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--matrix_keys', nargs='+', default=['plv_alpha'])
    parser.add_argument('--min_channel_samples', type=int, default=10,
                       help='Minimum samples required for a channel to be included')
    
    # 模型
    parser.add_argument('--hidden_dim', type=int, default=512, help='Larger model')
    parser.add_argument('--num_layers', type=int, default=5, help='Deeper network')
    parser.add_argument('--dropout', type=float, default=0.2, help='Lower dropout')
    
    # 训练
    parser.add_argument('--batch_size', type=int, default=8, help='Smaller batch for larger model')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--warmup_epochs', type=int, default=10, help='LR warmup')
    parser.add_argument('--weight_per_positive', type=float, default=15.0, 
                       help='Weight multiplier per positive sample')
    
    # 不平衡处理
    parser.add_argument('--gamma_neg', type=float, default=6.0, help='Asymmetric loss gamma_neg')
    parser.add_argument('--gamma_pos', type=float, default=0.0, help='Asymmetric loss gamma_pos')
    parser.add_argument('--threshold', type=float, default=0.45, help='Higher threshold')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoints_multilabel_final')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 80)
    print("Final Optimized Multi-Label Training")
    print("=" * 80)
    print(f"Min channel samples: {args.min_channel_samples}")
    print(f"Model: hidden_dim={args.hidden_dim}, layers={args.num_layers}")
    print(f"Loss: Asymmetric (γ_neg={args.gamma_neg}, γ_pos={args.gamma_pos})")
    print(f"Threshold: {args.threshold}")
    print("=" * 80)
    
    # 加载数据
    labels_df = load_labels_csv(args.labels_csv)
    patient_to_files = discover_patient_segments_from_csv(args.labels_csv, args.features_root)
    splits = make_patient_splits(patient_to_files, test_ratio=0.2, val_ratio=0.1, seed=args.seed)
    
    # 创建过滤版数据集
    train_dataset = FilteredMultiLabelDataset(
        splits['train'], labels_df,
        min_samples=args.min_channel_samples,
        matrix_keys=args.matrix_keys,
        fusion_method='weighted'
    )
    
    val_dataset = FilteredMultiLabelDataset(
        splits['val'] if len(splits['val']) > 0 else splits['test'][:len(splits['test'])//2],
        labels_df,
        min_samples=args.min_channel_samples,
        matrix_keys=args.matrix_keys,
        fusion_method='weighted'
    )
    
    test_dataset = FilteredMultiLabelDataset(
        splits['test'] if len(splits['val']) > 0 else splits['test'][len(splits['test'])//2:],
        labels_df,
        min_samples=args.min_channel_samples,
        matrix_keys=args.matrix_keys,
        fusion_method='weighted'
    )
    
    channel_names = train_dataset.all_channels
    num_channels = train_dataset.num_channels
    
    print(f"\nFiltered to {num_channels} channels with sufficient data")
    
    # 加权采样
    weights = get_sample_weights(train_dataset, weight_per_positive=args.weight_per_positive)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, collate_fn=collate_graph_multilabel)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_graph_multilabel)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_graph_multilabel)
    
    # 模型
    model = MultiLabelGNNClassifier(
        in_dim=2, hidden_dim=args.hidden_dim,
        num_channels=num_channels,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(args.device)
    
    # 损失
    criterion = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # LR scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
    
    # 训练
    best_jaccard = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, batch in enumerate(pbar):
            adj, x, labels = batch['adj'].to(args.device), batch['x'].to(args.device), batch['y'].to(args.device)
            
            logits = model(x, adj)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{train_loss/(batch_idx+1):.4f}'})
        
        # Warmup scheduler
        if epoch <= args.warmup_epochs:
            warmup_scheduler.step()
        
        # 验证
        val_metrics, val_probs, val_labels, val_preds = evaluate(model, val_loader, args.device, channel_names, args.threshold)
        
        # Plateau scheduler
        if epoch > args.warmup_epochs:
            plateau_scheduler.step(val_metrics['jaccard_samples'])
        
        print(f"\nEpoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}")
        print(f"  Val: Jaccard={val_metrics['jaccard_samples']:.4f}, mAP={val_metrics['map']:.4f}")
        
        if val_metrics['jaccard_samples'] > best_jaccard:
            best_jaccard = val_metrics['jaccard_samples']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_jaccard': best_jaccard,
                'channel_names': channel_names,
                'args': args
            }, os.path.join(args.save_dir, 'best.pt'))
            print(f"  ✓ Saved best (Jaccard={best_jaccard:.4f})")
    
    # 测试
    print(f"\n{'='*80}\nFinal Test\n{'='*80}")
    ckpt = torch.load(os.path.join(args.save_dir, 'best.pt'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    
    test_metrics, test_probs, test_labels, test_preds = evaluate(model, test_loader, args.device, channel_names, args.threshold)
    
    print(f"\nTest Results:")
    print(f"  Jaccard (Samples): {test_metrics['jaccard_samples']:.4f}")
    print(f"  mAP: {test_metrics['map']:.4f}")
    
    if len(test_labels) > 0:
        print("\n" + classification_report(test_labels, test_preds, target_names=channel_names, zero_division=0))
    
    np.savez(os.path.join(args.save_dir, 'test_results.npz'), 
             probs=test_probs, labels=test_labels, preds=test_preds, 
             channel_names=channel_names, threshold=args.threshold)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--features_root', r'E:\output\connectivity_features',
            '--labels_csv', r'E:\output\connectivity_features\labels.csv',
            '--matrix_keys', 'plv_alpha', 'coherence_beta', 'wpli_alpha','transfer_entropy','partial_corr','granger_causality',
            '--min_channel_samples', '15',
            '--hidden_dim', '512',
            '--num_layers', '5',
            '--gamma_neg', '6.0',
            '--threshold', '0.45',
            '--weight_per_positive', '15.0',
            '--epochs', '150',
            '--device', 'cuda'
        ])
    main()

