#!/usr/bin/env python3
"""
终极优化版多标签训练 - 组合3完整实现

集成所有11个高级优化方法:
1. ChannelAdaptiveFocalLoss - 通道自适应损失
2. MultiLevelWeightedSampler - 三级加权采样
3. CurriculumLearning - 课程学习
4. Graph SMOTE - 合成少数类样本
5. Graph Mixup - 图混合增强
6. Contrastive Pretraining - 对比学习预训练
7. OHEM - 难例挖掘
8. Temporal Consistency - 时序一致性
9. ClassBalanced Sampling - 类平衡采样
10. Adaptive Loss Weights - 自适应权重
11. Channel Filtering - 通道过滤

预期效果: Jaccard 0.26 → 0.60-0.75
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from sklearn.metrics import hamming_loss, jaccard_score, average_precision_score, classification_report

# 导入所有模块
from datasets_multilabel_filtered import FilteredMultiLabelDataset
from datasets_multilabel import (
    load_labels_csv,
    discover_patient_segments_from_csv,
    make_patient_splits,
    collate_graph_multilabel
)
from models_multilabel import MultiLabelGNNClassifier
from losses_advanced import ChannelAdaptiveFocalLoss, CombinedAdvancedLoss
from samplers_advanced import ClassBalancedSampler, MultiLevelWeightedSampler
from augmentations_graph import GraphAugmentor, create_augmented_dataset
from curriculum_trainer import CurriculumTrainer
from contrastive_pretrain_multilabel import pretrain_contrastive


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, channel_names, threshold=0.5, active_channels=None):
    """
    评估模型
    
    Args:
        active_channels: 如果提供，只评估这些通道（课程学习用）
    """
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
        
        # 如果使用课程学习，只评估活跃通道
        if active_channels is not None:
            labels = labels[:, active_channels]
            probs = probs[:, active_channels]
            preds = preds[:, active_channels]
        
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
    parser = argparse.ArgumentParser(description='Ultimate Optimized Multi-Label Training')
    
    # 数据
    parser.add_argument('--features_root', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--matrix_keys', nargs='+', default=['plv_alpha'])
    parser.add_argument('--min_channel_samples', type=int, default=15)
    
    # 模型
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.15)
    
    # 训练
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--warmup_epochs', type=int, default=15)
    
    # 对比学习预训练
    parser.add_argument('--use_contrastive_pretrain', action='store_true',
                       help='Use contrastive pretraining first')
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    
    # 课程学习
    parser.add_argument('--use_curriculum', action='store_true',
                       help='Use curriculum learning')
    parser.add_argument('--curriculum_stages', nargs=3, type=int, default=[40, 80, 120],
                       help='Epoch boundaries for curriculum stages')
    
    # 数据增强
    parser.add_argument('--use_smote', action='store_true',
                       help='Use Graph SMOTE for rare channels')
    parser.add_argument('--smote_per_channel', type=int, default=50)
    parser.add_argument('--use_mixup', action='store_true',
                       help='Use Graph Mixup')
    
    # 采样策略
    parser.add_argument('--sampler_type', type=str, default='multilevel',
                       choices=['basic', 'class_balanced', 'multilevel'],
                       help='Sampling strategy')
    parser.add_argument('--weight_per_positive', type=float, default=20.0)
    
    # 损失函数
    parser.add_argument('--loss_type', type=str, default='channel_adaptive',
                       choices=['channel_adaptive', 'combined_advanced'],
                       help='Loss function type')
    parser.add_argument('--base_gamma', type=float, default=3.0)
    parser.add_argument('--threshold', type=float, default=0.48)
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoints_ultimate')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 80)
    print("ULTIMATE OPTIMIZED Multi-Label Training (Research-Grade)")
    print("=" * 80)
    print(f"Optimizations enabled:")
    print(f"  [1] Channel Filtering: min_samples={args.min_channel_samples}")
    print(f"  [2] Model Size: hidden={args.hidden_dim}, layers={args.num_layers}")
    print(f"  [3] Contrastive Pretrain: {args.use_contrastive_pretrain}")
    print(f"  [4] Curriculum Learning: {args.use_curriculum}")
    print(f"  [5] Graph SMOTE: {args.use_smote}")
    print(f"  [6] Graph Mixup: {args.use_mixup}")
    print(f"  [7] Sampler: {args.sampler_type}")
    print(f"  [8] Loss: {args.loss_type}")
    print(f"  [9] Threshold: {args.threshold}")
    print("=" * 80)
    
    # 加载数据
    print("\nLoading data...")
    labels_df = load_labels_csv(args.labels_csv)
    patient_to_files = discover_patient_segments_from_csv(args.labels_csv, args.features_root)
    splits = make_patient_splits(patient_to_files, test_ratio=0.2, val_ratio=0.1, seed=args.seed)
    
    # 创建过滤版数据集
    print("\nCreating filtered datasets...")
    train_dataset_base = FilteredMultiLabelDataset(
        splits['train'], labels_df,
        min_samples=args.min_channel_samples,
        matrix_keys=args.matrix_keys,
        fusion_method='weighted'
    )
    
    # Graph SMOTE数据增强
    if args.use_smote:
        print("\nApplying Graph SMOTE...")
        train_dataset = create_augmented_dataset(
            train_dataset_base,
            rare_channel_threshold=args.min_channel_samples * 2,
            n_synthetic_per_channel=args.smote_per_channel
        )
    else:
        train_dataset = train_dataset_base
    
    # 验证和测试集
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
    
    channel_names = train_dataset_base.all_channels
    num_channels = train_dataset_base.num_channels
    
    # 统计通道频率
    channel_freqs = np.zeros(num_channels)
    for i in range(len(train_dataset_base)):
        y = train_dataset_base[i]['y'].numpy()
        channel_freqs += y
    
    print(f"\nDataset info:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Channels: {num_channels}")
    
    # 创建采样器
    print(f"\nCreating {args.sampler_type} sampler...")
    if args.sampler_type == 'class_balanced':
        sampler = ClassBalancedSampler(train_dataset_base, beta=0.9999)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 sampler=sampler, num_workers=0, collate_fn=collate_graph_multilabel)
    elif args.sampler_type == 'multilevel':
        sampler = MultiLevelWeightedSampler(
            train_dataset_base,
            weight_positive=args.weight_per_positive,
            weight_rare=8.0,
            weight_hard=5.0
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 sampler=sampler, num_workers=0, collate_fn=collate_graph_multilabel)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=0, collate_fn=collate_graph_multilabel)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate_graph_multilabel)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_graph_multilabel)
    
    # 创建模型
    print(f"\nCreating model (hidden={args.hidden_dim}, layers={args.num_layers})...")
    model = MultiLabelGNNClassifier(
        in_dim=2,
        hidden_dim=args.hidden_dim,
        num_channels=num_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batch_norm=True
    ).to(args.device)
    
    # 对比学习预训练（可选）
    if args.use_contrastive_pretrain:
        print("\n" + "="*80)
        print("Phase 1: Contrastive Pretraining (Unsupervised)")
        print("="*80)
        
        model = pretrain_contrastive(
            model, train_loader,
            epochs=args.pretrain_epochs,
            device=args.device,
            save_path=os.path.join(args.save_dir, 'contrastive_pretrained.pt')
        )
    
    # 创建损失函数
    print(f"\nCreating {args.loss_type} loss...")
    if args.loss_type == 'channel_adaptive':
        criterion = ChannelAdaptiveFocalLoss(
            channel_freqs,
            base_gamma=args.base_gamma,
            base_alpha=0.25
        )
    else:  # combined_advanced
        criterion = CombinedAdvancedLoss(
            channel_freqs,
            len(train_dataset_base),
            focal_weight=0.5,
            ohem_weight=0.3,
            consistency_weight=0.2
        )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # 课程学习（可选）
    curriculum_trainer = None
    if args.use_curriculum:
        print("\nInitializing Curriculum Learning...")
        curriculum_trainer = CurriculumTrainer(
            model, criterion, optimizer,
            channel_names, channel_freqs,
            stage_epochs=args.curriculum_stages,
            device=args.device
        )
    
    # 图增强器
    augmentor = GraphAugmentor(
        edge_dropout_rate=0.25,
        node_dropout_rate=0.1,
        feature_noise_std=0.08,
        use_mixup=args.use_mixup,
        mixup_alpha=0.3
    ) if args.use_mixup else None
    
    # 训练循环
    print(f"\n{'='*80}")
    if args.use_contrastive_pretrain:
        print("Phase 2: Supervised Fine-tuning")
    else:
        print("Supervised Training")
    print(f"{'='*80}\n")
    
    best_jaccard = 0.0
    mixup_batches = []  # 用于mixup的batch缓存
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        # 课程学习阶段信息
        if curriculum_trainer:
            curriculum_trainer.print_stage_info(epoch)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # 数据增强
            if augmentor and args.use_mixup:
                # 缓存batch用于mixup
                mixup_batches.append(batch)
                if len(mixup_batches) > 10:
                    mixup_batches.pop(0)
                
                # 随机选择一个batch进行mixup
                if len(mixup_batches) > 1 and np.random.rand() < 0.5:
                    mixup_batch = mixup_batches[np.random.randint(0, len(mixup_batches))]
                    adj, x, y = augmentor.augment_batch(batch, mixup_batch)
                else:
                    adj, x, y = augmentor.augment_batch(batch)
            else:
                adj, x, y = batch['adj'], batch['x'], batch['y']
            
            adj = adj.to(args.device)
            x = x.to(args.device)
            y = y.to(args.device)
            
            # 前向传播
            logits = model(x, adj)
            
            # 计算loss（课程学习或正常）
            if curriculum_trainer and args.use_curriculum:
                loss = curriculum_trainer.train_step(
                    {'adj': adj, 'x': x, 'y': y},
                    epoch
                )
            else:
                if isinstance(criterion, CombinedAdvancedLoss):
                    # 需要sample_indices
                    loss, loss_dict = criterion(logits, y, sample_indices=None)
                else:
                    loss = criterion(logits, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            pbar.set_postfix({'loss': f'{loss.item() if isinstance(loss, torch.Tensor) else loss:.4f}'})
        
        # 学习率调度
        if epoch <= args.warmup_epochs:
            warmup_scheduler.step()
        
        # 验证（课程学习时只评估当前阶段的通道）
        if curriculum_trainer is not None:
            active_channels = curriculum_trainer.get_active_channels(epoch)
            val_metrics, val_probs, val_labels, val_preds = evaluate(
                model, val_loader, args.device, channel_names, args.threshold, active_channels
            )
        else:
            val_metrics, val_probs, val_labels, val_preds = evaluate(
                model, val_loader, args.device, channel_names, args.threshold
            )
        
        if epoch > args.warmup_epochs:
            plateau_scheduler.step(val_metrics['jaccard_samples'])
        
        # 打印
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Jaccard: {val_metrics['jaccard_samples']:.4f}, mAP: {val_metrics['map']:.4f}")
        print(f"  Hamming: {val_metrics['hamming_loss']:.4f}")
        
        # 保存最佳模型
        if val_metrics['jaccard_samples'] > best_jaccard:
            best_jaccard = val_metrics['jaccard_samples']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_jaccard': best_jaccard,
                'channel_names': channel_names,
                'args': args
            }, os.path.join(args.save_dir, 'best.pt'))
            print(f"  ✓ Saved best (Jaccard={best_jaccard:.4f})")
    
    # 最终测试（评估所有通道，不使用课程学习）
    print(f"\n{'='*80}\nFinal Test Evaluation\n{'='*80}")
    
    ckpt = torch.load(os.path.join(args.save_dir, 'best.pt'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # 测试时评估所有通道
    test_metrics, test_probs, test_labels, test_preds = evaluate(
        model, test_loader, args.device, channel_names, args.threshold, active_channels=None
    )
    
    print(f"\nTest Results (threshold={args.threshold}):")
    print(f"  Jaccard (Samples): {test_metrics['jaccard_samples']:.4f}")
    print(f"  Jaccard (Macro): {test_metrics['jaccard_macro']:.4f}")
    print(f"  mAP: {test_metrics['map']:.4f}")
    print(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    
    if len(test_labels) > 0:
        print("\n" + "="*80)
        print("Per-Channel Results:")
        print("="*80)
        print(classification_report(test_labels, test_preds, target_names=channel_names, zero_division=0))
    
    # 保存结果
    np.savez(
        os.path.join(args.save_dir, 'test_results.npz'),
        probs=test_probs,
        labels=test_labels,
        preds=test_preds,
        channel_names=channel_names,
        threshold=args.threshold
    )
    print(f"\n✓ Results saved to {args.save_dir}/")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--features_root', r'E:\output\connectivity_features_v2',
            '--labels_csv', r'E:\output\connectivity_features_v2\labels.csv',
            '--matrix_keys', 'wpli_alpha', 'transfer_entropy', 'partial_corr', 'granger_causality',
            '--min_channel_samples', '15',
            '--hidden_dim', '512',
            '--num_layers', '6',
            '--dropout', '0.15',
            '--use_contrastive_pretrain',
            '--pretrain_epochs', '50',
            # '--use_curriculum',
            # '--curriculum_stages', '40', '80', '120',
            '--use_smote',
            '--smote_per_channel', '50',
            '--use_mixup',
            '--sampler_type', 'multilevel',
            '--loss_type', 'channel_adaptive',
            '--base_gamma', '3.0',
            '--threshold', '0.48',
            '--weight_per_positive', '20.0',
            '--epochs', '150',
            '--device', 'cuda'
        ])
    main()

