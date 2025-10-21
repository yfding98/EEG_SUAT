#!/usr/bin/env python3
"""
快速特征选择 - 简化版

只分析矩阵级别的重要性，不展平为细粒度特征
速度更快，适合快速迭代
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import json

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def extract_matrix_level_features(
    npz_files: List[str],
    labels_df: pd.DataFrame,
    target_channel: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    提取矩阵级别的特征（每个矩阵提取统计特征）
    
    对每个矩阵计算:
    - mean, std, min, max
    - median, 25%ile, 75%ile
    - 稀疏度
    """
    from datasets_multilabel import MultiLabelConnectivityDataset, discover_all_channels
    
    all_channels = discover_all_channels(labels_df)
    
    if target_channel not in all_channels:
        print(f"Warning: Target channel '{target_channel}' not found")
        return None, None, None
    
    # 创建数据集
    temp_dataset = MultiLabelConnectivityDataset(
        npz_files, labels_df,
        all_channels=all_channels,
        matrix_keys=['plv_alpha']
    )
    
    channel_idx = temp_dataset.channel_to_idx[target_channel]
    
    all_features = []
    all_labels = []
    matrix_names_ordered = None
    
    for i in tqdm(range(len(temp_dataset)), desc=f'Extracting for {target_channel}'):
        npz_file = npz_files[i]
        
        try:
            data = np.load(npz_file, allow_pickle=True)
        except:
            continue
        
        # 提取每个矩阵的统计特征
        sample_features = []
        sample_matrix_names = []
        
        for key in sorted(data.files):
            matrix = data[key]
            
            # 只处理2D矩阵
            if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
                continue
            
            # 计算统计特征
            stats = [
                np.mean(matrix),
                np.std(matrix),
                np.min(matrix),
                np.max(matrix),
                np.median(matrix),
                np.percentile(matrix, 25),
                np.percentile(matrix, 75),
                (matrix == 0).sum() / matrix.size,  # 稀疏度
            ]
            
            sample_features.extend(stats)
            
            if matrix_names_ordered is None:
                sample_matrix_names.append(key)
        
        if matrix_names_ordered is None:
            matrix_names_ordered = sample_matrix_names
        
        # 获取标签
        sample = temp_dataset[i]
        label = int(sample['y'][channel_idx].item())
        
        all_features.append(sample_features)
        all_labels.append(label)
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # 生成特征名称
    stat_names = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'sparsity']
    feature_names = []
    for matrix_name in matrix_names_ordered:
        for stat_name in stat_names:
            feature_names.append(f"{matrix_name}_{stat_name}")
    
    return X, y, feature_names, matrix_names_ordered


def analyze_quick_importance(
    npz_files: List[str],
    labels_df: pd.DataFrame,
    channels_to_analyze: List[str],
    seed: int = 42
):
    """快速重要性分析"""
    
    # 为每个通道训练模型
    models = {}
    matrix_importance_all = defaultdict(list)
    
    # 获取矩阵名称（从第一个文件）
    first_data = np.load(npz_files[0], allow_pickle=True)
    all_matrix_names = sorted([k for k in first_data.files 
                               if isinstance(first_data[k], np.ndarray) and first_data[k].ndim == 2])
    
    for channel in channels_to_analyze:
        print(f"\nProcessing channel: {channel}")
        
        # 提取特征
        result = extract_matrix_level_features(npz_files, labels_df, channel)
        
        if result[0] is None:
            continue
        
        X, y, feature_names, matrix_names = result
        
        if y.sum() < 5:
            print(f"  Skipping: too few positive samples ({y.sum()})")
            continue
        
        # 划分数据
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # LightGBM参数
        pos_ratio = y_train.mean()
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1,
            'scale_pos_weight': (1 - pos_ratio) / (pos_ratio + 1e-6),
        }
        
        # 训练
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params, train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # 评估
        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred) if len(np.unique(y_val)) > 1 else 0.5
        
        print(f"  Val AUC: {auc:.4f}, Positive ratio: {y.mean()*100:.1f}%")
        
        models[channel] = model
        
        # 计算每个矩阵的重要性
        importance = model.feature_importance(importance_type='gain')
        
        # 按矩阵聚合（每个矩阵有8个统计特征）
        matrix_scores = defaultdict(float)
        for i, imp in enumerate(importance):
            matrix_name = feature_names[i].rsplit('_', 1)[0]  # 去掉统计名称
            matrix_scores[matrix_name] += imp
        
        # 归一化
        total = sum(matrix_scores.values())
        if total > 0:
            matrix_scores = {k: v/total for k, v in matrix_scores.items()}
        
        # 记录
        for mname in all_matrix_names:
            matrix_importance_all[mname].append(matrix_scores.get(mname, 0.0))
    
    # 3. 汇总结果
    print(f"\n{'='*80}")
    print("Matrix Importance Summary")
    print(f"{'='*80}")
    
    # 计算平均重要性
    avg_importance = {}
    for matrix_name in all_matrix_names:
        scores = matrix_importance_all[matrix_name]
        if scores:
            avg_importance[matrix_name] = np.mean(scores)
        else:
            avg_importance[matrix_name] = 0.0
    
    # 排序
    sorted_matrices = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMatrix Importance Ranking:")
    print("-" * 80)
    for i, (matrix_name, importance) in enumerate(sorted_matrices, 1):
        bar = '█' * int(importance * 50)
        print(f"{i:2d}. {matrix_name:30s} {importance:.4f} {bar}")
    
    return sorted_matrices, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_root', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=300,
                       help='Max samples to use (for speed)')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='feature_selection_quick')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Quick Feature Selection with LightGBM")
    print("="*80)
    
    # 加载数据
    from datasets_multilabel import load_labels_csv, discover_patient_segments_from_csv, discover_all_channels
    
    labels_df = load_labels_csv(args.labels_csv)
    patient_to_files = discover_patient_segments_from_csv(args.labels_csv, args.features_root)
    
    all_files = []
    for files in patient_to_files.values():
        all_files.extend(files)
    
    if args.max_samples:
        all_files = all_files[:args.max_samples]
    
    print(f"Using {len(all_files)} files for analysis")
    
    # 发现通道
    all_channels = discover_all_channels(labels_df)
    
    # 选择要分析的通道（选择出现频率最高的几个）
    from datasets_multilabel import MultiLabelConnectivityDataset
    temp_ds = MultiLabelConnectivityDataset(all_files[:100], labels_df, matrix_keys=['plv_alpha'])
    
    # 统计每个通道的出现次数
    channel_counts = defaultdict(int)
    for i in range(len(temp_ds)):
        sample = temp_ds[i]
        for j, ch in enumerate(all_channels):
            if sample['y'][j] > 0:
                channel_counts[ch] += 1
    
    # 选择top-5最常见的通道
    top_channels = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    channels_to_analyze = [ch for ch, count in top_channels]
    
    print(f"\nAnalyzing top-5 most common channels:")
    for ch, count in top_channels:
        print(f"  {ch}: {count} occurrences")
    
    # 运行分析
    sorted_matrices, models = analyze_quick_importance(
        all_files, labels_df, channels_to_analyze, args.seed
    )
    
    # 选择top-k
    selected = [m for m, _ in sorted_matrices[:args.top_k]]
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDED MATRIX KEYS (Top-{args.top_k}):")
    print(f"{'='*80}")
    print(' '.join(selected))
    
    # 保存
    with open(os.path.join(args.output_dir, 'recommended_keys.txt'), 'w') as f:
        f.write(' '.join(selected))
    
    print(f"\n✓ Saved to {args.output_dir}/recommended_keys.txt")


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend([
            '--features_root', r'E:\output\connectivity_features',
            '--labels_csv', r'E:\output\connectivity_features\labels.csv',
        ])
    main()

