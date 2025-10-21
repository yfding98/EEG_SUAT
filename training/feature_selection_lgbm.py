#!/usr/bin/env python3
"""
使用 LightGBM 进行连接特征的重要性分析和选择

流程:
1. 加载所有NPZ文件中的连接矩阵
2. 将矩阵展平为特征向量
3. 使用 LightGBM 训练多个二分类器（每个通道一个）
4. 分析特征重要性
5. 选择最重要的矩阵/特征
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import json

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def load_npz_features(npz_file: str) -> Dict[str, np.ndarray]:
    """加载NPZ文件中的所有连接矩阵"""
    try:
        data = np.load(npz_file, allow_pickle=True)
        features = {}
        for key in data.files:
            matrix = data[key]
            # 只保留2D矩阵（连接矩阵）
            if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
                features[key] = matrix
        return features
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return {}


def flatten_matrix(matrix: np.ndarray, method: str = 'upper_triangle') -> np.ndarray:
    """
    将连接矩阵展平为特征向量
    
    Args:
        matrix: [N, N] 连接矩阵
        method: 'upper_triangle', 'all', 'diagonal'
    
    Returns:
        features: 1D array
    """
    if method == 'upper_triangle':
        # 只取上三角（因为矩阵对称）
        indices = np.triu_indices(matrix.shape[0], k=1)
        return matrix[indices]
    elif method == 'all':
        return matrix.flatten()
    elif method == 'diagonal':
        return np.diag(matrix)
    else:
        raise ValueError(f"Unknown method: {method}")


def extract_features_from_npz_files(
    npz_files: List[str],
    labels_df: pd.DataFrame,
    target_channel: str,
    flatten_method: str = 'upper_triangle'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    从NPZ文件中提取特征，用于LightGBM训练
    
    Args:
        npz_files: NPZ文件路径列表
        labels_df: 标签DataFrame
        target_channel: 目标通道（如 'Fp1'），预测该通道是否异常
        flatten_method: 矩阵展平方法
    
    Returns:
        X: [n_samples, n_features] 特征矩阵
        y: [n_samples] 标签（0或1）
        feature_names: 特征名称列表
    """
    from datasets_multilabel import MultiLabelConnectivityDataset, discover_all_channels
    
    # 发现所有通道
    all_channels = discover_all_channels(labels_df)
    
    if target_channel not in all_channels:
        raise ValueError(f"Target channel '{target_channel}' not found in data")
    
    print(f"\nExtracting features for target channel: {target_channel}")
    print(f"Total files: {len(npz_files)}")
    
    all_features = []
    all_labels = []
    feature_names = None
    
    # 创建临时数据集来获取标签
    temp_dataset = MultiLabelConnectivityDataset(
        npz_files, labels_df,
        all_channels=all_channels,
        matrix_keys=['plv_alpha']  # 临时的，我们会手动加载所有矩阵
    )
    
    channel_idx = temp_dataset.channel_to_idx[target_channel]
    
    for i, npz_file in enumerate(tqdm(npz_files, desc='Loading features')):
        # 加载所有矩阵
        matrices = load_npz_features(npz_file)
        
        if not matrices:
            continue
        
        # 提取特征
        sample_features = []
        sample_feature_names = []
        
        for matrix_name, matrix in sorted(matrices.items()):
            # 展平矩阵
            flat_features = flatten_matrix(matrix, method=flatten_method)
            sample_features.append(flat_features)
            
            # 特征名称
            for j in range(len(flat_features)):
                sample_feature_names.append(f"{matrix_name}_feat{j}")
        
        # 合并所有矩阵的特征
        sample_features = np.concatenate(sample_features)
        
        # 获取标签
        sample = temp_dataset[i]
        label = int(sample['y'][channel_idx].item())
        
        all_features.append(sample_features)
        all_labels.append(label)
        
        if feature_names is None:
            feature_names = sample_feature_names
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\nExtracted features:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Positive samples: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Negative samples: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")
    
    return X, y, feature_names


def train_lgbm_for_channel(
    X_train, y_train, X_val, y_val,
    target_channel: str,
    params: Dict = None
) -> Tuple[lgb.Booster, Dict]:
    """
    为单个通道训练LightGBM模型
    
    Returns:
        model: 训练好的LightGBM模型
        metrics: 评估指标
    """
    # 默认参数（针对不平衡数据优化）
    if params is None:
        pos_ratio = y_train.mean()
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'scale_pos_weight': (1 - pos_ratio) / (pos_ratio + 1e-6),  # 处理不平衡
            'min_child_samples': 20,
            'max_depth': 7,
        }
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 训练
    print(f"\nTraining LightGBM for channel: {target_channel}")
    print(f"  Positive samples: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)  # 不打印每轮
        ]
    )
    
    # 评估
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    metrics = {
        'train_auc': roc_auc_score(y_train, y_pred_train) if len(np.unique(y_train)) > 1 else 0.5,
        'val_auc': roc_auc_score(y_val, y_pred_val) if len(np.unique(y_val)) > 1 else 0.5,
        'train_ap': average_precision_score(y_train, y_pred_train) if len(np.unique(y_train)) > 1 else y_train.mean(),
        'val_ap': average_precision_score(y_val, y_pred_val) if len(np.unique(y_val)) > 1 else y_val.mean(),
        'best_iteration': model.best_iteration,
    }
    
    print(f"  Train AUC: {metrics['train_auc']:.4f}, Val AUC: {metrics['val_auc']:.4f}")
    print(f"  Val AP: {metrics['val_ap']:.4f}")
    
    return model, metrics


def analyze_matrix_importance(
    models: Dict[str, lgb.Booster],
    feature_names: List[str]
) -> pd.DataFrame:
    """
    分析不同矩阵的重要性
    
    Args:
        models: {channel_name: lgbm_model} 每个通道的模型
        feature_names: 特征名称列表
    
    Returns:
        importance_df: 矩阵重要性汇总
    """
    # 提取矩阵名称（去掉 _featXX 后缀）
    matrix_names = set()
    for fname in feature_names:
        matrix_name = fname.rsplit('_feat', 1)[0]
        matrix_names.add(matrix_name)
    
    matrix_names = sorted(matrix_names)
    
    # 收集每个矩阵的重要性
    matrix_importance = defaultdict(list)
    
    for channel_name, model in models.items():
        # 获取特征重要性
        importance = model.feature_importance(importance_type='gain')
        
        # 按矩阵聚合
        matrix_scores = {mname: 0.0 for mname in matrix_names}
        
        for feat_idx, score in enumerate(importance):
            feat_name = feature_names[feat_idx]
            matrix_name = feat_name.rsplit('_feat', 1)[0]
            matrix_scores[matrix_name] += score
        
        # 归一化
        total_score = sum(matrix_scores.values())
        if total_score > 0:
            matrix_scores = {k: v/total_score for k, v in matrix_scores.items()}
        
        # 记录
        for mname in matrix_names:
            matrix_importance[mname].append(matrix_scores[mname])
    
    # 创建DataFrame
    importance_df = pd.DataFrame(matrix_importance, index=list(models.keys()))
    
    # 计算总体重要性
    importance_df.loc['Mean'] = importance_df.mean()
    importance_df.loc['Std'] = importance_df.std()
    
    return importance_df


def visualize_matrix_importance(importance_df: pd.DataFrame, save_path: str):
    """可视化矩阵重要性"""
    # 去掉统计行
    plot_df = importance_df.drop(['Mean', 'Std'], errors='ignore')
    
    # 热图
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. 热图
    ax = axes[0]
    sns.heatmap(plot_df.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Matrix Type')
    ax.set_title('Feature Matrix Importance per Channel')
    
    # 2. 平均重要性柱状图
    ax = axes[1]
    mean_importance = importance_df.loc['Mean'].sort_values(ascending=False)
    colors = plt.cm.YlOrRd(mean_importance / mean_importance.max())
    ax.barh(range(len(mean_importance)), mean_importance.values, color=colors)
    ax.set_yticks(range(len(mean_importance)))
    ax.set_yticklabels(mean_importance.index)
    ax.set_xlabel('Mean Importance Score')
    ax.set_title('Average Matrix Importance Across All Channels')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    plt.close()


def select_top_matrices(
    importance_df: pd.DataFrame,
    top_k: int = None,
    threshold: float = None
) -> List[str]:
    """
    选择最重要的矩阵
    
    Args:
        importance_df: 重要性DataFrame
        top_k: 选择top-k个矩阵
        threshold: 或选择重要性>阈值的矩阵
    
    Returns:
        selected_matrices: 矩阵名称列表
    """
    mean_importance = importance_df.loc['Mean'].sort_values(ascending=False)
    
    if top_k is not None:
        selected = mean_importance.head(top_k).index.tolist()
        print(f"\n✓ Selected top-{top_k} matrices:")
    elif threshold is not None:
        selected = mean_importance[mean_importance > threshold].index.tolist()
        print(f"\n✓ Selected matrices with importance > {threshold}:")
    else:
        # 默认：选择累计重要性达到80%的矩阵
        cumsum = mean_importance.cumsum()
        selected = cumsum[cumsum <= 0.8].index.tolist()
        if len(selected) == 0:
            selected = [mean_importance.index[0]]
        print(f"\n✓ Selected matrices (cumulative importance ≈ 80%):")
    
    for i, matrix_name in enumerate(selected, 1):
        importance = mean_importance[matrix_name]
        print(f"  {i}. {matrix_name}: {importance:.4f}")
    
    return selected


def main():
    parser = argparse.ArgumentParser(
        description='Feature Selection using LightGBM for Connectivity Matrices'
    )
    
    # 数据参数
    parser.add_argument('--features_root', type=str, required=True,
                       help='Root directory containing connectivity features')
    parser.add_argument('--labels_csv', type=str, required=True,
                       help='CSV file with labels')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to use (for speed)')
    
    # 特征提取参数
    parser.add_argument('--flatten_method', type=str, default='upper_triangle',
                       choices=['upper_triangle', 'all', 'diagonal'],
                       help='How to flatten connectivity matrices')
    parser.add_argument('--target_channels', nargs='+', default=None,
                       help='Target channels to train models for (default: all)')
    
    # 选择参数
    parser.add_argument('--selection_method', type=str, default='top_k',
                       choices=['top_k', 'threshold', 'cumulative'],
                       help='How to select important matrices')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top matrices to select')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Importance threshold')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='feature_selection_results',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Feature Selection using LightGBM")
    print("=" * 80)
    print(f"Features root: {args.features_root}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n1. Loading data...")
    from datasets_multilabel import load_labels_csv, discover_patient_segments_from_csv, discover_all_channels
    
    labels_df = load_labels_csv(args.labels_csv)
    patient_to_files = discover_patient_segments_from_csv(args.labels_csv, args.features_root)
    
    # 获取所有文件
    all_files = []
    for files in patient_to_files.values():
        all_files.extend(files)
    
    if args.max_samples:
        all_files = all_files[:args.max_samples]
    
    print(f"Total files: {len(all_files)}")
    
    # 发现所有通道
    all_channels = discover_all_channels(labels_df)
    print(f"All channels: {all_channels}")
    
    # 确定要分析的通道
    if args.target_channels:
        target_channels = args.target_channels
    else:
        target_channels = all_channels
    
    print(f"\nWill train models for {len(target_channels)} channels")
    
    # 2. 为每个通道训练模型
    print("\n2. Training LightGBM models for each channel...")
    
    models = {}
    all_metrics = {}
    
    for target_channel in target_channels:
        print(f"\n{'='*80}")
        print(f"Channel: {target_channel}")
        print(f"{'='*80}")
        
        # 提取特征
        X, y, feature_names = extract_features_from_npz_files(
            all_files, labels_df, target_channel,
            flatten_method=args.flatten_method
        )
        
        # 检查是否有足够的正样本
        if y.sum() < 5:
            print(f"  ⚠ Skipping {target_channel}: too few positive samples ({y.sum()})")
            continue
        
        # 划分训练/验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y
        )
        
        # 训练模型
        model, metrics = train_lgbm_for_channel(
            X_train, y_train, X_val, y_val,
            target_channel
        )
        
        models[target_channel] = model
        all_metrics[target_channel] = metrics
    
    if not models:
        print("\n✗ No models trained! Check your data.")
        return
    
    # 3. 分析特征重要性
    print(f"\n{'='*80}")
    print("3. Analyzing Feature Importance")
    print(f"{'='*80}")
    
    importance_df = analyze_matrix_importance(models, feature_names)
    
    # 保存重要性表
    importance_csv = os.path.join(args.output_dir, 'matrix_importance.csv')
    importance_df.to_csv(importance_csv)
    print(f"\n✓ Importance table saved to {importance_csv}")
    
    # 显示结果
    print("\n" + "="*80)
    print("Matrix Importance Summary:")
    print("="*80)
    print(importance_df.loc[['Mean', 'Std']])
    
    # 4. 可视化
    print(f"\n4. Creating visualizations...")
    viz_path = os.path.join(args.output_dir, 'matrix_importance.png')
    visualize_matrix_importance(importance_df, viz_path)
    
    # 5. 选择重要矩阵
    print(f"\n5. Selecting important matrices...")
    
    if args.selection_method == 'top_k':
        selected_matrices = select_top_matrices(importance_df, top_k=args.top_k)
    elif args.selection_method == 'threshold':
        selected_matrices = select_top_matrices(importance_df, threshold=args.threshold)
    else:  # cumulative
        selected_matrices = select_top_matrices(importance_df)
    
    # 6. 保存结果
    print(f"\n6. Saving results...")
    
    # 保存选择的矩阵列表
    selected_file = os.path.join(args.output_dir, 'selected_matrices.txt')
    with open(selected_file, 'w') as f:
        for matrix in selected_matrices:
            f.write(f"{matrix}\n")
    print(f"  ✓ Selected matrices saved to {selected_file}")
    
    # 保存完整报告
    report = {
        'selected_matrices': selected_matrices,
        'importance_scores': importance_df.loc['Mean'].to_dict(),
        'metrics_per_channel': all_metrics,
        'parameters': vars(args)
    }
    
    report_file = os.path.join(args.output_dir, 'feature_selection_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Full report saved to {report_file}")
    
    # 7. 生成使用建议
    print(f"\n{'='*80}")
    print("Usage Recommendations")
    print(f"{'='*80}")
    
    print(f"\n✓ Use these matrices in your GNN training:")
    print(f"\n  --matrix_keys {' '.join(selected_matrices)}")
    
    print(f"\n Example command:")
    print(f"  python training/train_multilabel_improved.py \\")
    print(f"    --features_root {args.features_root} \\")
    print(f"    --labels_csv {args.labels_csv} \\")
    print(f"    --matrix_keys {' '.join(selected_matrices)} \\")
    print(f"    --loss_type focal \\")
    print(f"    --use_weighted_sampler")
    
    print(f"\n✓ Feature selection complete!")
    print(f"  Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--features_root', r'E:\output\connectivity_features',
            '--labels_csv', r'E:\output\connectivity_features\labels.csv',
            '--max_samples', '500',
            '--selection_method', 'top_k',
            '--top_k', '5',
            '--output_dir', 'feature_selection_results'
        ])
    main()

