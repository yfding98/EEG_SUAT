#!/usr/bin/env python3
"""
多标签分类评估和可视化脚本
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from datasets_multilabel import (
    MultiLabelConnectivityDataset,
    load_labels_csv,
    discover_patient_segments_from_csv,
    make_patient_splits,
    collate_graph_multilabel
)
from models_multilabel import MultiLabelGNNClassifier, MultiLabelGNNWithAttention
from torch.utils.data import DataLoader


def load_model_and_data(checkpoint_path, features_root, labels_csv, device='cpu'):
    """加载模型和数据"""
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    channel_names = checkpoint['channel_names']
    
    # 加载数据
    labels_df = load_labels_csv(labels_csv)
    patient_to_files = discover_patient_segments_from_csv(labels_csv, features_root)
    splits = make_patient_splits(patient_to_files, test_ratio=0.2, val_ratio=0.1, seed=args.seed)
    
    # 创建测试数据集
    test_dataset = MultiLabelConnectivityDataset(
        splits['test'], labels_df,
        all_channels=channel_names,
        matrix_keys=args.matrix_keys,
        fusion_method='weighted'
    )
    
    # 创建模型
    if args.model_type == 'basic':
        model = MultiLabelGNNClassifier(
            in_dim=2,
            hidden_dim=args.hidden_dim,
            num_channels=len(channel_names),
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        model = MultiLabelGNNWithAttention(
            in_dim=2,
            hidden_dim=args.hidden_dim,
            num_channels=len(channel_names),
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_heads=args.num_heads
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, test_dataset, channel_names, args


def visualize_confusion_matrices(labels, preds, channel_names, save_path):
    """可视化每个通道的混淆矩阵"""
    from sklearn.metrics import multilabel_confusion_matrix
    
    cm_array = multilabel_confusion_matrix(labels, preds)
    
    n_channels = len(channel_names)
    n_cols = 6
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows))
    axes = axes.flatten()
    
    for idx, (cm, ch_name) in enumerate(zip(cm_array, channel_names)):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        ax.set_title(f'{ch_name}')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
    
    # 隐藏多余的子图
    for idx in range(n_channels, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrices to {save_path}")
    plt.close()


def visualize_channel_performance(labels, preds, probs, channel_names, save_path):
    """可视化每个通道的性能"""
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score
    
    # 计算每个通道的指标
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    # 计算AP
    ap_scores = []
    for i in range(len(channel_names)):
        ap = average_precision_score(labels[:, i], probs[:, i])
        ap_scores.append(ap)
    ap_scores = np.array(ap_scores)
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Precision
    ax = axes[0, 0]
    ax.barh(channel_names, precision)
    ax.set_xlabel('Precision')
    ax.set_title('Precision per Channel')
    ax.grid(True, alpha=0.3)
    
    # Recall
    ax = axes[0, 1]
    ax.barh(channel_names, recall)
    ax.set_xlabel('Recall')
    ax.set_title('Recall per Channel')
    ax.grid(True, alpha=0.3)
    
    # F1
    ax = axes[1, 0]
    ax.barh(channel_names, f1)
    ax.set_xlabel('F1 Score')
    ax.set_title('F1 Score per Channel')
    ax.grid(True, alpha=0.3)
    
    # Average Precision
    ax = axes[1, 1]
    ax.barh(channel_names, ap_scores)
    ax.set_xlabel('Average Precision')
    ax.set_title('Average Precision per Channel')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved channel performance to {save_path}")
    plt.close()


def visualize_prediction_examples(model, dataset, channel_names, device, save_path, n_examples=5):
    """可视化预测示例"""
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=n_examples, shuffle=True, collate_fn=collate_graph_multilabel)
    batch = next(iter(loader))
    
    adj = batch['adj'].to(device)
    x = batch['x'].to(device)
    true_labels = batch['y'].numpy()
    
    with torch.no_grad():
        logits = model(x, adj)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for idx in range(n_examples):
        ax = axes[idx]
        
        # 真实标签和预测概率
        true = true_labels[idx]
        pred = probs[idx]
        
        x_pos = np.arange(len(channel_names))
        
        # 绘制预测概率
        colors = ['red' if t == 1 else 'blue' for t in true]
        bars = ax.bar(x_pos, pred, color=colors, alpha=0.6)
        
        # 添加阈值线
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Threshold=0.5')
        
        # 标记真实异常通道
        for i, t in enumerate(true):
            if t == 1:
                ax.plot(i, pred[i], 'r*', markersize=15)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(channel_names, rotation=45, ha='right')
        ax.set_ylabel('Abnormality Probability')
        ax.set_ylim([0, 1])
        ax.set_title(f'Example {idx+1}: True abnormal channels = {", ".join([channel_names[i] for i, t in enumerate(true) if t==1])}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved prediction examples to {save_path}")
    plt.close()


def analyze_top_k_predictions(probs, labels, channel_names, k=5):
    """分析Top-K预测"""
    print(f"\nTop-{k} Analysis:")
    print("="*60)
    
    n_samples = len(probs)
    top_k_correct = 0
    
    for i in range(n_samples):
        # Top-k预测
        top_k_indices = np.argsort(probs[i])[::-1][:k]
        top_k_channels = [channel_names[idx] for idx in top_k_indices]
        
        # 真实异常通道
        true_indices = np.where(labels[i] == 1)[0]
        true_channels = [channel_names[idx] for idx in true_indices]
        
        # 计算overlap
        overlap = len(set(top_k_indices) & set(true_indices))
        top_k_correct += overlap
        
        if i < 5:  # 打印前5个示例
            print(f"\nSample {i+1}:")
            print(f"  True abnormal: {', '.join(true_channels) if true_channels else 'None'}")
            print(f"  Top-{k} predicted: {', '.join(top_k_channels)}")
            print(f"  Overlap: {overlap}/{len(true_channels) if true_channels else 0}")
    
    # 总体统计
    total_true = labels.sum()
    top_k_recall = top_k_correct / total_true if total_true > 0 else 0
    print(f"\nOverall Top-{k} Recall: {top_k_recall:.4f}")
    print(f"  (Found {top_k_correct} out of {int(total_true)} true abnormal channels)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Multi-Label Model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_multilabel/best.pt',
                       help='Path to checkpoint')
    parser.add_argument('--features_root', type=str, required=True,
                       help='Root directory containing connectivity features')
    parser.add_argument('--labels_csv', type=str, required=True,
                       help='CSV file with labels')
    parser.add_argument('--output_dir', type=str, default='multilabel_results',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Multi-Label Model Evaluation")
    print("="*60)
    
    # 加载模型和数据
    print("\nLoading model and data...")
    model, test_dataset, channel_names, model_args = load_model_and_data(
        args.checkpoint, args.features_root, args.labels_csv, args.device
    )
    
    # 创建DataLoader
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=0, collate_fn=collate_graph_multilabel
    )
    
    # 获取预测
    print("\nMaking predictions...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            adj = batch['adj'].to(args.device)
            x = batch['x'].to(args.device)
            labels = batch['y'].numpy()
            
            logits = model(x, adj)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels)
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    all_preds = (all_probs > args.threshold).astype(np.float32)
    
    print(f"Total test samples: {len(all_labels)}")
    print(f"Number of channels: {len(channel_names)}")
    
    # 可视化
    print("\nGenerating visualizations...")
    
    # 1. 混淆矩阵
    visualize_confusion_matrices(
        all_labels, all_preds, channel_names,
        os.path.join(args.output_dir, 'confusion_matrices.png')
    )
    
    # 2. 通道性能
    visualize_channel_performance(
        all_labels, all_preds, all_probs, channel_names,
        os.path.join(args.output_dir, 'channel_performance.png')
    )
    
    # 3. 预测示例
    visualize_prediction_examples(
        model, test_dataset, channel_names, args.device,
        os.path.join(args.output_dir, 'prediction_examples.png'),
        n_examples=5
    )
    
    # 4. Top-K分析
    analyze_top_k_predictions(all_probs, all_labels, channel_names, k=5)
    
    print(f"\n✓ All visualizations saved to {args.output_dir}/")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--checkpoint', 'checkpoints_multilabel/best.pt',
            '--features_root', r'E:\output\connectivity_features',
            '--labels_csv', r'E:\output\connectivity_features\labels.csv',
            '--output_dir', 'multilabel_results',
            '--device', 'cpu'
        ])
    main()

