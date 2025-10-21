#!/usr/bin/env python3
"""
阈值优化脚本

在验证集上寻找最佳分类阈值，平衡 Precision 和 Recall
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, jaccard_score,
    average_precision_score, f1_score
)


def find_optimal_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    metric: str = 'f1',
    beta: float = 1.0
):
    """
    寻找最优阈值
    
    Args:
        probs: [N, num_channels] 预测概率
        labels: [N, num_channels] 真实标签
        metric: 优化目标 ('f1', 'precision', 'recall', 'jaccard')
        beta: F-beta score的beta值 (beta=1是F1, beta=2更重视recall)
    
    Returns:
        best_threshold: 最优阈值
        best_score: 最优分数
        all_results: 所有阈值的结果
    """
    thresholds = np.arange(0.05, 0.95, 0.05)
    results = []
    
    for thresh in thresholds:
        preds = (probs > thresh).astype(np.float32)
        
        # 计算各种指标
        try:
            # Sample-wise metrics
            jaccard = jaccard_score(labels, preds, average='samples', zero_division=0)
            
            # Micro-average metrics
            precision = ((preds * labels).sum()) / (preds.sum() + 1e-8)
            recall = ((preds * labels).sum()) / (labels.sum() + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
            
            results.append({
                'threshold': thresh,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fbeta': fbeta,
                'jaccard': jaccard,
            })
        except:
            continue
    
    # 选择最优阈值
    if metric == 'f1':
        best = max(results, key=lambda x: x['f1'])
    elif metric == 'fbeta':
        best = max(results, key=lambda x: x['fbeta'])
    elif metric == 'precision':
        best = max(results, key=lambda x: x['precision'])
    elif metric == 'recall':
        best = max(results, key=lambda x: x['recall'])
    else:  # jaccard
        best = max(results, key=lambda x: x['jaccard'])
    
    return best['threshold'], best[metric], results


def visualize_threshold_analysis(results, save_path):
    """可视化阈值分析结果"""
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    jaccards = [r['jaccard'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Precision-Recall曲线
    ax = axes[0, 0]
    ax.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')
    ax.scatter(recalls, precisions, c=thresholds, cmap='viridis', s=50)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Trade-off')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. 各指标随阈值变化
    ax = axes[0, 1]
    ax.plot(thresholds, precisions, 'r-', label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, 'b-', label='Recall', linewidth=2)
    ax.plot(thresholds, f1s, 'g-', label='F1', linewidth=2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. F1和Jaccard
    ax = axes[1, 0]
    ax.plot(thresholds, f1s, 'g-', label='F1', linewidth=2)
    ax.plot(thresholds, jaccards, 'm-', label='Jaccard', linewidth=2)
    
    # 标记最优点
    best_f1_idx = np.argmax(f1s)
    best_jaccard_idx = np.argmax(jaccards)
    ax.scatter([thresholds[best_f1_idx]], [f1s[best_f1_idx]], 
              c='green', s=200, marker='*', label=f'Best F1 @ {thresholds[best_f1_idx]:.2f}')
    ax.scatter([thresholds[best_jaccard_idx]], [jaccards[best_jaccard_idx]], 
              c='magenta', s=200, marker='*', label=f'Best Jaccard @ {thresholds[best_jaccard_idx]:.2f}')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('F1 and Jaccard vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 详细数值表
    ax = axes[1, 1]
    ax.axis('off')
    
    # 找出关键阈值点
    best_f1 = max(results, key=lambda x: x['f1'])
    best_jaccard = max(results, key=lambda x: x['jaccard'])
    best_balanced = max(results, key=lambda x: abs(x['precision'] - x['recall']))  # P和R最接近
    
    table_text = f"""
    Optimal Thresholds:
    
    Best F1 Score:
      Threshold: {best_f1['threshold']:.3f}
      Precision: {best_f1['precision']:.3f}
      Recall: {best_f1['recall']:.3f}
      F1: {best_f1['f1']:.3f}
      Jaccard: {best_f1['jaccard']:.3f}
    
    Best Jaccard:
      Threshold: {best_jaccard['threshold']:.3f}
      Precision: {best_jaccard['precision']:.3f}
      Recall: {best_jaccard['recall']:.3f}
      F1: {best_jaccard['f1']:.3f}
      Jaccard: {best_jaccard['jaccard']:.3f}
    
    Best Balanced (P≈R):
      Threshold: {best_balanced['threshold']:.3f}
      Precision: {best_balanced['precision']:.3f}
      Recall: {best_balanced['recall']:.3f}
      F1: {best_balanced['f1']:.3f}
    """
    
    ax.text(0.1, 0.5, table_text, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Threshold Optimization')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to test_results.npz from training')
    parser.add_argument('--metric', type=str, default='f1',
                       choices=['f1', 'jaccard', 'precision', 'recall', 'fbeta'],
                       help='Metric to optimize')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta for F-beta score (beta=1: F1, beta=2: favor recall)')
    parser.add_argument('--output_dir', type=str, default='threshold_optimization',
                       help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Threshold Optimization")
    print("=" * 80)
    
    # 加载结果
    print(f"\nLoading results from: {args.results_file}")
    data = np.load(args.results_file, allow_pickle=True)
    
    probs = data['probs']
    labels = data['labels']
    channel_names = data['channel_names'].tolist()
    
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Channels: {len(channel_names)}")
    
    # 寻找最优阈值
    print(f"\nOptimizing for: {args.metric}")
    best_threshold, best_score, all_results = find_optimal_threshold(
        probs, labels,
        metric=args.metric,
        beta=args.beta
    )
    
    print(f"\n✓ Optimal threshold found: {best_threshold:.3f}")
    print(f"  {args.metric}: {best_score:.4f}")
    
    # 显示详细结果
    print(f"\n{'='*80}")
    print("Detailed Results at Optimal Threshold")
    print(f"{'='*80}")
    
    optimal_result = [r for r in all_results if r['threshold'] == best_threshold][0]
    
    print(f"\nThreshold: {optimal_result['threshold']:.3f}")
    print(f"  Precision: {optimal_result['precision']:.3f}")
    print(f"  Recall: {optimal_result['recall']:.3f}")
    print(f"  F1: {optimal_result['f1']:.3f}")
    print(f"  Jaccard: {optimal_result['jaccard']:.3f}")
    
    # 也显示其他关键阈值点
    print(f"\n{'='*80}")
    print("Other Important Thresholds")
    print(f"{'='*80}")
    
    best_f1 = max(all_results, key=lambda x: x['f1'])
    best_jaccard = max(all_results, key=lambda x: x['jaccard'])
    best_precision = max(all_results, key=lambda x: x['precision'])
    best_recall = max(all_results, key=lambda x: x['recall'])
    
    print(f"\nBest F1: {best_f1['threshold']:.3f} (F1={best_f1['f1']:.3f})")
    print(f"Best Jaccard: {best_jaccard['threshold']:.3f} (Jaccard={best_jaccard['jaccard']:.3f})")
    print(f"Best Precision: {best_precision['threshold']:.3f} (P={best_precision['precision']:.3f}, R={best_precision['recall']:.3f})")
    print(f"Best Recall: {best_recall['threshold']:.3f} (R={best_recall['recall']:.3f}, P={best_recall['precision']:.3f})")
    
    # 可视化
    print(f"\nGenerating visualization...")
    viz_path = os.path.join(args.output_dir, 'threshold_analysis.png')
    visualize_threshold_analysis(all_results, viz_path)
    
    # 保存结果
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, 'threshold_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to {csv_path}")
    
    # 保存推荐
    recommendation = {
        'optimal_threshold': float(best_threshold),
        'metric_optimized': args.metric,
        'score_at_optimal': float(best_score),
        'best_f1_threshold': float(best_f1['threshold']),
        'best_jaccard_threshold': float(best_jaccard['threshold']),
    }
    
    import json
    rec_path = os.path.join(args.output_dir, 'recommended_threshold.json')
    with open(rec_path, 'w') as f:
        json.dump(recommendation, f, indent=2)
    print(f"✓ Recommendation saved to {rec_path}")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    print(f"\nUse threshold: {best_threshold:.3f}")
    print(f"\nIn your training script:")
    print(f"  --threshold {best_threshold:.3f}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--results_file', 'checkpoints_multilabel_improved/test_results.npz',
            '--metric', 'f1',
            '--output_dir', 'threshold_optimization'
        ])
    main()

