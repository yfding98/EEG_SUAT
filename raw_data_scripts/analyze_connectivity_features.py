#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_connectivity_features.py

示例脚本：展示如何读取和分析extract_connectivity_features.py生成的连接性特征

使用方法:
    python analyze_connectivity_features.py --feature_dir "path/to/features_directory"
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_connectivity_features(feature_dir):
    """
    加载单个特征目录的所有数据
    
    返回:
        scalar_df: DataFrame 标量特征
        matrices: list of dict 连接矩阵
        channel_names: list 通道名
        summary: dict 汇总信息
    """
    # 1. 读取标量特征
    scalar_path = os.path.join(feature_dir, 'scalar_features.csv')
    if not os.path.exists(scalar_path):
        raise FileNotFoundError(f"Scalar features not found: {scalar_path}")
    
    scalar_df = pd.read_csv(scalar_path)
    print(f"Loaded {len(scalar_df)} segments from {feature_dir}")
    
    # 2. 读取连接矩阵
    matrices = []
    for seg_id in range(len(scalar_df)):
        npz_path = os.path.join(feature_dir, f'connectivity_matrices_seg{seg_id:03d}.npz')
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            matrices.append(dict(data))
    
    print(f"Loaded {len(matrices)} connectivity matrix files")
    
    # 3. 读取通道名
    channel_path = os.path.join(feature_dir, 'channel_names.txt')
    with open(channel_path, 'r') as f:
        channel_names = [line.strip() for line in f.readlines()]
    
    print(f"Channels: {', '.join(channel_names)}")
    
    # 4. 读取汇总信息
    summary = {}
    summary_path = os.path.join(feature_dir, 'summary.txt')
    with open(summary_path, 'r') as f:
        for line in f:
            if ':' in line and 'channel_names' not in line:
                key, value = line.strip().split(':', 1)
                try:
                    summary[key.strip()] = float(value.strip())
                except:
                    summary[key.strip()] = value.strip()
    
    return scalar_df, matrices, channel_names, summary


def visualize_scalar_features(scalar_df, output_dir):
    """可视化标量特征的时间演化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 图网络指标随时间变化
    graph_cols = [col for col in scalar_df.columns if 'graph_' in col]
    
    if graph_cols:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # 选择几个关键指标
        key_metrics = ['degree_mean', 'clustering_mean', 'betweenness_mean', 'modularity']
        
        for idx, metric in enumerate(key_metrics):
            matching_cols = [col for col in graph_cols if metric in col]
            if matching_cols and idx < len(axes):
                for col in matching_cols:
                    axes[idx].plot(scalar_df['start_time'], scalar_df[col], 
                                  marker='o', label=col.replace('_graph_', '\n'))
                axes[idx].set_xlabel('Time (s)')
                axes[idx].set_ylabel(metric.replace('_', ' ').title())
                axes[idx].set_title(f'{metric.replace("_", " ").title()} Over Time')
                axes[idx].legend(fontsize=8)
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'graph_metrics_timeseries.png'), dpi=150)
        plt.close()
        print(f"✓ Saved: graph_metrics_timeseries.png")
    
    # 2. 动态连接指标
    dfc_cols = [col for col in scalar_df.columns if 'dfc_' in col]
    
    if dfc_cols:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for idx, col in enumerate(dfc_cols):
            if idx < len(axes):
                axes[idx].plot(scalar_df['start_time'], scalar_df[col], 
                              marker='o', color='steelblue')
                axes[idx].set_xlabel('Time (s)')
                axes[idx].set_ylabel(col.replace('dfc_', '').replace('_', ' ').title())
                axes[idx].set_title(f'{col.replace("dfc_", "").replace("_", " ").title()}')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dfc_metrics_timeseries.png'), dpi=150)
        plt.close()
        print(f"✓ Saved: dfc_metrics_timeseries.png")


def visualize_connectivity_matrices(matrices, channel_names, output_dir, segment_id=0):
    """可视化某个片段的连接性矩阵"""
    os.makedirs(output_dir, exist_ok=True)
    
    if segment_id >= len(matrices):
        print(f"Warning: segment {segment_id} not found")
        return
    
    seg_data = matrices[segment_id]
    
    # 选择要可视化的矩阵
    matrices_to_plot = {
        'Pearson Correlation': 'pearson_corr',
        'Spearman Correlation': 'spearman_corr',
        'PLV (Alpha)': 'plv_alpha',
        'wPLI (Alpha)': 'wpli_alpha',
        'Coherence (Alpha)': 'coherence_alpha',
        'iCOH (Alpha)': 'icoh_alpha'
    }
    
    available_matrices = {k: v for k, v in matrices_to_plot.items() if v in seg_data}
    
    if not available_matrices:
        print("No matrices to visualize")
        return
    
    # 创建子图
    n_matrices = len(available_matrices)
    n_cols = 3
    n_rows = (n_matrices + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (title, key) in enumerate(available_matrices.items()):
        matrix = seg_data[key]
        
        # 绘制热图
        im = axes[idx].imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Channels')
        axes[idx].set_ylabel('Channels')
        
        # 设置刻度标签
        if len(channel_names) <= 20:  # 只在通道数较少时显示名称
            axes[idx].set_xticks(range(len(channel_names)))
            axes[idx].set_yticks(range(len(channel_names)))
            axes[idx].set_xticklabels(channel_names, rotation=90, fontsize=8)
            axes[idx].set_yticklabels(channel_names, fontsize=8)
        
        plt.colorbar(im, ax=axes[idx])
    
    # 隐藏多余的子图
    for idx in range(len(available_matrices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Connectivity Matrices - Segment {segment_id}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'connectivity_matrices_seg{segment_id:03d}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: connectivity_matrices_seg{segment_id:03d}.png")


def compare_frequency_bands(matrices, channel_names, output_dir, segment_id=0, metric='plv'):
    """比较不同频段的连接性"""
    os.makedirs(output_dir, exist_ok=True)
    
    if segment_id >= len(matrices):
        return
    
    seg_data = matrices[segment_id]
    
    # 查找所有频段
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    available_bands = [b for b in bands if f'{metric}_{b}' in seg_data]
    
    if not available_bands:
        print(f"No {metric} data for frequency bands")
        return
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, band in enumerate(available_bands):
        if idx >= len(axes):
            break
        
        key = f'{metric}_{band}'
        matrix = seg_data[key]
        
        # 绘制热图
        im = axes[idx].imshow(matrix, cmap='hot', vmin=0, vmax=1, aspect='auto')
        axes[idx].set_title(f'{band.upper()} ({metric.upper()})')
        axes[idx].set_xlabel('Channels')
        axes[idx].set_ylabel('Channels')
        
        if len(channel_names) <= 20:
            axes[idx].set_xticks(range(len(channel_names)))
            axes[idx].set_yticks(range(len(channel_names)))
            axes[idx].set_xticklabels(channel_names, rotation=90, fontsize=8)
            axes[idx].set_yticklabels(channel_names, fontsize=8)
        
        plt.colorbar(im, ax=axes[idx])
    
    # 隐藏多余子图
    for idx in range(len(available_bands), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{metric.upper()} Across Frequency Bands - Segment {segment_id}', 
                 fontsize=16, y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric}_frequency_comparison_seg{segment_id:03d}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {metric}_frequency_comparison_seg{segment_id:03d}.png")


def compute_averaged_connectivity(matrices, metric='pearson_corr'):
    """计算所有片段的平均连接性矩阵"""
    all_matrices = []
    
    for seg_data in matrices:
        if metric in seg_data:
            all_matrices.append(seg_data[metric])
    
    if not all_matrices:
        return None
    
    return np.mean(all_matrices, axis=0), np.std(all_matrices, axis=0)


def generate_report(scalar_df, matrices, channel_names, summary, output_dir):
    """生成分析报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EEG连接性特征分析报告\n")
        f.write("="*80 + "\n\n")
        
        # 基本信息
        f.write("1. 数据概况\n")
        f.write("-"*80 + "\n")
        f.write(f"通道数: {summary.get('n_channels', 'N/A')}\n")
        f.write(f"片段数: {summary.get('n_segments', 'N/A')}\n")
        f.write(f"总时长: {summary.get('total_duration', 'N/A'):.2f} 秒\n")
        f.write(f"窗口大小: {summary.get('window_size', 'N/A'):.2f} 秒\n")
        f.write(f"通道列表: {', '.join(channel_names)}\n\n")
        
        # 标量特征统计
        f.write("2. 标量特征统计\n")
        f.write("-"*80 + "\n")
        
        # 图网络指标
        graph_cols = [col for col in scalar_df.columns if 'graph_' in col]
        if graph_cols:
            f.write("\n图网络指标:\n")
            for col in graph_cols[:10]:  # 只显示前10个
                mean_val = scalar_df[col].mean()
                std_val = scalar_df[col].std()
                f.write(f"  {col}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        # 动态连接指标
        dfc_cols = [col for col in scalar_df.columns if 'dfc_' in col]
        if dfc_cols:
            f.write("\n动态连接指标:\n")
            for col in dfc_cols:
                mean_val = scalar_df[col].mean()
                std_val = scalar_df[col].std()
                f.write(f"  {col}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        # 连接性矩阵统计
        f.write("\n3. 连接性矩阵统计\n")
        f.write("-"*80 + "\n")
        
        if matrices:
            sample_matrix_keys = matrices[0].keys()
            f.write(f"可用的连接性指标: {len(sample_matrix_keys)}\n")
            f.write(f"  {', '.join(list(sample_matrix_keys)[:10])}\n")
            
            # 计算平均连接强度
            for metric in ['pearson_corr', 'plv_alpha', 'coherence_alpha']:
                avg_matrix, std_matrix = compute_averaged_connectivity(matrices, metric)
                if avg_matrix is not None:
                    # 提取上三角（去除对角线）
                    n = avg_matrix.shape[0]
                    triu_indices = np.triu_indices(n, k=1)
                    avg_conn = avg_matrix[triu_indices]
                    
                    f.write(f"\n{metric}:\n")
                    f.write(f"  平均连接强度: {np.mean(avg_conn):.4f} ± {np.std(avg_conn):.4f}\n")
                    f.write(f"  最大连接: {np.max(avg_conn):.4f}\n")
                    f.write(f"  最小连接: {np.min(avg_conn):.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("报告生成完成\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Saved: analysis_report.txt")
    
    # 打印到控制台
    with open(report_path, 'r', encoding='utf-8') as f:
        print("\n" + f.read())


def main():
    parser = argparse.ArgumentParser(
        description="分析EEG连接性特征",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析单个特征目录
  python analyze_connectivity_features.py --feature_dir "path/to/features"
  
  # 指定输出目录
  python analyze_connectivity_features.py --feature_dir "path/to/features" --output_dir "analysis_results"
  
  # 可视化特定片段
  python analyze_connectivity_features.py --feature_dir "path/to/features" --segment_id 5
        """
    )
    
    parser.add_argument('--feature_dir', required=True, 
                       help="特征目录路径（包含scalar_features.csv等文件）")
    parser.add_argument('--output_dir', default=None,
                       help="输出目录，默认为feature_dir/analysis")
    parser.add_argument('--segment_id', type=int, default=0,
                       help="要可视化的片段ID，默认为0")
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.feature_dir, 'analysis')
    
    print(f"\n{'='*80}")
    print(f"EEG连接性特征分析")
    print(f"{'='*80}\n")
    
    # 加载数据
    print("加载数据...")
    scalar_df, matrices, channel_names, summary = load_connectivity_features(args.feature_dir)
    
    # 生成可视化
    print("\n生成可视化...")
    visualize_scalar_features(scalar_df, args.output_dir)
    visualize_connectivity_matrices(matrices, channel_names, args.output_dir, args.segment_id)
    compare_frequency_bands(matrices, channel_names, args.output_dir, args.segment_id, 'plv')
    compare_frequency_bands(matrices, channel_names, args.output_dir, args.segment_id, 'coherence')
    
    # 生成报告
    print("\n生成报告...")
    generate_report(scalar_df, matrices, channel_names, summary, args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"分析完成！")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

