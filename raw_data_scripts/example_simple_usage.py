#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_simple_usage.py

简单的连接性特征提取和分析示例

演示如何：
1. 提取单个文件的连接性特征
2. 读取和分析结果
3. 进行简单的可视化

使用方法:
    python example_simple_usage.py --input_file "your_data.set"
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_features(input_file, output_dir=None):
    """
    步骤1: 提取连接性特征
    """
    print("\n" + "="*60)
    print("步骤1: 提取连接性特征")
    print("="*60)
    
    if output_dir is None:
        output_dir = input_file.replace('.set', '_connectivity_features')
    
    # 调用特征提取脚本
    cmd = f'python extract_connectivity_features.py --input_file "{input_file}" --output_dir "{output_dir}"'
    print(f"运行命令: {cmd}")
    
    ret = os.system(cmd)
    if ret != 0:
        print("错误: 特征提取失败")
        return None
    
    print(f"\n✓ 特征已保存到: {output_dir}")
    return output_dir


def analyze_results(feature_dir):
    """
    步骤2: 读取和分析结果
    """
    print("\n" + "="*60)
    print("步骤2: 读取和分析结果")
    print("="*60)
    
    # 读取标量特征
    scalar_file = os.path.join(feature_dir, 'scalar_features.csv')
    if not os.path.exists(scalar_file):
        print(f"错误: 找不到 {scalar_file}")
        return None
    
    df = pd.read_csv(scalar_file)
    print(f"\n加载了 {len(df)} 个片段的特征")
    print(f"特征维度: {df.shape}")
    
    # 显示基本统计
    print("\n标量特征统计:")
    print("-"*60)
    
    # 图网络指标
    graph_cols = [col for col in df.columns if 'graph_' in col]
    if graph_cols:
        print(f"\n找到 {len(graph_cols)} 个图网络指标")
        print("\n示例指标 (前5个):")
        for col in graph_cols[:5]:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 动态连接指标
    dfc_cols = [col for col in df.columns if 'dfc_' in col]
    if dfc_cols:
        print(f"\n找到 {len(dfc_cols)} 个动态连接指标")
        print("\n动态连接指标:")
        for col in dfc_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")
    
    # 读取一个连接矩阵示例
    print("\n连接矩阵示例:")
    print("-"*60)
    
    matrix_file = os.path.join(feature_dir, 'connectivity_matrices_seg000.npz')
    if os.path.exists(matrix_file):
        data = np.load(matrix_file)
        print(f"\n可用的连接性指标 ({len(data.keys())} 个):")
        for idx, key in enumerate(list(data.keys())[:10]):  # 只显示前10个
            matrix = data[key]
            print(f"  {idx+1}. {key}: shape {matrix.shape}")
        
        if len(data.keys()) > 10:
            print(f"  ... 还有 {len(data.keys()) - 10} 个")
        
        # 分析Pearson相关矩阵
        if 'pearson_corr' in data:
            pearson = data['pearson_corr']
            n = pearson.shape[0]
            triu_indices = np.triu_indices(n, k=1)
            corr_values = pearson[triu_indices]
            
            print(f"\nPearson相关统计:")
            print(f"  平均相关: {np.mean(corr_values):.4f}")
            print(f"  标准差: {np.std(corr_values):.4f}")
            print(f"  最大相关: {np.max(corr_values):.4f}")
            print(f"  最小相关: {np.min(corr_values):.4f}")
    
    return df


def simple_visualization(feature_dir, df):
    """
    步骤3: 简单的可视化
    """
    print("\n" + "="*60)
    print("步骤3: 生成可视化")
    print("="*60)
    
    output_dir = os.path.join(feature_dir, 'simple_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 图网络指标时间演化
    graph_cols = [col for col in df.columns if 'graph_degree_mean' in col or 'graph_clustering_mean' in col]
    
    if graph_cols:
        plt.figure(figsize=(12, 6))
        
        for col in graph_cols:
            plt.plot(df['start_time'], df[col], marker='o', label=col, alpha=0.7)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Graph Metrics Over Time', fontsize=14)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, 'graph_metrics_time.png')
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"✓ 保存: {plot_file}")
    
    # 2. 动态连接指标
    dfc_cols = [col for col in df.columns if 'dfc_' in col]
    
    if dfc_cols:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for idx, col in enumerate(dfc_cols):
            if idx < len(axes):
                axes[idx].plot(df['start_time'], df[col], marker='o', color='steelblue', linewidth=2)
                axes[idx].set_xlabel('Time (s)')
                axes[idx].set_ylabel(col.replace('dfc_', '').replace('_', ' ').title())
                axes[idx].set_title(col.replace('dfc_', '').replace('_', ' ').title())
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'dfc_metrics.png')
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"✓ 保存: {plot_file}")
    
    # 3. 连接矩阵热图
    matrix_file = os.path.join(feature_dir, 'connectivity_matrices_seg000.npz')
    if os.path.exists(matrix_file):
        data = np.load(matrix_file)
        
        # 选择几个关键矩阵绘制
        matrices_to_plot = {
            'Pearson Correlation': 'pearson_corr',
            'PLV (Alpha)': 'plv_alpha',
            'wPLI (Alpha)': 'wpli_alpha',
            'Coherence (Alpha)': 'coherence_alpha'
        }
        
        available = {k: v for k, v in matrices_to_plot.items() if v in data}
        
        if available:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for idx, (title, key) in enumerate(available.items()):
                if idx < len(axes):
                    matrix = data[key]
                    im = axes[idx].imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                    axes[idx].set_title(title)
                    axes[idx].set_xlabel('Channels')
                    axes[idx].set_ylabel('Channels')
                    plt.colorbar(im, ax=axes[idx])
            
            # 隐藏多余的子图
            for idx in range(len(available), len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle('Connectivity Matrices (First Segment)', fontsize=16)
            plt.tight_layout()
            
            plot_file = os.path.join(output_dir, 'connectivity_matrices.png')
            plt.savefig(plot_file, dpi=150)
            plt.close()
            print(f"✓ 保存: {plot_file}")
    
    print(f"\n✓ 所有图表已保存到: {output_dir}")


def generate_report(feature_dir, df):
    """
    步骤4: 生成简单报告
    """
    print("\n" + "="*60)
    print("步骤4: 生成报告")
    print("="*60)
    
    report_file = os.path.join(feature_dir, 'simple_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("EEG连接性分析简报\n")
        f.write("="*60 + "\n\n")
        
        # 基本信息
        f.write("1. 基本信息\n")
        f.write("-"*60 + "\n")
        f.write(f"片段数量: {len(df)}\n")
        f.write(f"总时长: {df['duration'].sum():.2f} 秒\n")
        f.write(f"通道数: {df['n_channels'].iloc[0] if len(df) > 0 else 'N/A'}\n\n")
        
        # 关键指标
        f.write("2. 关键指标平均值\n")
        f.write("-"*60 + "\n")
        
        # 选择几个关键指标
        key_metrics = [
            'pearson_corr_graph_degree_mean',
            'pearson_corr_graph_clustering_mean',
            'pearson_corr_graph_modularity',
            'dfc_variance',
            'dfc_state_switches'
        ]
        
        for metric in key_metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                f.write(f"{metric}:\n")
                f.write(f"  平均: {mean_val:.4f}\n")
                f.write(f"  标准差: {std_val:.4f}\n")
                f.write(f"  范围: [{df[metric].min():.4f}, {df[metric].max():.4f}]\n\n")
        
        # 连接性矩阵信息
        f.write("3. 连接性矩阵\n")
        f.write("-"*60 + "\n")
        
        matrix_file = os.path.join(feature_dir, 'connectivity_matrices_seg000.npz')
        if os.path.exists(matrix_file):
            data = np.load(matrix_file)
            f.write(f"可用的连接性指标: {len(data.keys())}\n")
            f.write("指标列表:\n")
            for key in sorted(data.keys()):
                f.write(f"  - {key}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("报告结束\n")
        f.write("="*60 + "\n")
    
    print(f"✓ 报告已保存: {report_file}")
    
    # 打印到屏幕
    with open(report_file, 'r', encoding='utf-8') as f:
        print("\n" + f.read())


def main():
    parser = argparse.ArgumentParser(
        description="简单的EEG连接性分析示例",
        epilog="""
示例用法:
  python example_simple_usage.py --input_file "data.set"
  python example_simple_usage.py --input_file "data.set" --skip_extract
        """
    )
    
    parser.add_argument('--input_file', required=True, help="输入.set文件")
    parser.add_argument('--output_dir', help="输出目录（可选）")
    parser.add_argument('--skip_extract', action='store_true', 
                       help="跳过特征提取（假设已提取）")
    
    args = parser.parse_args()
    
    print("="*60)
    print("EEG连接性分析 - 简单示例")
    print("="*60)
    
    # 确定特征目录
    if args.output_dir:
        feature_dir = args.output_dir
    else:
        feature_dir = args.input_file.replace('.set', '_connectivity_features')
    
    # 步骤1: 提取特征（可跳过）
    if not args.skip_extract:
        result = extract_features(args.input_file, feature_dir)
        if result is None:
            print("\n❌ 特征提取失败")
            return 1
    else:
        print(f"\n跳过特征提取，使用已有数据: {feature_dir}")
    
    # 检查特征目录是否存在
    if not os.path.exists(feature_dir):
        print(f"\n❌ 特征目录不存在: {feature_dir}")
        return 1
    
    # 步骤2: 分析结果
    df = analyze_results(feature_dir)
    if df is None:
        print("\n❌ 结果分析失败")
        return 1
    
    # 步骤3: 可视化
    try:
        simple_visualization(feature_dir, df)
    except Exception as e:
        print(f"\n⚠ 可视化失败: {e}")
    
    # 步骤4: 生成报告
    try:
        generate_report(feature_dir, df)
    except Exception as e:
        print(f"\n⚠ 报告生成失败: {e}")
    
    # 完成
    print("\n" + "="*60)
    print("✅ 分析完成！")
    print("="*60)
    print(f"\n输出目录: {feature_dir}")
    print("\n后续建议:")
    print("  1. 查看生成的图表和报告")
    print("  2. 运行 analyze_connectivity_features.py 进行更详细的分析")
    print("  3. 使用标量特征进行机器学习分类")
    print("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

