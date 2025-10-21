#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_segment_distribution.py

统计所有connectivity_features数据中片段长度的分布情况
- 患者层面统计
- 测试层面统计  
- 通道组合层面统计
- 总体统计

使用方法:
    python analyze_segment_distribution.py --features_root "E:\output\connectivity_features"
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter


def extract_info_from_path(features_dir, features_root):
    """
    从特征文件夹路径提取信息
    
    返回:
        test_name, patient_name, channel_combo
    """
    features_path = Path(features_dir)
    features_root_path = Path(features_root)
    
    try:
        rel_path = features_path.relative_to(features_root_path)
        path_parts = rel_path.parts
        
        # 路径结构：测试名称/患者名称/xxx_connectivity_features
        if len(path_parts) >= 3:
            test_name = path_parts[0]
            patient_name = path_parts[1]
        elif len(path_parts) >= 2:
            test_name = path_parts[0]
            patient_name = "未知患者"
        else:
            test_name = "未知测试"
            patient_name = "未知患者"
        
        # 从文件夹名提取通道组合
        folder_name = features_path.name
        # 例如: SZ1_merged_F8_Fp2_connectivity_features
        import re
        pattern = r'_merged_(.+)_connectivity_features'
        match = re.search(pattern, folder_name)
        
        if match:
            channel_combo = match.group(1).replace('_', ',')
        else:
            channel_combo = "未知通道"
        
        return test_name, patient_name, channel_combo
        
    except Exception as e:
        return "未知测试", "未知患者", "未知通道"


def collect_all_segments(features_root):
    """
    收集所有特征文件夹中的片段信息
    
    返回:
        all_segments: list of dict
    """
    all_segments = []
    features_root_path = Path(features_root)
    
    print("扫描特征文件夹...")
    print("-"*80)
    
    # 查找所有 *_connectivity_features 文件夹
    feature_dirs = list(features_root_path.rglob('*_connectivity_features'))
    
    print(f"找到 {len(feature_dirs)} 个特征文件夹\n")
    
    for features_dir in feature_dirs:
        if not features_dir.is_dir():
            continue
        
        # 查找 scalar_features.csv
        csv_file = features_dir / 'scalar_features.csv'
        
        if not csv_file.exists():
            print(f"⚠ 未找到 scalar_features.csv: {features_dir.name}")
            continue
        
        # 提取测试名称、患者名称、通道组合
        test_name, patient_name, channel_combo = extract_info_from_path(features_dir, features_root)
        
        # 读取CSV
        try:
            df = pd.read_csv(csv_file)
            
            # 提取每个片段的信息
            for _, row in df.iterrows():
                all_segments.append({
                    'test_name': test_name,
                    'patient_name': patient_name,
                    'channel_combo': channel_combo,
                    'features_dir': str(features_dir.relative_to(features_root_path)).replace('\\', '/'),
                    'segment_id': row.get('segment_id', 0),
                    'start_time': row.get('start_time', 0),
                    'end_time': row.get('end_time', 0),
                    'duration': row.get('duration', 0),
                    'n_channels': row.get('n_channels', 0)
                })
            
            print(f"✓ [{test_name}] {patient_name} - {channel_combo}: {len(df)} 个片段")
            
        except Exception as e:
            print(f"✗ 读取失败 {csv_file}: {e}")
            continue
    
    return all_segments


def analyze_distribution(all_segments, output_dir):
    """
    分析片段长度分布
    
    参数:
        all_segments: list of dict
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("统计分析")
    print("="*80)
    
    df = pd.DataFrame(all_segments)
    
    # ========================================================================
    # 1. 总体统计
    # ========================================================================
    print("\n【1. 总体统计】")
    print("-"*80)
    print(f"总片段数: {len(df)}")
    print(f"总时长: {df['duration'].sum():.2f} 秒 ({df['duration'].sum()/60:.2f} 分钟)")
    print(f"平均片段长度: {df['duration'].mean():.2f} ± {df['duration'].std():.2f} 秒")
    print(f"片段长度范围: [{df['duration'].min():.2f}, {df['duration'].max():.2f}] 秒")
    print(f"\n片段长度分位数:")
    for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"  {int(q*100)}%: {df['duration'].quantile(q):.2f} 秒")
    
    # ========================================================================
    # 2. 按测试名称统计
    # ========================================================================
    print("\n【2. 按测试分组统计】")
    print("-"*80)
    
    test_stats = df.groupby('test_name').agg({
        'duration': ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'patient_name': 'nunique'
    }).round(2)
    
    test_stats.columns = ['片段数', '总时长(秒)', '平均长度', '标准差', '最小值', '最大值', '患者数']
    test_stats['总时长(分钟)'] = (test_stats['总时长(秒)'] / 60).round(2)
    
    print(test_stats.to_string())
    
    # 保存到CSV
    test_stats.to_csv(os.path.join(output_dir, 'test_statistics.csv'))
    
    # ========================================================================
    # 3. 按患者统计
    # ========================================================================
    print("\n【3. 按患者分组统计（前20个）】")
    print("-"*80)
    
    patient_stats = df.groupby(['test_name', 'patient_name']).agg({
        'duration': ['count', 'sum', 'mean', 'std'],
        'channel_combo': 'nunique'
    }).round(2)
    
    patient_stats.columns = ['片段数', '总时长(秒)', '平均长度', '标准差', '通道组合数']
    patient_stats['总时长(分钟)'] = (patient_stats['总时长(秒)'] / 60).round(2)
    patient_stats = patient_stats.sort_values('总时长(秒)', ascending=False)
    
    print(patient_stats.head(20).to_string())
    if len(patient_stats) > 20:
        print(f"\n... 还有 {len(patient_stats) - 20} 个患者")
    
    # 保存完整统计
    patient_stats.to_csv(os.path.join(output_dir, 'patient_statistics.csv'))
    
    # ========================================================================
    # 4. 按通道组合统计
    # ========================================================================
    print("\n【4. 按通道组合统计（前15个）】")
    print("-"*80)
    
    channel_stats = df.groupby('channel_combo').agg({
        'duration': ['count', 'sum', 'mean', 'std'],
        'patient_name': 'nunique'
    }).round(2)
    
    channel_stats.columns = ['片段数', '总时长(秒)', '平均长度', '标准差', '患者数']
    channel_stats['总时长(分钟)'] = (channel_stats['总时长(秒)'] / 60).round(2)
    channel_stats = channel_stats.sort_values('片段数', ascending=False)
    
    print(channel_stats.head(15).to_string())
    if len(channel_stats) > 15:
        print(f"\n... 还有 {len(channel_stats) - 15} 个通道组合")
    
    channel_stats.to_csv(os.path.join(output_dir, 'channel_combination_statistics.csv'))
    
    # ========================================================================
    # 5. 片段长度分布
    # ========================================================================
    print("\n【5. 片段长度分布】")
    print("-"*80)
    
    # 创建长度区间
    bins = [0, 10, 20, 30, 40, 50, 60, 90, 120, float('inf')]
    labels = ['0-10s', '10-20s', '20-30s', '30-40s', '40-50s', '50-60s', '60-90s', '90-120s', '>120s']
    
    df['duration_bin'] = pd.cut(df['duration'], bins=bins, labels=labels)
    
    duration_dist = df['duration_bin'].value_counts().sort_index()
    print("\n片段长度区间分布:")
    for bin_label, count in duration_dist.items():
        percentage = count / len(df) * 100
        print(f"  {bin_label}: {count:4d} 个片段 ({percentage:5.2f}%)")
    
    return df


def create_visualizations(df, output_dir):
    """
    创建可视化图表
    
    参数:
        df: DataFrame 所有片段数据
        output_dir: 输出目录
    """
    print("\n" + "="*80)
    print("生成可视化图表")
    print("="*80)
    
    # 设置中文字体（可选）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ========================================================================
    # 1. 片段长度直方图（总体）
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df['duration'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Segment Duration (seconds)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Overall Segment Duration Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    mean_dur = df['duration'].mean()
    median_dur = df['duration'].median()
    ax.axvline(mean_dur, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dur:.1f}s')
    ax.axvline(median_dur, color='green', linestyle='--', linewidth=2, label=f'Median: {median_dur:.1f}s')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_histogram_overall.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ 保存: duration_histogram_overall.png")
    
    # ========================================================================
    # 2. 按测试分组的箱线图
    # ========================================================================
    if df['test_name'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        test_names = df['test_name'].unique()
        data_by_test = [df[df['test_name'] == test]['duration'].values for test in test_names]
        
        bp = ax.boxplot(data_by_test, labels=test_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Test Name', fontsize=12)
        ax.set_ylabel('Segment Duration (seconds)', fontsize=12)
        ax.set_title('Segment Duration by Test', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'duration_boxplot_by_test.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("✓ 保存: duration_boxplot_by_test.png")
    
    # ========================================================================
    # 3. 按患者的片段数和总时长
    # ========================================================================
    patient_summary = df.groupby(['test_name', 'patient_name']).agg({
        'duration': ['count', 'sum']
    }).round(2)
    patient_summary.columns = ['片段数', '总时长(秒)']
    patient_summary = patient_summary.sort_values('总时长(秒)', ascending=False).head(20)
    
    if len(patient_summary) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 3.1 片段数
        patient_labels = [f"[{t}] {p}" for t, p in patient_summary.index]
        ax1.barh(range(len(patient_summary)), patient_summary['片段数'], color='coral', alpha=0.7)
        ax1.set_yticks(range(len(patient_summary)))
        ax1.set_yticklabels(patient_labels, fontsize=8)
        ax1.set_xlabel('Number of Segments', fontsize=11)
        ax1.set_title('Top 20 Patients by Segment Count', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
        
        # 3.2 总时长
        ax2.barh(range(len(patient_summary)), patient_summary['总时长(秒)'], color='skyblue', alpha=0.7)
        ax2.set_yticks(range(len(patient_summary)))
        ax2.set_yticklabels(patient_labels, fontsize=8)
        ax2.set_xlabel('Total Duration (seconds)', fontsize=11)
        ax2.set_title('Top 20 Patients by Total Duration', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'patient_segment_summary.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("✓ 保存: patient_segment_summary.png")
    
    # ========================================================================
    # 4. 通道组合的片段长度分布
    # ========================================================================
    top_channels = df['channel_combo'].value_counts().head(10).index.tolist()
    
    if len(top_channels) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        data_by_channel = [df[df['channel_combo'] == ch]['duration'].values for ch in top_channels]
        
        bp = ax.boxplot(data_by_channel, labels=top_channels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Channel Combination', fontsize=12)
        ax.set_ylabel('Segment Duration (seconds)', fontsize=12)
        ax.set_title('Segment Duration by Top 10 Channel Combinations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'duration_by_channel_combo.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("✓ 保存: duration_by_channel_combo.png")
    
    # ========================================================================
    # 5. 片段长度分布区间图
    # ========================================================================
    bins = [0, 10, 20, 30, 40, 50, 60, 90, 120, float('inf')]
    labels = ['0-10s', '10-20s', '20-30s', '30-40s', '40-50s', '50-60s', '60-90s', '90-120s', '>120s']
    df['duration_bin'] = pd.cut(df['duration'], bins=bins, labels=labels)
    
    duration_counts = df['duration_bin'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(duration_counts)), duration_counts.values, color='mediumpurple', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(duration_counts)))
    ax.set_xticklabels(duration_counts.index, rotation=45, ha='right')
    ax.set_xlabel('Duration Range', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Segment Duration Distribution by Range', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(duration_counts.values):
        percentage = v / len(df) * 100
        ax.text(i, v, f'{v}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_distribution_bins.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ 保存: duration_distribution_bins.png")
    
    # ========================================================================
    # 6. 累积分布曲线
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_durations = np.sort(df['duration'].values)
    cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100
    
    ax.plot(sorted_durations, cumulative, linewidth=2, color='darkblue')
    ax.set_xlabel('Segment Duration (seconds)', fontsize=12)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax.set_title('Cumulative Distribution of Segment Duration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加参考线
    for p in [25, 50, 75, 90]:
        dur_at_p = np.percentile(df['duration'], p)
        ax.axhline(p, color='red', linestyle='--', alpha=0.3)
        ax.axvline(dur_at_p, color='red', linestyle='--', alpha=0.3)
        ax.text(dur_at_p, p, f' {p}%: {dur_at_p:.1f}s', fontsize=9, va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_cumulative_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ 保存: duration_cumulative_distribution.png")
    
    # ========================================================================
    # 7. 热力图：测试 x 患者的片段数
    # ========================================================================
    pivot_count = df.pivot_table(
        values='duration', 
        index='patient_name', 
        columns='test_name', 
        aggfunc='count',
        fill_value=0
    )
    
    if len(pivot_count) > 0 and len(pivot_count.columns) > 0:
        fig, ax = plt.subplots(figsize=(max(10, len(pivot_count.columns)*2), 
                                       max(8, len(pivot_count)*0.3)))
        
        sns.heatmap(pivot_count, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Segment Count'}, ax=ax)
        ax.set_xlabel('Test Name', fontsize=12, fontweight='bold')
        ax.set_ylabel('Patient Name', fontsize=12, fontweight='bold')
        ax.set_title('Segment Count Heatmap (Patient x Test)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmap_patient_test_count.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("✓ 保存: heatmap_patient_test_count.png")
    
    # ========================================================================
    # 8. 热力图：测试 x 患者的总时长
    # ========================================================================
    pivot_duration = df.pivot_table(
        values='duration',
        index='patient_name',
        columns='test_name',
        aggfunc='sum',
        fill_value=0
    )
    
    if len(pivot_duration) > 0 and len(pivot_duration.columns) > 0:
        fig, ax = plt.subplots(figsize=(max(10, len(pivot_duration.columns)*2), 
                                       max(8, len(pivot_duration)*0.3)))
        
        sns.heatmap(pivot_duration, annot=True, fmt='.1f', cmap='Blues',
                   cbar_kws={'label': 'Total Duration (seconds)'}, ax=ax)
        ax.set_xlabel('Test Name', fontsize=12, fontweight='bold')
        ax.set_ylabel('Patient Name', fontsize=12, fontweight='bold')
        ax.set_title('Total Duration Heatmap (Patient x Test)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmap_patient_test_duration.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("✓ 保存: heatmap_patient_test_duration.png")


def generate_summary_report(df, output_dir):
    """
    生成文本报告
    
    参数:
        df: DataFrame 所有片段数据
        output_dir: 输出目录
    """
    report_file = os.path.join(output_dir, 'segment_distribution_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EEG片段长度分布分析报告\n")
        f.write("="*80 + "\n\n")
        
        # 1. 总体统计
        f.write("1. 总体统计\n")
        f.write("-"*80 + "\n")
        f.write(f"总片段数: {len(df)}\n")
        f.write(f"总时长: {df['duration'].sum():.2f} 秒 ({df['duration'].sum()/60:.2f} 分钟, {df['duration'].sum()/3600:.2f} 小时)\n")
        f.write(f"平均片段长度: {df['duration'].mean():.2f} ± {df['duration'].std():.2f} 秒\n")
        f.write(f"片段长度中位数: {df['duration'].median():.2f} 秒\n")
        f.write(f"片段长度范围: [{df['duration'].min():.2f}, {df['duration'].max():.2f}] 秒\n\n")
        
        f.write("分位数:\n")
        for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            f.write(f"  {int(q*100):2d}%: {df['duration'].quantile(q):6.2f} 秒\n")
        f.write("\n")
        
        # 2. 基本信息
        f.write("2. 数据集概况\n")
        f.write("-"*80 + "\n")
        f.write(f"测试数量: {df['test_name'].nunique()}\n")
        f.write(f"患者数量: {df['patient_name'].nunique()}\n")
        f.write(f"通道组合数量: {df['channel_combo'].nunique()}\n\n")
        
        # 3. 按测试统计
        f.write("3. 按测试统计\n")
        f.write("-"*80 + "\n")
        test_stats = df.groupby('test_name').agg({
            'duration': ['count', 'sum', 'mean'],
            'patient_name': 'nunique'
        }).round(2)
        test_stats.columns = ['片段数', '总时长(秒)', '平均长度', '患者数']
        f.write(test_stats.to_string() + "\n\n")
        
        # 4. 片段长度区间分布
        f.write("4. 片段长度区间分布\n")
        f.write("-"*80 + "\n")
        duration_dist = df['duration_bin'].value_counts().sort_index()
        for bin_label, count in duration_dist.items():
            percentage = count / len(df) * 100
            bar = '█' * int(percentage / 2)
            f.write(f"  {bin_label:>10s}: {count:5d} ({percentage:5.2f}%) {bar}\n")
        f.write("\n")
        
        # 5. Top患者
        f.write("5. Top 10 患者（按总时长）\n")
        f.write("-"*80 + "\n")
        patient_stats = df.groupby(['test_name', 'patient_name']).agg({
            'duration': ['count', 'sum']
        }).round(2)
        patient_stats.columns = ['片段数', '总时长(秒)']
        patient_stats['总时长(分钟)'] = (patient_stats['总时长(秒)'] / 60).round(2)
        patient_stats = patient_stats.sort_values('总时长(秒)', ascending=False).head(10)
        f.write(patient_stats.to_string() + "\n\n")
        
        # 6. Top通道组合
        f.write("6. Top 10 通道组合（按片段数）\n")
        f.write("-"*80 + "\n")
        channel_stats = df.groupby('channel_combo').agg({
            'duration': ['count', 'sum', 'mean']
        }).round(2)
        channel_stats.columns = ['片段数', '总时长(秒)', '平均长度']
        channel_stats = channel_stats.sort_values('片段数', ascending=False).head(10)
        f.write(channel_stats.to_string() + "\n\n")
        
        f.write("="*80 + "\n")
        f.write("报告结束\n")
        f.write("="*80 + "\n")
    
    print(f"✓ 保存: segment_distribution_report.txt")
    
    # 同时输出到控制台
    print("\n" + "="*80)
    print("报告摘要")
    print("="*80)
    with open(report_file, 'r', encoding='utf-8') as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(
        description="分析EEG片段长度分布",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python analyze_segment_distribution.py --features_root "E:\\output\\connectivity_features"
  
  # 指定输出目录
  python analyze_segment_distribution.py \\
      --features_root "E:\\output\\connectivity_features" \\
      --output_dir "E:\\output\\segment_analysis"

输出文件:
  - segment_distribution_report.txt         # 文本报告
  - test_statistics.csv                     # 测试级别统计
  - patient_statistics.csv                  # 患者级别统计  
  - channel_combination_statistics.csv      # 通道组合统计
  - all_segments.csv                        # 所有片段详细信息
  - duration_histogram_overall.png          # 总体直方图
  - duration_boxplot_by_test.png           # 按测试箱线图
  - patient_segment_summary.png            # 患者汇总
  - duration_by_channel_combo.png          # 按通道组合
  - heatmap_patient_test_count.png         # 热力图（片段数）
  - heatmap_patient_test_duration.png      # 热力图（时长）
        """
    )
    
    parser.add_argument(
        '--features_root',
        required=True,
        help="特征文件根目录（包含*_connectivity_features文件夹）"
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help="输出目录，默认为features_root/segment_analysis"
    )
    
    args = parser.parse_args()
    
    # 检查目录
    if not os.path.exists(args.features_root):
        print(f"错误: 特征根目录不存在: {args.features_root}")
        return 1
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.features_root, 'segment_analysis')
    
    print("="*80)
    print("EEG片段长度分布分析")
    print("="*80)
    print(f"特征根目录: {args.features_root}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    # 收集所有片段数据
    all_segments = collect_all_segments(args.features_root)
    
    if not all_segments:
        print("\n❌ 未找到任何片段数据")
        return 1
    
    # 分析分布
    df = analyze_distribution(all_segments, args.output_dir)
    
    # 保存详细数据
    df.to_csv(os.path.join(args.output_dir, 'all_segments.csv'), index=False)
    print(f"\n✓ 所有片段数据已保存: all_segments.csv")
    
    # 创建可视化
    create_visualizations(df, args.output_dir)
    
    # 生成报告
    generate_summary_report(df, args.output_dir)
    
    print("\n" + "="*80)
    print("✅ 分析完成！")
    print(f"所有结果已保存到: {args.output_dir}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    
    # 如果不提供命令行参数，使用默认值
    if len(sys.argv) == 1:
        print("使用默认参数运行...")
        sys.argv.extend([
            '--features_root', r'E:\output\connectivity_features'
        ])
    
    sys.exit(main())

