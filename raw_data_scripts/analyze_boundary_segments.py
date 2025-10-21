#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_boundary_segments.py

统计.set文件中boundary事件之间的片段长度分布
- 读取每个.set文件的annotations
- 提取boundary事件
- 计算boundary之间的片段长度
- 患者层面和总体层面的统计

使用方法:
    python analyze_boundary_segments.py --data_root "E:\DataSet\EEG\dataset"
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from pathlib import Path
from collections import defaultdict, Counter

# 抑制MNE警告
mne.set_log_level('ERROR')


def extract_boundary_segments(set_file):
    """
    提取单个.set文件中boundary之间的片段信息
    
    参数:
        set_file: .set文件路径
    
    返回:
        segments: list of dict，每个片段的信息
    """
    try:
        # 读取EEG文件
        raw = mne.io.read_raw_eeglab(set_file, preload=False, verbose='ERROR')
        
        total_duration = raw.times[-1]  # 总时长
        sfreq = raw.info['sfreq']
        
        # 获取annotations
        annotations = raw.annotations
        
        # 提取所有boundary事件
        boundaries = []
        for ann in annotations:
            if ann['description'] == 'boundary':
                boundaries.append({
                    'onset': ann['onset'],
                    'duration': ann['duration']
                })
        
        # 如果没有boundary，整个文件是一个片段
        if len(boundaries) == 0:
            return [{
                'start': 0,
                'end': total_duration,
                'duration': total_duration,
                'segment_type': 'complete'
            }]
        
        # 计算boundary之间的片段
        segments = []
        
        # 按onset排序
        boundaries = sorted(boundaries, key=lambda x: x['onset'])
        
        # 第一个片段（从0到第一个boundary）
        if boundaries[0]['onset'] > 0:
            segments.append({
                'start': 0,
                'end': boundaries[0]['onset'],
                'duration': boundaries[0]['onset'],
                'segment_type': 'before_first_boundary'
            })
        
        # 中间的片段（boundary之间）
        for i in range(len(boundaries) - 1):
            # 当前boundary的结束时间
            current_end = boundaries[i]['onset'] + boundaries[i]['duration']
            # 下一个boundary的开始时间
            next_start = boundaries[i+1]['onset']
            
            # 如果有间隔
            if next_start > current_end:
                segments.append({
                    'start': current_end,
                    'end': next_start,
                    'duration': next_start - current_end,
                    'segment_type': 'between_boundaries'
                })
        
        # 最后一个片段（最后一个boundary之后到文件结尾）
        last_boundary_end = boundaries[-1]['onset'] + boundaries[-1]['duration']
        if last_boundary_end < total_duration:
            segments.append({
                'start': last_boundary_end,
                'end': total_duration,
                'duration': total_duration - last_boundary_end,
                'segment_type': 'after_last_boundary'
            })
        
        return segments
        
    except Exception as e:
        print(f"  ✗ 读取失败: {e}")
        return []


def extract_info_from_path(set_file, data_root):
    """
    从.set文件路径提取测试名称和患者名称
    
    参数:
        set_file: .set文件路径
        data_root: 数据根目录
    
    返回:
        test_name, patient_name
    """
    set_path = Path(set_file)
    data_root_path = Path(data_root)
    
    try:
        rel_path = set_path.relative_to(data_root_path)
        path_parts = rel_path.parts
        
        # 路径结构：测试名称/患者名称/xxx.set
        if len(path_parts) >= 3:
            test_name = path_parts[0]
            patient_name = path_parts[1]
        elif len(path_parts) >= 2:
            test_name = path_parts[0]
            patient_name = set_path.stem
        else:
            test_name = "未知测试"
            patient_name = set_path.stem
        
        return test_name, patient_name
        
    except Exception as e:
        return "未知测试", set_path.stem


def collect_all_boundary_segments(data_root, pattern="*_merged_*.set"):
    """
    收集所有.set文件中的boundary片段信息
    
    参数:
        data_root: 数据根目录
        pattern: 文件匹配模式
    
    返回:
        all_segments: list of dict
    """
    all_segments = []
    data_root_path = Path(data_root)
    
    print("扫描.set文件...")
    print("-"*80)
    
    # 查找所有匹配的.set文件
    set_files = list(data_root_path.rglob(pattern))
    
    print(f"找到 {len(set_files)} 个.set文件\n")
    
    for set_file in set_files:
        # 提取测试名称和患者名称
        test_name, patient_name = extract_info_from_path(set_file, data_root)
        
        print(f"处理: [{test_name}] {patient_name} - {set_file.name}")
        
        # 提取boundary片段
        segments = extract_boundary_segments(set_file)
        
        if not segments:
            print(f"  ⚠ 未找到片段")
            continue
        
        # 添加文件信息
        for seg in segments:
            seg['test_name'] = test_name
            seg['patient_name'] = patient_name
            seg['file_name'] = set_file.name
            seg['file_path'] = str(set_file.relative_to(data_root_path)).replace('\\', '/')
            all_segments.append(seg)
        
        print(f"  ✓ 提取 {len(segments)} 个片段，总时长: {sum([s['duration'] for s in segments]):.2f}秒")
    
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
    
    # 先创建所有的分组列（在使用之前）
    # 创建1秒精度的区间
    bins_fine = list(range(0, 61, 1))  # 0-1s, 1-2s, ..., 59-60s
    bins_coarse = [70, 80, 90, 100, 120, 150, 180, 240, 300, float('inf')]
    bins = bins_fine + bins_coarse
    
    labels_fine = [f'{i}-{i+1}s' for i in range(60)]
    labels_coarse = ['60-70s', '70-80s', '80-90s', '90-100s', '100-120s', 
                     '120-150s', '150-180s', '180-240s', '240-300s', '>300s']
    labels = labels_fine + labels_coarse
    
    df['duration_bin'] = pd.cut(df['duration'], bins=bins, labels=labels)
    
    # 创建聚合区间
    bins_overview = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90, 120, 180, float('inf')]
    labels_overview = ['0-5s', '5-10s', '10-15s', '15-20s', '20-25s', '25-30s', 
                      '30-40s', '40-50s', '50-60s', '60-90s', '90-120s', '120-180s', '>180s']
    df['duration_bin_overview'] = pd.cut(df['duration'], bins=bins_overview, labels=labels_overview)
    
    # ========================================================================
    # 1. 总体统计
    # ========================================================================
    print("\n【1. 总体统计】")
    print("-"*80)
    print(f"总片段数: {len(df)}")
    print(f"总文件数: {df['file_name'].nunique()}")
    print(f"总时长: {df['duration'].sum():.2f} 秒 ({df['duration'].sum()/60:.2f} 分钟, {df['duration'].sum()/3600:.2f} 小时)")
    print(f"平均片段长度: {df['duration'].mean():.2f} ± {df['duration'].std():.2f} 秒")
    print(f"片段长度中位数: {df['duration'].median():.2f} 秒")
    print(f"片段长度范围: [{df['duration'].min():.2f}, {df['duration'].max():.2f}] 秒")
    print(f"\n片段长度分位数:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"  {int(q*100):2d}%: {df['duration'].quantile(q):7.2f} 秒")
    
    # ========================================================================
    # 2. 按测试名称统计
    # ========================================================================
    print("\n【2. 按测试分组统计】")
    print("-"*80)
    
    test_stats = df.groupby('test_name').agg({
        'duration': ['count', 'sum', 'mean', 'std', 'median', 'min', 'max'],
        'patient_name': 'nunique',
        'file_name': 'nunique'
    }).round(2)
    
    test_stats.columns = ['片段数', '总时长(秒)', '平均长度', '标准差', '中位数', '最小值', '最大值', '患者数', '文件数']
    test_stats['总时长(分钟)'] = (test_stats['总时长(秒)'] / 60).round(2)
    test_stats['总时长(小时)'] = (test_stats['总时长(秒)'] / 3600).round(2)
    
    print(test_stats.to_string())
    
    # 保存到CSV
    test_stats.to_csv(os.path.join(output_dir, 'test_statistics.csv'))
    
    # ========================================================================
    # 1.5 保存1秒精度的完整分布到CSV
    # ========================================================================
    duration_dist_1s = df['duration_bin'].value_counts().sort_index()
    duration_dist_1s_df = pd.DataFrame({
        'duration_range': duration_dist_1s.index,
        'count': duration_dist_1s.values,
        'percentage': (duration_dist_1s.values / len(df) * 100).round(2)
    })
    duration_dist_1s_df.to_csv(os.path.join(output_dir, 'duration_distribution_1s_precision.csv'), index=False)
    print("\n✓ 1秒精度完整分布已保存: duration_distribution_1s_precision.csv")
    
    # ========================================================================
    # 3. 按患者统计
    # ========================================================================
    print("\n【3. 按患者分组统计（前30个）】")
    print("-"*80)
    
    patient_stats = df.groupby(['test_name', 'patient_name']).agg({
        'duration': ['count', 'sum', 'mean', 'std', 'median', 'min', 'max'],
        'file_name': 'nunique'
    }).round(2)
    
    patient_stats.columns = ['片段数', '总时长(秒)', '平均长度', '标准差', '中位数', '最小值', '最大值', '文件数']
    patient_stats['总时长(分钟)'] = (patient_stats['总时长(秒)'] / 60).round(2)
    patient_stats = patient_stats.sort_values('总时长(秒)', ascending=False)
    
    print(patient_stats.head(30).to_string())
    if len(patient_stats) > 30:
        print(f"\n... 还有 {len(patient_stats) - 30} 个患者")
    
    # 保存完整统计
    patient_stats.to_csv(os.path.join(output_dir, 'patient_statistics.csv'))
    
    # ========================================================================
    # 4. 按文件统计
    # ========================================================================
    print("\n【4. 按文件统计（前20个）】")
    print("-"*80)
    
    file_stats = df.groupby(['test_name', 'patient_name', 'file_name']).agg({
        'duration': ['count', 'sum', 'mean', 'min', 'max']
    }).round(2)
    
    file_stats.columns = ['片段数', '总时长(秒)', '平均长度', '最小值', '最大值']
    file_stats['总时长(分钟)'] = (file_stats['总时长(秒)'] / 60).round(2)
    file_stats = file_stats.sort_values('总时长(秒)', ascending=False)
    
    print(file_stats.head(20).to_string())
    if len(file_stats) > 20:
        print(f"\n... 还有 {len(file_stats) - 20} 个文件")
    
    file_stats.to_csv(os.path.join(output_dir, 'file_statistics.csv'))
    
    # ========================================================================
    # 5. 片段长度分布（1秒精度）
    # ========================================================================
    print("\n【5. 片段长度分布（1秒精度）】")
    print("-"*80)
    
    duration_dist = df['duration_bin'].value_counts().sort_index()
    duration_dist_filtered = duration_dist[duration_dist > 0]
    
    print(f"\n片段长度区间分布（1秒精度）:")
    print(f"有数据的区间总数: {len(duration_dist_filtered)}")
    print("\n最常见的30个区间:")
    total = len(df)
    for bin_label, count in duration_dist_filtered.head(30).items():
        percentage = count / total * 100
        bar = '█' * int(percentage / 2)
        print(f"  {bin_label:>12s}: {count:5d} 个片段 ({percentage:5.2f}%) {bar}")
    
    if len(duration_dist_filtered) > 30:
        print(f"\n... 还有 {len(duration_dist_filtered) - 30} 个区间")
    
    # 显示聚合区间总览
    print("\n聚合区间总览:")
    overview_dist = df['duration_bin_overview'].value_counts().sort_index()
    for bin_label, count in overview_dist.items():
        percentage = count / total * 100
        bar = '█' * int(percentage / 2)
        print(f"  {bin_label:>12s}: {count:5d} 个片段 ({percentage:5.2f}%) {bar}")
    
    # ========================================================================
    # 6. 长片段统计（≥30秒）
    # ========================================================================
    print("\n【6. 长片段统计（≥30秒）】")
    print("-"*80)
    
    long_segments = df[df['duration'] >= 30]
    print(f"长片段数量: {len(long_segments)} ({len(long_segments)/len(df)*100:.2f}%)")
    print(f"长片段总时长: {long_segments['duration'].sum():.2f} 秒 ({long_segments['duration'].sum()/60:.2f} 分钟)")
    print(f"长片段平均长度: {long_segments['duration'].mean():.2f} ± {long_segments['duration'].std():.2f} 秒")
    
    # 按患者统计长片段
    long_by_patient = long_segments.groupby(['test_name', 'patient_name']).agg({
        'duration': ['count', 'sum']
    }).round(2)
    long_by_patient.columns = ['长片段数', '总时长(秒)']
    long_by_patient = long_by_patient.sort_values('总时长(秒)', ascending=False)
    
    print(f"\n各患者的长片段统计（前15个）:")
    print(long_by_patient.head(15).to_string())
    
    long_by_patient.to_csv(os.path.join(output_dir, 'long_segments_by_patient.csv'))
    
    # ========================================================================
    # 7. 短片段统计（<30秒）
    # ========================================================================
    print("\n【7. 短片段统计（<30秒）】")
    print("-"*80)
    
    short_segments = df[df['duration'] < 30]
    print(f"短片段数量: {len(short_segments)} ({len(short_segments)/len(df)*100:.2f}%)")
    print(f"短片段总时长: {short_segments['duration'].sum():.2f} 秒 ({short_segments['duration'].sum()/60:.2f} 分钟)")
    print(f"短片段平均长度: {short_segments['duration'].mean():.2f} ± {short_segments['duration'].std():.2f} 秒")
    
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
    
    # 设置样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ========================================================================
    # 1. 片段长度直方图（总体）
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制直方图
    n, bins, patches = ax.hist(df['duration'], bins=100, color='steelblue', 
                               alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 添加统计线
    mean_dur = df['duration'].mean()
    median_dur = df['duration'].median()
    ax.axvline(mean_dur, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_dur:.1f}s')
    ax.axvline(median_dur, color='green', linestyle='--', linewidth=2, 
              label=f'Median: {median_dur:.1f}s')
    ax.axvline(30, color='orange', linestyle='--', linewidth=2, 
              label='30s threshold')
    
    ax.set_xlabel('Segment Duration (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Overall Segment Duration Distribution (Boundary Intervals)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_histogram_overall.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ 保存: duration_histogram_overall.png")
    
    # ========================================================================
    # 2. 对数尺度直方图（更好地显示长尾分布）
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 过滤掉极小值
    df_filtered = df[df['duration'] > 0.1]
    
    ax.hist(df_filtered['duration'], bins=100, color='purple', 
           alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Segment Duration (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Segment Duration Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加30秒线
    ax.axvline(30, color='red', linestyle='--', linewidth=2, label='30s threshold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_histogram_log.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ 保存: duration_histogram_log.png")
    
    # ========================================================================
    # 3. 按测试分组的箱线图
    # ========================================================================
    if df['test_name'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        test_names = sorted(df['test_name'].unique())
        data_by_test = [df[df['test_name'] == test]['duration'].values for test in test_names]
        
        bp = ax.boxplot(data_by_test, labels=test_names, patch_artist=True, showfliers=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.axhline(30, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='30s threshold')
        ax.set_xlabel('Test Name', fontsize=12, fontweight='bold')
        ax.set_ylabel('Segment Duration (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Segment Duration Distribution by Test', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'duration_boxplot_by_test.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("✓ 保存: duration_boxplot_by_test.png")
    
    # ========================================================================
    # 4. 按患者的片段数和总时长（Top 20）
    # ========================================================================
    patient_summary = df.groupby(['test_name', 'patient_name']).agg({
        'duration': ['count', 'sum', 'mean'],
        'file_name': 'nunique'
    }).round(2)
    patient_summary.columns = ['片段数', '总时长(秒)', '平均长度', '文件数']
    patient_summary = patient_summary.sort_values('总时长(秒)', ascending=False).head(20)
    
    if len(patient_summary) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 4.1 片段数
        patient_labels = [f"[{t}]\n{p}" for t, p in patient_summary.index]
        y_pos = np.arange(len(patient_summary))
        
        ax1.barh(y_pos, patient_summary['片段数'], color='coral', alpha=0.7, edgecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(patient_labels, fontsize=8)
        ax1.set_xlabel('Number of Segments', fontsize=11, fontweight='bold')
        ax1.set_title('Top 20 Patients by Segment Count', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
        
        # 添加数值标签
        for i, v in enumerate(patient_summary['片段数']):
            ax1.text(v, i, f' {v:.0f}', va='center', fontsize=8)
        
        # 4.2 总时长
        ax2.barh(y_pos, patient_summary['总时长(秒)'], color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(patient_labels, fontsize=8)
        ax2.set_xlabel('Total Duration (seconds)', fontsize=11, fontweight='bold')
        ax2.set_title('Top 20 Patients by Total Duration', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()
        
        # 添加数值标签
        for i, v in enumerate(patient_summary['总时长(秒)']):
            ax2.text(v, i, f' {v:.0f}s ({v/60:.1f}min)', va='center', fontsize=7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'patient_segment_summary.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("✓ 保存: patient_segment_summary.png")
    
    # ========================================================================
    # 5. 片段长度分布区间图（1秒精度）
    # ========================================================================
    duration_counts = df['duration_bin'].value_counts().sort_index()
    
    # 只绘制有数据的区间
    duration_counts_filtered = duration_counts[duration_counts > 0]
    
    fig, ax = plt.subplots(figsize=(20, 6))
    x_pos = np.arange(len(duration_counts_filtered))
    bars = ax.bar(x_pos, duration_counts_filtered.values, color='mediumpurple', 
                 alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 给≥30秒的区间使用不同颜色
    bins_labels = duration_counts_filtered.index.tolist()
    for i, label in enumerate(bins_labels):
        # 提取区间起始值
        try:
            start_val = int(label.split('-')[0].replace('s', '').replace('>', ''))
            if start_val >= 30:
                bars[i].set_color('lightgreen')
        except:
            if '>' in label:
                bars[i].set_color('lightgreen')
    
    # 设置x轴刻度（每隔几个显示一次，避免拥挤）
    tick_step = max(1, len(duration_counts_filtered) // 50)  # 最多显示50个刻度
    tick_indices = list(range(0, len(duration_counts_filtered), tick_step))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([bins_labels[i] for i in tick_indices], 
                       rotation=90, ha='center', fontsize=7)
    
    ax.set_xlabel('Duration Range (1-second precision)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Segment Duration Distribution by Range (1s bins, 0-60s)\n(Purple: <30s, Green: ≥30s)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加30秒分界线
    try:
        idx_30 = bins_labels.index('30-31s')
        ax.axvline(idx_30, color='red', linestyle='--', linewidth=2, 
                  alpha=0.7, label='30s threshold')
        ax.legend(fontsize=10)
    except:
        pass
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_distribution_bins_1s.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ 保存: duration_distribution_bins_1s.png")
    
    # ========================================================================
    # 5.2 聚合区间图（更清晰的总览）
    # ========================================================================
    # 创建聚合区间用于总览
    bins_overview = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90, 120, 180, float('inf')]
    labels_overview = ['0-5s', '5-10s', '10-15s', '15-20s', '20-25s', '25-30s', 
                      '30-40s', '40-50s', '50-60s', '60-90s', '90-120s', '120-180s', '>180s']
    df['duration_bin_overview'] = pd.cut(df['duration'], bins=bins_overview, labels=labels_overview)
    
    overview_counts = df['duration_bin_overview'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(overview_counts))
    bars = ax.bar(x_pos, overview_counts.values, alpha=0.7, edgecolor='black', linewidth=1)
    
    # 着色：<30s用紫色，≥30s用绿色
    for i, label in enumerate(overview_counts.index):
        start_val = int(label.split('-')[0].replace('s', '').replace('>', ''))
        if start_val >= 30:
            bars[i].set_color('lightgreen')
        else:
            bars[i].set_color('mediumpurple')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(overview_counts.index, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Duration Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Segment Duration Distribution Overview\n(Purple: <30s, Green: ≥30s)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(overview_counts.values):
        percentage = v / len(df) * 100
        ax.text(i, v, f'{v}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_distribution_overview.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ 保存: duration_distribution_overview.png")
    
    # ========================================================================
    # 6. 累积分布曲线
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 7))
    
    sorted_durations = np.sort(df['duration'].values)
    cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100
    
    ax.plot(sorted_durations, cumulative, linewidth=2.5, color='darkblue', label='Cumulative Distribution')
    ax.set_xlabel('Segment Duration (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Distribution of Segment Duration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加30秒线
    ax.axvline(30, color='red', linestyle='--', linewidth=2, alpha=0.7, label='30s threshold')
    
    # 计算30秒以下的百分比
    pct_below_30 = (df['duration'] < 30).sum() / len(df) * 100
    ax.text(30, pct_below_30, f' {pct_below_30:.1f}% < 30s', 
           fontsize=10, color='red', va='bottom', fontweight='bold')
    
    # 添加分位数参考线
    for p in [25, 50, 75, 90]:
        dur_at_p = np.percentile(df['duration'], p)
        ax.axhline(p, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        ax.axvline(dur_at_p, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        ax.text(dur_at_p, p+2, f'{p}%\n{dur_at_p:.1f}s', fontsize=8, ha='center')
    
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_cumulative_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ 保存: duration_cumulative_distribution.png")
    
    # ========================================================================
    # 7. 小提琴图：按患者的分布（Top 15患者）
    # ========================================================================
    # 选择片段数最多的前15个患者
    top_patients = df.groupby(['test_name', 'patient_name'])['duration'].count().sort_values(ascending=False).head(15).index
    
    if len(top_patients) > 1:
        df_top = df[df.set_index(['test_name', 'patient_name']).index.isin(top_patients)]
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # 创建组合标签
        df_top['patient_label'] = df_top.apply(lambda x: f"[{x['test_name']}] {x['patient_name']}", axis=1)
        
        # 按总时长排序
        patient_order = df_top.groupby('patient_label')['duration'].sum().sort_values(ascending=False).index.tolist()
        
        # 绘制小提琴图
        parts = ax.violinplot([df_top[df_top['patient_label'] == p]['duration'].values 
                               for p in patient_order],
                              positions=range(len(patient_order)),
                              showmeans=True, showmedians=True)
        
        ax.set_xticks(range(len(patient_order)))
        ax.set_xticklabels(patient_order, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Segment Duration (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Segment Duration Distribution by Top 15 Patients (Violin Plot)', 
                    fontsize=14, fontweight='bold')
        ax.axhline(30, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='30s threshold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'duration_violin_by_patient.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("✓ 保存: duration_violin_by_patient.png")
    
    # ========================================================================
    # 8. 散点图：每个文件的片段数 vs 平均片段长度
    # ========================================================================
    file_summary = df.groupby(['test_name', 'patient_name', 'file_name']).agg({
        'duration': ['count', 'mean', 'sum']
    }).reset_index()
    file_summary.columns = ['test_name', 'patient_name', 'file_name', '片段数', '平均长度', '总时长']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 按测试名称着色
    test_names = file_summary['test_name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_names)))
    
    for idx, test in enumerate(test_names):
        df_test = file_summary[file_summary['test_name'] == test]
        ax.scatter(df_test['片段数'], df_test['平均长度'], 
                  s=df_test['总时长']/10, alpha=0.6, 
                  color=colors[idx], label=test, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Number of Segments per File', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Segment Duration (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('File-level Analysis: Segment Count vs Average Duration\n(Bubble size = total duration)', 
                fontsize=14, fontweight='bold')
    ax.axhline(30, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='30s threshold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_count_vs_duration.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ 保存: scatter_count_vs_duration.png")


def generate_summary_report(df, output_dir):
    """
    生成详细文本报告
    
    参数:
        df: DataFrame 所有片段数据
        output_dir: 输出目录
    """
    report_file = os.path.join(output_dir, 'boundary_segments_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EEG Boundary片段长度分布分析报告\n")
        f.write("="*80 + "\n\n")
        
        # 1. 总体统计
        f.write("1. 总体统计\n")
        f.write("-"*80 + "\n")
        f.write(f"总片段数: {len(df)}\n")
        f.write(f"总文件数: {df['file_name'].nunique()}\n")
        f.write(f"总时长: {df['duration'].sum():.2f} 秒 ({df['duration'].sum()/60:.2f} 分钟, {df['duration'].sum()/3600:.2f} 小时)\n")
        f.write(f"平均片段长度: {df['duration'].mean():.2f} ± {df['duration'].std():.2f} 秒\n")
        f.write(f"片段长度中位数: {df['duration'].median():.2f} 秒\n")
        f.write(f"片段长度范围: [{df['duration'].min():.2f}, {df['duration'].max():.2f}] 秒\n\n")
        
        f.write("分位数分布:\n")
        for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            f.write(f"  {int(q*100):2d}%: {df['duration'].quantile(q):7.2f} 秒\n")
        f.write("\n")
        
        # 2. 数据集概况
        f.write("2. 数据集概况\n")
        f.write("-"*80 + "\n")
        f.write(f"测试数量: {df['test_name'].nunique()}\n")
        f.write(f"患者数量: {df['patient_name'].nunique()}\n")
        f.write(f"文件总数: {df['file_name'].nunique()}\n")
        f.write(f"平均每个文件的片段数: {len(df) / df['file_name'].nunique():.2f}\n\n")
        
        # 3. 长短片段对比
        f.write("3. 长短片段对比（以30秒为界）\n")
        f.write("-"*80 + "\n")
        long_segs = df[df['duration'] >= 30]
        short_segs = df[df['duration'] < 30]
        
        f.write(f"长片段（≥30秒）:\n")
        f.write(f"  数量: {len(long_segs)} ({len(long_segs)/len(df)*100:.2f}%)\n")
        f.write(f"  总时长: {long_segs['duration'].sum():.2f} 秒 ({long_segs['duration'].sum()/60:.2f} 分钟)\n")
        f.write(f"  平均长度: {long_segs['duration'].mean():.2f} ± {long_segs['duration'].std():.2f} 秒\n\n")
        
        f.write(f"短片段（<30秒）:\n")
        f.write(f"  数量: {len(short_segs)} ({len(short_segs)/len(df)*100:.2f}%)\n")
        f.write(f"  总时长: {short_segs['duration'].sum():.2f} 秒 ({short_segs['duration'].sum()/60:.2f} 分钟)\n")
        f.write(f"  平均长度: {short_segs['duration'].mean():.2f} ± {short_segs['duration'].std():.2f} 秒\n\n")
        
        # 4. 按测试统计
        f.write("4. 按测试统计\n")
        f.write("-"*80 + "\n")
        test_stats = df.groupby('test_name').agg({
            'duration': ['count', 'sum', 'mean', 'median'],
            'patient_name': 'nunique',
            'file_name': 'nunique'
        }).round(2)
        test_stats.columns = ['片段数', '总时长(秒)', '平均长度', '中位数', '患者数', '文件数']
        test_stats['总时长(分钟)'] = (test_stats['总时长(秒)'] / 60).round(2)
        f.write(test_stats.to_string() + "\n\n")
        
        # 每个测试的长短片段分布
        f.write("各测试的长短片段分布:\n")
        for test in df['test_name'].unique():
            df_test = df[df['test_name'] == test]
            long_count = (df_test['duration'] >= 30).sum()
            short_count = (df_test['duration'] < 30).sum()
            f.write(f"  {test}:\n")
            f.write(f"    ≥30秒: {long_count} ({long_count/len(df_test)*100:.1f}%)\n")
            f.write(f"    <30秒: {short_count} ({short_count/len(df_test)*100:.1f}%)\n")
        f.write("\n")
        
        # 5. 片段长度区间分布（只显示有数据的区间）
        f.write("5. 片段长度区间分布（1秒精度，仅显示有数据的区间）\n")
        f.write("-"*80 + "\n")
        duration_dist = df['duration_bin'].value_counts().sort_index()
        duration_dist_filtered = duration_dist[duration_dist > 0]
        
        # 显示前50个最常见的区间
        f.write(f"有数据的区间总数: {len(duration_dist_filtered)}\n\n")
        f.write("最常见的50个区间:\n")
        for bin_label, count in duration_dist_filtered.head(50).items():
            percentage = count / len(df) * 100
            bar = '█' * int(percentage / 2)
            f.write(f"  {bin_label:>12s}: {count:5d} ({percentage:5.2f}%) {bar}\n")
        
        if len(duration_dist_filtered) > 50:
            f.write(f"\n... 还有 {len(duration_dist_filtered) - 50} 个区间\n")
        f.write("\n")
        
        # 5.2 聚合区间分布（总览）
        f.write("聚合区间分布（总览）:\n")
        f.write("-"*80 + "\n")
        overview_dist = df['duration_bin_overview'].value_counts().sort_index()
        for bin_label, count in overview_dist.items():
            percentage = count / len(df) * 100
            bar = '█' * int(percentage / 2)
            f.write(f"  {bin_label:>12s}: {count:5d} ({percentage:5.2f}%) {bar}\n")
        f.write("\n")
        
        # 6. Top患者
        f.write("6. Top 15 患者（按总时长）\n")
        f.write("-"*80 + "\n")
        patient_stats = df.groupby(['test_name', 'patient_name']).agg({
            'duration': ['count', 'sum', 'mean'],
            'file_name': 'nunique'
        }).round(2)
        patient_stats.columns = ['片段数', '总时长(秒)', '平均长度', '文件数']
        patient_stats['总时长(分钟)'] = (patient_stats['总时长(秒)'] / 60).round(2)
        patient_stats = patient_stats.sort_values('总时长(秒)', ascending=False).head(15)
        f.write(patient_stats.to_string() + "\n\n")
        
        # 7. 片段类型统计
        f.write("7. 片段类型统计\n")
        f.write("-"*80 + "\n")
        type_counts = df['segment_type'].value_counts()
        for seg_type, count in type_counts.items():
            percentage = count / len(df) * 100
            f.write(f"  {seg_type}: {count} ({percentage:.2f}%)\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("报告结束\n")
        f.write("="*80 + "\n")
    
    print(f"✓ 保存: boundary_segments_report.txt")
    
    # 打印到控制台
    print("\n" + "="*80)
    print("报告内容")
    print("="*80)
    with open(report_file, 'r', encoding='utf-8') as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(
        description="分析.set文件中boundary事件之间的片段长度分布",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python analyze_boundary_segments.py --data_root "E:\\DataSet\\EEG\\dataset"
  
  # 指定文件匹配模式
  python analyze_boundary_segments.py \\
      --data_root "E:\\DataSet\\EEG\\dataset" \\
      --pattern "*_postICA.set"
  
  # 指定输出目录
  python analyze_boundary_segments.py \\
      --data_root "E:\\DataSet\\EEG\\dataset" \\
      --output_dir "E:\\output\\boundary_analysis"

功能说明:
  1. 扫描所有.set文件
  2. 读取每个文件的annotations
  3. 提取boundary事件
  4. 计算boundary之间的片段长度
  5. 生成多层级统计和可视化

输出文件:
  - boundary_segments_report.txt           # 详细报告
  - test_statistics.csv                    # 测试统计
  - patient_statistics.csv                 # 患者统计
  - file_statistics.csv                    # 文件统计
  - long_segments_by_patient.csv          # 长片段（≥30s）统计
  - all_segments.csv                       # 所有片段详细数据
  - duration_histogram_overall.png         # 直方图
  - duration_histogram_log.png            # 对数尺度直方图
  - duration_boxplot_by_test.png          # 箱线图
  - patient_segment_summary.png           # 患者汇总
  - duration_distribution_bins.png        # 区间分布
  - duration_cumulative_distribution.png  # 累积分布
  - duration_violin_by_patient.png        # 小提琴图
  - scatter_count_vs_duration.png         # 散点图
        """
    )
    
    parser.add_argument(
        '--data_root',
        required=True,
        help="数据文件根目录（包含.set文件）"
    )
    parser.add_argument(
        '--pattern',
        default='*_merged_*.set',
        help="文件匹配模式，默认: *_merged_*.set"
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help="输出目录，默认为data_root/boundary_analysis"
    )
    
    args = parser.parse_args()
    
    # 检查目录
    if not os.path.exists(args.data_root):
        print(f"错误: 数据根目录不存在: {args.data_root}")
        return 1
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_root, 'boundary_analysis')
    
    print("="*80)
    print("Boundary片段长度分布分析")
    print("="*80)
    print(f"数据根目录: {args.data_root}")
    print(f"文件模式: {args.pattern}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    # 收集所有boundary片段数据
    all_segments = collect_all_boundary_segments(args.data_root, args.pattern)
    
    if not all_segments:
        print("\n❌ 未找到任何片段数据")
        return 1
    
    # 分析分布
    df = analyze_distribution(all_segments, args.output_dir)
    
    # 保存详细数据
    df.to_csv(os.path.join(args.output_dir, 'all_boundary_segments.csv'), index=False)
    print(f"\n✓ 所有片段数据已保存: all_boundary_segments.csv ({len(df)} 条记录)")
    
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
            '--data_root', r'E:\DataSet\EEG\EEG dataset_SUAT_processed',
            '--pattern', '*_merged_*.set'
        ])
    
    sys.exit(main())

