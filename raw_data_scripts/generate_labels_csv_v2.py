#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_labels_csv_v2.py

为V2版本特征（5秒窗口，无拼接）生成labels.csv

与V1的区别：
1. 特征文件夹名改为 *_connectivity_v2
2. 特征文件名改为 connectivity_matrices_segXXXX.npz
3. 增加 is_concatenated, segment_id 等元数据

使用方法:
    python generate_labels_csv_v2.py --features_root "E:\output\connectivity_features_v2"
"""

import os
import csv
import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np


def extract_channels_from_folder_name(folder_name):
    """
    从文件夹名中提取通道组合
    
    例如: xxx_merged_F8_Fp2_Sph_R_T4_connectivity_v2
    提取: F8, Fp2, Sph_R, T4
    """
    # 移除后缀
    folder_name = folder_name.replace('_connectivity_v2', '')
    folder_name = folder_name.replace('_connectivity_features', '')
    
    # 查找 "_merged_" 后的部分
    if '_merged_' in folder_name:
        parts = folder_name.split('_merged_')
        if len(parts) > 1:
            channel_part = parts[1]
            
            # 清理常见的后缀
            for suffix in ['_preICA', '_postICA', '_reject', '_channels']:
                channel_part = channel_part.split(suffix)[0]
            
            # 分割通道（考虑下划线通道名如 Sph_R）
            # 假设通道名模式：大写字母开头
            channels = []
            current = ""
            tokens = channel_part.split('_')
            
            for token in tokens:
                if not token:
                    continue
                
                # 如果是单个字母或数字，可能是通道名的一部分（如 Sph_R）
                if current and (token in ['L', 'R', 'l', 'r'] or len(token) <= 2):
                    current += '_' + token
                else:
                    if current:
                        channels.append(current)
                    current = token
            
            if current:
                channels.append(current)
            
            # 清理和验证通道名
            valid_channels = []
            for ch in channels:
                # 只保留看起来像通道名的（包含字母）
                if any(c.isalpha() for c in ch) and len(ch) >= 2:
                    valid_channels.append(ch)
            
            return valid_channels
    
    return []


def find_data_file_by_channels(data_root, test_name, patient_name, channels):
    """
    根据通道组合查找对应的原始.set文件
    """
    patient_dir = Path(data_root) / test_name / patient_name
    
    if not patient_dir.exists():
        return None
    
    # 查找包含这些通道的.set文件
    set_files = list(patient_dir.glob('*.set'))
    
    # 尝试匹配通道
    for set_file in set_files:
        set_name = set_file.stem
        
        # 检查是否包含所有通道
        match_count = sum(1 for ch in channels if ch in set_name)
        if match_count == len(channels):
            # 返回相对路径
            return str(set_file.relative_to(data_root))
    
    # 如果没有完全匹配，返回第一个merged文件
    for set_file in set_files:
        if '_merged_' in set_file.stem:
            return str(set_file.relative_to(data_root))
    
    return None


def scan_features_directory(features_root, data_root=None):
    """
    扫描特征目录，提取所有信息
    
    返回:
        records: list of dict with:
            - test_name
            - patient_name
            - data_file_path
            - features_dir_path
            - channel_combination
            - n_windows (新增)
            - n_segments (新增)
    """
    features_path = Path(features_root)
    records = []
    
    # 遍历特征目录
    # 结构: features_root / test_name / patient_name / xxx_connectivity_v2
    for test_dir in features_path.iterdir():
        if not test_dir.is_dir():
            continue
        
        test_name = test_dir.name
        print(f"\n扫描测试: {test_name}")
        
        for patient_dir in test_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            
            patient_name = patient_dir.name
            
            # 查找特征文件夹（*_connectivity_v2）
            feature_dirs = list(patient_dir.glob('*_connectivity_v2'))
            
            for feature_dir in feature_dirs:
                # 检查是否有有效的特征文件
                npz_files = list(feature_dir.glob('connectivity_matrices_seg*.npz'))
                
                if len(npz_files) == 0:
                    print(f"  ⚠ 跳过（无特征文件）: {feature_dir.name}")
                    continue
                
                # 读取元数据
                meta_file = feature_dir / 'windows_metadata.csv'
                n_windows = len(npz_files)
                n_segments = 1
                
                if meta_file.exists():
                    try:
                        meta_df = pd.read_csv(meta_file)
                        n_windows = len(meta_df)
                        n_segments = meta_df['segment_id'].nunique()
                    except:
                        pass
                
                # 从文件夹名提取通道
                channels = extract_channels_from_folder_name(feature_dir.name)
                
                if not channels:
                    print(f"  ⚠ 无法提取通道: {feature_dir.name}")
                    continue
                
                # 查找对应的原始数据文件
                data_file_path = None
                if data_root:
                    data_file_path = find_data_file_by_channels(
                        data_root, test_name, patient_name, channels
                    )
                
                # 特征文件夹相对路径
                features_dir_path = str(feature_dir.relative_to(features_path))
                
                # 格式化通道组合
                channel_combination = ','.join(channels)
                
                records.append({
                    'test_name': test_name,
                    'patient_name': patient_name,
                    'data_file_path': data_file_path if data_file_path else '',
                    'features_dir_path': features_dir_path,
                    'channel_combination': channel_combination,
                    'n_windows': n_windows,
                    'n_segments': n_segments
                })
                
                print(f"  ✓ {patient_name}/{feature_dir.name}: {channel_combination} ({n_windows} windows, {n_segments} segments)")
    
    return records


def generate_labels_csv(features_root, output_file, data_root=None):
    """
    生成labels.csv
    """
    print(f"{'='*80}")
    print(f"生成 Labels CSV (V2版本)")
    print(f"{'='*80}")
    print(f"特征根目录: {features_root}")
    if data_root:
        print(f"数据根目录: {data_root}")
    print(f"输出文件: {output_file}")
    
    # 扫描目录
    records = scan_features_directory(features_root, data_root)
    
    if len(records) == 0:
        print("\n错误：未找到任何特征文件夹！")
        return
    
    # 保存为CSV
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # 统计信息
    print(f"\n{'='*80}")
    print(f"生成完成")
    print(f"{'='*80}")
    print(f"总记录数: {len(records)}")
    print(f"测试数: {df['test_name'].nunique()}")
    print(f"患者数: {df['patient_name'].nunique()}")
    print(f"总窗口数: {df['n_windows'].sum()}")
    print(f"总片段数: {df['n_segments'].sum()}")
    
    print(f"\n按测试分组:")
    test_stats = df.groupby('test_name').agg({
        'patient_name': 'nunique',
        'features_dir_path': 'count',
        'n_windows': 'sum'
    })
    test_stats.columns = ['患者数', '特征文件夹数', '总窗口数']
    print(test_stats)
    
    print(f"\n通道组合统计:")
    channel_counts = df['channel_combination'].value_counts().head(10)
    print(channel_counts)
    
    print(f"\n✓ Labels CSV已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="为V2版本特征生成labels.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--features_root', required=True,
                       help="特征根目录（包含test_name/patient_name/xxx_connectivity_v2）")
    parser.add_argument('--data_root', 
                       help="数据根目录（可选，用于查找原始.set文件）")
    parser.add_argument('--output', default='labels.csv',
                       help="输出CSV文件路径，默认: labels.csv")
    
    args = parser.parse_args()
    
    # 自动设置输出路径
    if args.output == 'labels.csv':
        output_file = os.path.join(args.features_root, 'labels.csv')
    else:
        output_file = args.output
    
    generate_labels_csv(args.features_root, output_file, args.data_root)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # 默认参数
        sys.argv.extend([
            '--features_root', r'E:\output\connectivity_features_v2',
            '--data_root', r'E:\DataSet\EEG\EEG dataset_SUAT_processed'
        ])
    sys.exit(main())

