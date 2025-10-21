#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_labels_csv.py

从connectivity_features文件夹中提取通道组合信息，生成labels.csv

注意：
- .set数据文件在数据文件根目录（data_root）中查找
- 特征文件夹在特征文件根目录（features_root）中查找

目录结构：
    data_root/
    ├── 测试名称1（如：新增10例）/
    │   ├── 患者1/
    │   │   └── xxx_merged_channels.set
    │   └── 患者2/
    └── 测试名称2（如：补充）/
    
    features_root/
    ├── 测试名称1（如：新增10例）/
    │   ├── 患者1/
    │   │   └── xxx_merged_channels_connectivity_features/
    │   └── 患者2/
    └── 测试名称2（如：补充）/

输出CSV格式：
- test_name: 测试名称（第一层目录，如：新增10例、补充）
- patient_name: 患者名称（第二层目录）
- data_file_path: 原始.set文件相对于data_root的路径
- features_dir_path: 特征文件夹相对于features_root的路径
- channel_combination: 通道组合列表，格式：[Ch1,Ch2,Ch3]

使用方法:
    python generate_labels_csv.py --data_root "E:\data" --features_root "E:\features"
"""

import os
import csv
import argparse
import re
from pathlib import Path
from collections import Counter


def extract_channels_from_folder_name(folder_name):
    """
    从文件夹名中提取通道组合
    
    例如: SZ1_preICA_reject_1_postICA_merged_F8_Fp2_Sph_R_T4_connectivity_features
    提取: F8, Fp2, Sph_R, T4
    
    参数:
        folder_name: 文件夹名称
    
    返回:
        channels: 通道列表，如 ['F8', 'Fp2', 'Sph_R', 'T4']
    """
    # 查找 _merged_ 和 _connectivity_features 之间的部分
    pattern = r'_merged_(.+)_connectivity_features'
    match = re.search(pattern, folder_name)
    
    if match:
        channels_str = match.group(1)
        # 将下划线分隔的通道名分割
        channels = split_channel_names(channels_str)
        return channels
    
    return []


def split_channel_names(channels_str):
    """
    智能分割通道名称
    
    常见模式：
    - 单个字母+数字: F8, T4, O1, O2
    - 两个字母+数字: Fp1, Fp2
    - 带下划线: Sph_R, Sph_L
    - 特殊: FPz, Cz
    
    参数:
        channels_str: 通道字符串，如 "F8_Fp2_Sph_R_T4"
    
    返回:
        channels: 通道列表
    """
    channels = []
    parts = channels_str.split('_')
    
    i = 0
    while i < len(parts):
        part = parts[i]
        
        # 检查是否是通道名的一部分（如 Sph_R 中的 Sph）
        # 规则：如果当前部分是纯字母且下一部分是单个字母或很短，可能是一个通道名
        if i + 1 < len(parts):
            next_part = parts[i + 1]
            # 如果当前部分没有数字，且下一部分很短（可能是 R/L 等后缀）
            if not any(c.isdigit() for c in part) and len(next_part) <= 2 and not any(c.isdigit() for c in next_part):
                # 合并为一个通道名
                channels.append(f"{part}_{next_part}")
                i += 2
                continue
        
        # 否则作为独立通道
        channels.append(part)
        i += 1
    
    return channels


def find_matching_data_file(features_dir, data_root, features_root):
    """
    根据特征文件夹，在数据根目录中查找对应的.set文件
    
    参数:
        features_dir: 特征文件夹路径（绝对路径）
        data_root: 数据文件根目录
        features_root: 特征文件根目录
    
    返回:
        data_file_path: 数据文件路径，None表示未找到
        test_name: 测试名称
        patient_name: 患者名称
    """
    features_path = Path(features_dir)
    data_root_path = Path(data_root)
    features_root_path = Path(features_root)
    
    # 获取特征目录相对于features_root的路径
    try:
        rel_path = features_path.relative_to(features_root_path)
        path_parts = rel_path.parts
        
        # 提取测试名称和患者名称
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
        
        # 构建.set文件名
        folder_name = features_path.name
        set_name = folder_name.replace('_connectivity_features', '.set')
        
        # 在数据根目录中查找对应的.set文件
        # 尝试相同的相对路径结构
        if len(path_parts) >= 3:
            # 构建可能的.set文件路径
            possible_data_path = data_root_path / test_name / patient_name / set_name
            
            if possible_data_path.exists():
                return possible_data_path, test_name, patient_name
        
        # 如果上述路径不存在，在整个data_root中搜索
        for data_file in data_root_path.rglob(set_name):
            # 提取测试名称和患者名称
            try:
                data_rel = data_file.relative_to(data_root_path)
                data_parts = data_rel.parts
                if len(data_parts) >= 3:
                    test_name = data_parts[0]
                    patient_name = data_parts[1]
                return data_file, test_name, patient_name
            except:
                continue
        
        return None, test_name, patient_name
        
    except Exception as e:
        print(f"      错误: {e}")
        return None, "未知测试", "未知患者"


def find_all_labels(data_root, features_root):
    """
    查找所有的特征文件夹并匹配数据文件
    
    参数:
        data_root: 数据文件根目录
        features_root: 特征文件根目录
    
    返回:
        results: list of dict
    """
    results = []
    features_root_path = Path(features_root)
    data_root_path = Path(data_root)
    
    print("扫描特征文件夹...")
    print("-"*80)
    
    # 递归查找所有 *_connectivity_features 文件夹
    feature_dirs = list(features_root_path.rglob('*_connectivity_features'))
    
    print(f"找到 {len(feature_dirs)} 个特征文件夹")
    print()
    
    for features_dir in feature_dirs:
        if not features_dir.is_dir():
            continue
        
        folder_name = features_dir.name
        
        # 提取通道组合
        channels = extract_channels_from_folder_name(folder_name)
        
        if not channels:
            print(f"⚠ 无法提取通道信息: {folder_name}")
            continue
        
        # 查找对应的.set文件
        data_file, test_name, patient_name = find_matching_data_file(features_dir, data_root, features_root)
        
        if data_file is None:
            set_name = folder_name.replace('_connectivity_features', '.set')
            print(f"⚠ 找不到对应的.set文件: {set_name}")
            continue
        
        # 计算相对路径
        try:
            data_rel_path = data_file.relative_to(data_root_path)
            features_rel_path = features_dir.relative_to(features_root_path)
        except ValueError as e:
            print(f"⚠ 路径错误: {e}")
            continue
        
        results.append({
            'test_name': test_name,
            'patient_name': patient_name,
            'data_file': str(data_rel_path).replace('\\', '/'),
            'features_dir': str(features_rel_path).replace('\\', '/'),
            'channels': channels
        })
        
        print(f"✓ [{test_name}] {patient_name} -> {channels}")
    
    return results


def generate_labels_csv(data_root, features_root, output_file, output_dir=None):
    """
    生成 labels.csv 文件
    
    参数:
        data_root: 数据文件根目录
        features_root: 特征文件根目录
        output_file: 输出CSV文件名
        output_dir: 输出目录，默认为features_root
    """
    print("="*80)
    print("生成 Labels CSV 文件")
    print("="*80)
    print(f"数据文件根目录: {data_root}")
    print(f"特征文件根目录: {features_root}")
    print()
    
    # 查找所有特征文件夹并匹配数据文件
    results = find_all_labels(data_root, features_root)
    
    if not results:
        print("\n❌ 未找到任何匹配的文件")
        return False
    
    print(f"\n成功匹配 {len(results)} 个样本")
    print("="*80)
    
    # 确定输出路径
    if output_dir is None:
        output_dir = features_root
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, output_file)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['test_name', 'patient_name', 'data_file_path', 'features_dir_path', 'channel_combination'])
        
        # 按测试名称、患者名称、路径排序
        results_sorted = sorted(results, key=lambda x: (x['test_name'], x['patient_name'], x['data_file']))
        
        for item in results_sorted:
            # 通道组合格式: [Ch1,Ch2,Ch3]
            channels_str = '[' + ','.join(item['channels']) + ']'
            
            writer.writerow([
                item['test_name'],
                item['patient_name'],
                item['data_file'],
                item['features_dir'],
                channels_str
            ])
    
    print(f"\n✓ Labels CSV 已生成: {csv_path}")
    print(f"  总共 {len(results)} 条记录")
    
    # 显示前几条示例
    print(f"\n示例数据（前5条）:")
    print("-"*80)
    for i, item in enumerate(results_sorted[:5]):
        channels_str = '[' + ','.join(item['channels']) + ']'
        print(f"{i+1}. 测试名称: {item['test_name']}")
        print(f"   患者名称: {item['patient_name']}")
        print(f"   数据文件: {item['data_file']}")
        print(f"   特征目录: {item['features_dir']}")
        print(f"   通道组合: {channels_str}")
        print()
    
    if len(results) > 5:
        print(f"... 还有 {len(results) - 5} 条记录")
    
    # 统计信息
    print("\n统计信息:")
    print("="*80)
    
    # 1. 按测试名称统计
    test_counts = Counter([item['test_name'] for item in results])
    print(f"\n按测试分组:")
    print("-"*80)
    for test_name, count in sorted(test_counts.items()):
        print(f"  {test_name}: {count} 个样本")
    
    # 2. 按患者统计
    patient_counts = Counter([item['patient_name'] for item in results])
    print(f"\n按患者分组（前20个）:")
    print("-"*80)
    for patient_name, count in patient_counts.most_common(20):
        print(f"  {patient_name}: {count} 个样本")
    if len(patient_counts) > 20:
        print(f"  ... 还有 {len(patient_counts) - 20} 个患者")
    
    # 3. 按测试+患者组合统计
    test_patient_counts = Counter([(item['test_name'], item['patient_name']) for item in results])
    print(f"\n测试-患者组合统计（前15个）:")
    print("-"*80)
    for (test, patient), count in test_patient_counts.most_common(15):
        print(f"  [{test}] {patient}: {count} 个样本")
    
    # 4. 统计通道组合的频率
    channel_combos = [tuple(item['channels']) for item in results]
    combo_counts = Counter(channel_combos)
    
    print(f"\n通道组合统计:")
    print("-"*80)
    print(f"不同的通道组合数量: {len(combo_counts)}")
    print(f"\n最常见的通道组合（前10个）:")
    for combo, count in combo_counts.most_common(10):
        channels_str = '[' + ','.join(combo) + ']'
        print(f"  {channels_str}: {count} 次")
    
    # 5. 统计通道数量分布
    channel_counts = Counter([len(item['channels']) for item in results])
    print(f"\n通道数量分布:")
    print("-"*80)
    for n_channels, count in sorted(channel_counts.items()):
        print(f"  {n_channels} 个通道: {count} 个样本")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="生成 labels.csv 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 数据文件和特征文件在不同目录
  python generate_labels_csv.py \\
      --data_root "E:\\DataSet\\EEG\\dataset" \\
      --features_root "E:\\output\\connectivity_features"
  
  # 数据文件和特征文件在同一目录
  python generate_labels_csv.py \\
      --data_root "E:\\DataSet\\EEG\\dataset" \\
      --features_root "E:\\DataSet\\EEG\\dataset"
  
  # 指定输出文件名
  python generate_labels_csv.py \\
      --data_root "E:\\data" \\
      --features_root "E:\\features" \\
      --output_file "my_labels.csv"

输出格式:
  test_name,patient_name,data_file_path,features_dir_path,channel_combination
  新增10例,患者1,新增10例/患者1/data.set,新增10例/患者1/features/,"[F8,Fp2,Sph_R,T4]"
  补充,患者2,补充/患者2/data.set,补充/患者2/features/,"[O1,O2]"

目录结构示例:
  data_root/
  ├── 新增10例/
  │   ├── 患者1/
  │   │   └── SZ1_merged_F8_Fp2_Sph_R_T4.set
  │   └── 患者2/
  └── 补充/
  
  features_root/
  ├── 新增10例/
  │   ├── 患者1/
  │   │   └── SZ1_merged_F8_Fp2_Sph_R_T4_connectivity_features/
  │   └── 患者2/
  └── 补充/
        """
    )
    
    parser.add_argument(
        '--data_root',
        required=True,
        help="数据文件根目录（包含.set文件）"
    )
    parser.add_argument(
        '--features_root',
        required=True,
        help="特征文件根目录（包含*_connectivity_features文件夹）"
    )
    parser.add_argument(
        '--output_file',
        default='labels.csv',
        help="输出CSV文件名，默认: labels.csv"
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help="输出目录，默认为features_root"
    )
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.data_root):
        print(f"错误: 数据根目录不存在: {args.data_root}")
        return 1
    
    if not os.path.exists(args.features_root):
        print(f"错误: 特征根目录不存在: {args.features_root}")
        return 1
    
    # 生成CSV
    success = generate_labels_csv(args.data_root, args.features_root, args.output_file, args.output_dir)
    
    if success:
        print("\n" + "="*80)
        print("✅ 完成！")
        print("="*80)
        return 0
    else:
        return 1


if __name__ == "__main__":
    import sys
    
    # 如果不提供命令行参数，使用默认值
    if len(sys.argv) == 1:
        print("使用默认参数运行...")
        sys.argv.extend([
            '--data_root', r'E:\DataSet\EEG\EEG dataset_SUAT_processed',
            '--features_root', r'E:\output\connectivity_features'
        ])
    
    sys.exit(main())
