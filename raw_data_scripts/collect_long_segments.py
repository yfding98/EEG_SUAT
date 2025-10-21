#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_long_segments.py

统计所有患者中大于指定时长的合并片段，并打包相关文件。

功能:
1. 扫描所有 _channel_merge_statistics.csv 文件
2. 筛选出时长 >= 阈值（默认30秒）的片段
3. 汇总到总结CSV文件
4. 将相关的 .set 文件（及 .fdt）打包到 ZIP 文件，保留相对路径

使用方法:
    # 默认30秒阈值
    python collect_long_segments.py --root_dir "E:\data" --out_dir "E:\output"
    
    # 自定义阈值（例如60秒）
    python collect_long_segments.py --root_dir "E:\data" --out_dir "E:\output" --min_duration 60
"""

import os
import csv
import argparse
import zipfile
import shutil
import pandas as pd
from pathlib import Path


def find_long_segments(root_dir, min_duration_sec=30):
    """
    查找所有大于指定时长的合并片段
    
    返回: list of dict
        {
            'patient': 患者名,
            'file': 文件名,
            'channel_combo': 通道组合,
            'segment_count': 片段数量,
            'duration_sec': 总时长(秒),
            'duration_min': 总时长(分钟),
            'data_file': 数据文件路径,
            'csv_file': 统计CSV文件路径,
            'relative_path': 相对路径
        }
    """
    long_segments = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('_channel_merge_statistics.csv'):
                csv_path = os.path.join(dirpath, filename)
                
                try:
                    # 读取CSV
                    df = pd.read_csv(csv_path, encoding='utf-8')
                    
                    # 筛选符合条件的行
                    mask = df['总时长(秒)'] >= min_duration_sec
                    filtered_df = df[mask]
                    
                    if len(filtered_df) == 0:
                        continue
                    
                    # 提取患者和文件信息
                    # 假设路径结构：root_dir/类别/患者名/...
                    try:
                        relative_path = os.path.relpath(dirpath, root_dir)
                        path_parts = relative_path.split(os.sep)
                        
                        # 尝试提取患者名（通常是倒数第二或第三级目录）
                        if len(path_parts) >= 2:
                            patient_name = path_parts[-2] if path_parts[-1].endswith('_results') else path_parts[-1]
                            category = path_parts[0] if len(path_parts) > 1 else "未分类"
                        else:
                            patient_name = path_parts[-1]
                            category = "未分类"
                    except:
                        patient_name = os.path.basename(dirpath)
                        category = "未分类"
                        relative_path = dirpath
                    
                    # 处理每一行
                    for _, row in filtered_df.iterrows():
                        channel_combo = row['通道组合']
                        count = row['片段数量']
                        duration_s = row['总时长(秒)']
                        duration_m = row['总时长(分钟)']
                        output_file = row['输出文件名']
                        
                        # 构建数据文件完整路径
                        data_file_path = os.path.join(dirpath, output_file)
                        
                        long_segments.append({
                            'category': category,
                            'patient': patient_name,
                            'file': filename.replace('_channel_merge_statistics.csv', ''),
                            'channel_combo': channel_combo,
                            'segment_count': int(count),
                            'duration_sec': float(duration_s),
                            'duration_min': float(duration_m),
                            'data_file': data_file_path,
                            'csv_file': csv_path,
                            'relative_path': relative_path
                        })
                
                except Exception as e:
                    print(f"Error processing {csv_path}: {e}")
    
    return long_segments


def save_summary_csv(long_segments, output_path):
    """
    保存汇总CSV文件
    
    参数:
        long_segments: find_long_segments 返回的列表
        output_path: 输出CSV路径
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            '类别', '患者', '文件', '通道组合', '片段数量', 
            '总时长(秒)', '总时长(分钟)', '数据文件路径', '相对路径'
        ])
        
        # 按类别、患者、时长排序
        sorted_segments = sorted(
            long_segments, 
            key=lambda x: (x['category'], x['patient'], -x['duration_sec'])
        )
        
        for seg in sorted_segments:
            writer.writerow([
                seg['category'],
                seg['patient'],
                seg['file'],
                seg['channel_combo'],
                seg['segment_count'],
                f"{seg['duration_sec']:.2f}",
                f"{seg['duration_min']:.2f}",
                seg['data_file'],
                seg['relative_path']
            ])
    
    print(f"✓ 汇总CSV已保存: {output_path}")


def create_zip_with_files(long_segments, root_dir, zip_path):
    """
    创建ZIP文件，包含所有相关的 .set 和 .fdt 文件
    
    参数:
        long_segments: find_long_segments 返回的列表
        root_dir: 根目录
        zip_path: 输出ZIP路径
    """
    # 收集所有需要复制的文件
    files_to_zip = set()
    
    for seg in long_segments:
        data_file = seg['data_file']
        
        # 添加数据文件
        if os.path.exists(data_file):
            files_to_zip.add(data_file)
            
            # 添加对应的 .fdt 文件（如果存在）
            fdt_file = data_file.replace('.set', '.fdt')
            if os.path.exists(fdt_file):
                files_to_zip.add(fdt_file)
        else:
            print(f"⚠ 文件不存在: {data_file}")
    
    if not files_to_zip:
        print("❌ 没有文件需要打包")
        return False
    
    print(f"\n正在创建ZIP文件: {zip_path}")
    print(f"将打包 {len(files_to_zip)} 个文件...")
    
    # 创建ZIP文件
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in sorted(files_to_zip):
            # 计算相对路径（保留目录结构）
            try:
                arcname = os.path.relpath(file_path, root_dir)
            except ValueError:
                # 如果不在同一驱动器，使用文件的最后几级目录
                arcname = os.path.join(*Path(file_path).parts[-3:])
            
            zipf.write(file_path, arcname)
            print(f"  + {arcname}")
    
    # 显示文件大小
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n✓ ZIP文件创建成功!")
    print(f"  路径: {zip_path}")
    print(f"  大小: {zip_size_mb:.2f} MB")
    
    return True


def print_statistics(long_segments, min_duration_sec):
    """
    打印统计信息
    """
    if not long_segments:
        return
    
    print("\n" + "=" * 80)
    print(f"统计摘要 (时长 >= {min_duration_sec}秒)")
    print("=" * 80)
    
    # 按患者统计
    patient_stats = {}
    for seg in long_segments:
        patient = seg['patient']
        if patient not in patient_stats:
            patient_stats[patient] = {
                'count': 0,
                'total_duration': 0,
                'channel_combos': set()
            }
        patient_stats[patient]['count'] += 1
        patient_stats[patient]['total_duration'] += seg['duration_sec']
        patient_stats[patient]['channel_combos'].add(seg['channel_combo'])
    
    print(f"\n总共找到 {len(long_segments)} 个符合条件的片段")
    print(f"涉及 {len(patient_stats)} 个患者")
    print(f"\n患者统计:")
    print(f"{'患者':<20} {'片段数':<10} {'总时长(秒)':<15} {'总时长(分)':<15} {'通道组合数':<15}")
    print("-" * 80)
    
    # 按总时长排序
    sorted_patients = sorted(patient_stats.items(), key=lambda x: x[1]['total_duration'], reverse=True)
    for patient, stats in sorted_patients:
        print(f"{patient:<20} {stats['count']:<10} {stats['total_duration']:<15.2f} "
              f"{stats['total_duration']/60:<15.2f} {len(stats['channel_combos']):<15}")
    
    # 通道组合统计
    channel_stats = {}
    for seg in long_segments:
        channel = seg['channel_combo']
        if channel not in channel_stats:
            channel_stats[channel] = {
                'count': 0,
                'total_duration': 0
            }
        channel_stats[channel]['count'] += 1
        channel_stats[channel]['total_duration'] += seg['duration_sec']
    
    print(f"\n通道组合统计:")
    print(f"{'通道组合':<30} {'片段数':<10} {'总时长(秒)':<15} {'总时长(分)':<15}")
    print("-" * 80)
    
    sorted_channels = sorted(channel_stats.items(), key=lambda x: x[1]['total_duration'], reverse=True)
    for channel, stats in sorted_channels:
        print(f"{channel:<30} {stats['count']:<10} {stats['total_duration']:<15.2f} "
              f"{stats['total_duration']/60:<15.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="统计并收集长片段数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
功能说明:
  1. 扫描所有 _channel_merge_statistics.csv 文件
  2. 筛选时长 >= 阈值的片段
  3. 汇总到总结CSV
  4. 打包所有相关的 .set 和 .fdt 文件

输出:
  - summary_long_segments.csv: 汇总表格
  - long_segments_files.zip: 所有相关文件的压缩包

使用方法:
  # 默认30秒阈值
  python collect_long_segments.py --root_dir "E:\\data" --out_dir "E:\\output"
  
  # 自定义阈值（60秒）
  python collect_long_segments.py --root_dir "E:\\data" --out_dir "E:\\output" --min_duration 60
  
  # 只生成汇总不打包
  python collect_long_segments.py --root_dir "E:\\data" --out_dir "E:\\output" --no-zip
        """
    )
    
    parser.add_argument(
        '--root_dir',
        required=True,
        help="数据集根目录"
    )
    parser.add_argument(
        '--out_dir',
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        '--min_duration',
        type=float,
        default=30.0,
        help="最小时长阈值（秒），默认30秒"
    )
    parser.add_argument(
        '--no-zip',
        action='store_true',
        help="不创建ZIP文件"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"收集长片段数据 (时长 >= {args.min_duration}秒)")
    print("=" * 80)
    print(f"\n扫描目录: {args.root_dir}")
    
    # 查找长片段
    long_segments = find_long_segments(args.root_dir, args.min_duration)
    
    if not long_segments:
        print(f"\n❌ 未找到任何时长 >= {args.min_duration}秒的片段")
        return 1
    
    # 打印统计
    print_statistics(long_segments, args.min_duration)
    
    # 保存汇总CSV
    summary_csv_path = os.path.join(args.out_dir, f"summary_long_segments_{int(args.min_duration)}s.csv")
    save_summary_csv(long_segments, summary_csv_path)
    
    # 创建ZIP文件
    if not args.no_zip:
        zip_path = os.path.join(args.out_dir, f"long_segments_{int(args.min_duration)}s_files.zip")
        create_zip_with_files(long_segments, args.root_dir, zip_path)
    
    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    # 如果不提供命令行参数，使用默认值
    import sys
    if len(sys.argv) == 1:
        print("使用默认参数运行...")
        sys.argv.extend([
            '--root_dir', r'E:\DataSet\EEG\EEG dataset_SUAT_processed',
            '--out_dir', r'E:\output\long_segments_30s',
            '--min_duration', '30'
        ])
    
    sys.exit(main())

