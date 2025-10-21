#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_by_channels.py

根据 marked_abnormal_segments_postICA.csv，按异常通道列表分组合并数据段。
相同异常通道列表的所有片段（不论时间是否连续）会被合并到一个 .set 文件中。

使用方法:
    # 处理单个结果目录
    python merge_by_channels.py --result_dir "E:\data\patient\SZ1_results"
    
    # 批量处理所有结果目录
    python merge_by_channels.py --root_dir "E:\DataSet\EEG\EEG dataset_SUAT_processed"
"""

import os
import csv
import argparse
import numpy as np
import mne
from collections import defaultdict
from pathlib import Path


def read_marked_segments_postica(csv_path):
    """
    读取 marked_abnormal_segments_postICA.csv
    
    返回: [(start, end, channels_str), ...]
    """
    segments = []
    
    if not os.path.exists(csv_path):
        print(f"⚠ CSV file not found: {csv_path}")
        return segments
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        for row in reader:
            if len(row) >= 3:
                start = float(row[0])
                end = float(row[1])
                channels = row[2].strip()
                segments.append((start, end, channels))
    
    return segments


def group_segments_by_channels(segments):
    """
    按异常通道列表分组
    
    返回: {channels_str: [(start, end), ...]}
    """
    grouped = defaultdict(list)
    
    for start, end, channels in segments:
        # 标准化通道列表（排序，确保一致性）
        channel_list = sorted([ch.strip() for ch in channels.split(',')])
        channels_key = ','.join(channel_list)
        grouped[channels_key].append((start, end))
    
    # 对每组内的时间段按开始时间排序
    for channels_key in grouped:
        grouped[channels_key].sort(key=lambda x: x[0])
    
    return grouped


def merge_segments_for_channel_group(set_file, time_segments, channels_str, output_set_file):
    """
    合并指定通道组的所有时间段数据
    
    参数:
        set_file: 输入的 _postICA.set 文件路径
        time_segments: [(start, end), ...] 时间段列表（已排序）
        channels_str: 通道列表字符串，用于注释
        output_set_file: 输出的 .set 文件路径
    """
    # 读取 EEG 数据
    print(f"  Loading: {os.path.basename(set_file)}")
    raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose='ERROR')
    sfreq = raw.info['sfreq']
    
    # 提取所有片段的数据
    extracted_data_list = []
    segment_info = []
    
    for i, (start, end) in enumerate(time_segments):
        start_sample = int(start * sfreq)
        end_sample = int(end * sfreq)
        
        # 确保不越界
        start_sample = max(0, start_sample)
        end_sample = min(len(raw.times), end_sample)
        
        if start_sample >= end_sample:
            print(f"    ⚠ Invalid segment: {start:.2f}-{end:.2f}s")
            continue
        
        # 提取数据片段
        segment_data = raw.get_data(start=start_sample, stop=end_sample)
        extracted_data_list.append(segment_data)
        segment_info.append((start, end, end_sample - start_sample))
        print(f"    Segment {i+1}: {start:.2f}s - {end:.2f}s ({end_sample - start_sample} samples)")
    
    if not extracted_data_list:
        print(f"    ⚠ No valid segments extracted")
        return False
    
    # 拼接所有片段
    concatenated_data = np.concatenate(extracted_data_list, axis=1)
    total_samples = concatenated_data.shape[1]
    total_duration = total_samples / sfreq
    print(f"    ✓ Merged {len(extracted_data_list)} segments → {total_samples} samples ({total_duration:.2f}s)")
    
    # 创建新的 Raw 对象
    info = raw.info.copy()
    new_raw = mne.io.RawArray(concatenated_data, info, verbose='ERROR')
    
    # 添加注释说明每个片段的来源
    descriptions = []
    onsets = []
    durations = []
    cumulative_time = 0
    
    for i, (orig_start, orig_end, n_samples) in enumerate(segment_info):
        descriptions.append(f"Original: {orig_start:.2f}s-{orig_end:.2f}s")
        onsets.append(cumulative_time)
        duration = n_samples / sfreq
        durations.append(duration)
        cumulative_time += duration
    
    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    new_raw.set_annotations(annotations)
    
    # 保存为 EEGLAB 格式
    print(f"    Saving: {os.path.basename(output_set_file)}")
    mne.export.export_raw(output_set_file, new_raw, fmt='eeglab', overwrite=True, verbose='ERROR')
    print(f"    ✓ Saved: {output_set_file}")
    
    return True


def process_single_result_dir(result_dir):
    """
    处理单个结果目录
    
    参数:
        result_dir: 结果目录路径（包含 marked_abnormal_segments_postICA.csv）
    """
    csv_path = os.path.join(result_dir, 'marked_abnormal_segments_postICA.csv')
    
    if not os.path.exists(csv_path):
        print(f"⚠ marked_abnormal_segments_postICA.csv not found in: {result_dir}")
        print(f"  提示: 请先运行 process_marked_segments.py 生成此文件")
        return False
    
    # 获取对应的 _postICA.set 文件
    result_dir_name = os.path.basename(result_dir)
    if not result_dir_name.endswith('_results'):
        print(f"⚠ Directory name doesn't end with _results: {result_dir}")
        return False
    
    set_name = result_dir_name.replace('_results', '.set')
    parent_dir = os.path.dirname(result_dir)
    set_file = os.path.join(parent_dir, set_name)
    
    if not os.path.exists(set_file):
        print(f"⚠ SET file not found: {set_file}")
        return False
    
    print(f"\n{'='*80}")
    print(f"处理: {result_dir_name}")
    print(f"{'='*80}")
    
    # 读取标记的片段
    segments = read_marked_segments_postica(csv_path)
    if not segments:
        print("⚠ No segments found in CSV")
        return False
    
    print(f"总共 {len(segments)} 个标记片段")
    
    # 按通道分组
    grouped = group_segments_by_channels(segments)
    print(f"按通道分组后: {len(grouped)} 个不同的通道组合\n")
    
    # 为每个通道组创建合并的数据文件
    success_count = 0
    statistics = {}
    
    for channels_str, time_segments in grouped.items():
        print(f"通道组: {channels_str}")
        print(f"  包含 {len(time_segments)} 个片段")
        
        # 计算总时长
        total_duration = sum([end - start for start, end in time_segments])
        
        # 创建输出文件名
        # 通道名清理（去除特殊字符）
        safe_channels = channels_str.replace(',', '_').replace(' ', '').replace('-', '_')
        output_name = set_name.replace('.set', f'_merged_{safe_channels}.set')
        output_set_file = os.path.join(parent_dir, output_name)
        
        # 合并数据
        success = merge_segments_for_channel_group(
            set_file, time_segments, channels_str, output_set_file
        )
        
        if success:
            success_count += 1
            # 记录统计信息
            statistics[channels_str] = {
                'count': len(time_segments),
                'total_duration': total_duration,
                'output_file': output_name
            }
        print()
    
    print(f"{'='*80}")
    print(f"✅ 完成！成功创建 {success_count}/{len(grouped)} 个合并文件")
    print(f"{'='*80}")
    
    # 保存统计信息到 CSV
    if statistics:
        stats_csv_path = os.path.join(parent_dir, set_name.replace('.set', '_channel_merge_statistics.csv'))
        with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['通道组合', '片段数量', '总时长(秒)', '总时长(分钟)', '输出文件名'])
            
            # 按总时长排序（从长到短）
            sorted_stats = sorted(statistics.items(), key=lambda x: x[1]['total_duration'], reverse=True)
            for channels_str, info in sorted_stats:
                writer.writerow([
                    channels_str,
                    info['count'],
                    f"{info['total_duration']:.2f}",
                    f"{info['total_duration']/60:.2f}",
                    info['output_file']
                ])
        
        print(f"\n📊 统计信息已保存到: {stats_csv_path}")
        print(f"\n通道组统计 (按总时长排序):")
        print(f"{'通道组合':<30} {'片段数':<10} {'总时长(秒)':<15} {'总时长(分钟)':<15}")
        print("-" * 75)
        for channels_str, info in sorted_stats:
            print(f"{channels_str:<30} {info['count']:<10} {info['total_duration']:<15.2f} {info['total_duration']/60:<15.2f}")
        print()
    
    return True


def process_all_results(root_dir):
    """
    递归处理所有包含 marked_abnormal_segments_postICA.csv 的结果目录
    
    参数:
        root_dir: 根目录
    """
    processed_count = 0
    failed_count = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'marked_abnormal_segments_postICA.csv' in filenames:
            try:
                success = process_single_result_dir(dirpath)
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"❌ Error processing {dirpath}: {e}")
                failed_count += 1
    
    return processed_count, failed_count


def main():
    parser = argparse.ArgumentParser(
        description="按异常通道分组合并数据段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
功能说明:
  将相同异常通道列表的所有时间段合并到一个文件中，不考虑时间是否连续。
  例如：
    - 10-15s: Sph-R,T4
    - 25-30s: Sph-R,T4
    - 40-45s: O1,F7
  
  会生成:
    - xxx_merged_Sph_R_T4.set (包含 10-15s 和 25-30s 的数据拼接)
    - xxx_merged_O1_F7.set (包含 40-45s 的数据)

使用方法:
  # 处理单个结果目录
  python merge_by_channels.py --result_dir "E:\\data\\patient\\SZ1_results"
  
  # 批量处理所有结果目录
  python merge_by_channels.py --root_dir "E:\\DataSet\\EEG\\EEG dataset_SUAT_processed"

注意:
  需要先运行 process_marked_segments.py 生成 marked_abnormal_segments_postICA.csv
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--root_dir',
        help="根目录，递归处理所有包含 marked_abnormal_segments_postICA.csv 的目录"
    )
    group.add_argument(
        '--result_dir',
        help="单个结果目录路径（包含 marked_abnormal_segments_postICA.csv）"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("按异常通道分组合并数据段")
    print("=" * 80)
    
    if args.root_dir:
        print(f"\n扫描目录: {args.root_dir}\n")
        processed_count, failed_count = process_all_results(args.root_dir)
        
        print("\n" + "=" * 80)
        print(f"✅ 批量处理完成！")
        print(f"   成功: {processed_count} 个目录")
        if failed_count > 0:
            print(f"   失败: {failed_count} 个目录")
        print("=" * 80)
    
    elif args.result_dir:
        print(f"\n处理单个目录: {args.result_dir}\n")
        success = process_single_result_dir(args.result_dir)
        
        if not success:
            print("\n❌ 处理失败")
            return 1
    
    return 0


if __name__ == "__main__":
    # 如果不提供命令行参数，使用默认值
    import sys
    if len(sys.argv) == 1:
        print("使用默认参数运行...")
        sys.argv.extend([
            '--root_dir', r'E:\DataSet\EEG\EEG dataset_SUAT_processed'
        ])
    
    sys.exit(main())

