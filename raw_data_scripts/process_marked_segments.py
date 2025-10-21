#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_marked_segments.py

处理 marked_abnormal_segments.csv 文件：
1. 将原始时间转换回 postICA 数据的时间
2. 可选：根据标记的时间段裁剪并拼接数据

使用方法:
    # 只转换时间
    python process_marked_segments.py --root_dir "E:\DataSet\EEG\EEG dataset_SUAT_processed"
    
    # 转换时间并拼接数据
    python process_marked_segments.py --root_dir "E:\DataSet\EEG\EEG dataset_SUAT_processed" --extract
"""

import os
import csv
import argparse
import numpy as np
import mne
from pathlib import Path


def get_boundary_segments(raw):
    """从raw.annotations中提取所有boundary事件
    返回 [(start, end), ...] 在原始数据时间轴上的位置
    """
    try:
        sfreq = raw.info['sfreq']
        boundaries = []
        cumulative_offset = 0
        
        boundary_anns = [ann for ann in raw.annotations if ann['description'] == 'boundary']
        boundary_anns.sort(key=lambda x: x['onset'])
        
        for ann in boundary_anns:
            start_sample = ann['onset'] * sfreq + cumulative_offset
            dur_samples = ann['duration'] * sfreq
            end_sample = start_sample + dur_samples
            
            start_time = start_sample / sfreq
            end_time = end_sample / sfreq
            
            boundaries.append((start_time, end_time))
            cumulative_offset += dur_samples
        
        return boundaries
        
    except Exception as e:
        print(f"Warning: Could not extract boundary segments: {e}")
        return []


def calculate_cropped_time_from_original(original_time, raw):
    """将原始数据的时间转换回裁剪后数据的时间
    这是 calculate_original_time_from_cropped 的逆运算
    """
    try:
        sfreq = raw.info['sfreq']
        original_sample = original_time * sfreq
        
        # 按onset排序所有boundary annotations
        boundary_anns = [ann for ann in raw.annotations if ann['description'] == 'boundary']
        boundary_anns.sort(key=lambda x: x['onset'])
        
        # 计算在这个原始时间之前有多少删除的采样点
        cumulative_offset = 0
        for ann in boundary_anns:
            # boundary在原始数据中的位置
            boundary_original_start = ann['onset'] * sfreq + cumulative_offset
            
            if boundary_original_start <= original_sample:
                # 这个boundary在当前时间之前
                dur_samples = ann['duration'] * sfreq
                boundary_original_end = boundary_original_start + dur_samples
                
                if original_sample >= boundary_original_end:
                    # 完全在boundary之后，需要减去整个boundary的duration
                    cumulative_offset += dur_samples
                else:
                    # 在boundary内部，这个时间点在裁剪后的数据中不存在
                    # 返回boundary之前的裁剪后时间
                    cropped_sample = ann['onset'] * sfreq
                    return cropped_sample / sfreq, True  # True 表示在删除区域内
            else:
                break
        
        # 裁剪后时间 = 原始采样点 - 累计删除的采样点
        cropped_sample = original_sample - cumulative_offset
        cropped_time = cropped_sample / sfreq
        
        return cropped_time, False  # False 表示不在删除区域内
        
    except Exception as e:
        print(f"Warning: Could not calculate cropped time: {e}")
        return original_time, False


def convert_marked_csv_to_postica_time(csv_path, set_file, output_csv_path):
    """
    读取 marked_abnormal_segments.csv，将原始时间转换为 postICA 时间
    
    参数:
        csv_path: marked_abnormal_segments.csv 路径
        set_file: 对应的 _postICA.set 文件路径
        output_csv_path: 输出的 marked_abnormal_segments_postICA.csv 路径
    """
    if not os.path.exists(csv_path):
        print(f"⚠ CSV file not found: {csv_path}")
        return None
    
    if not os.path.exists(set_file):
        print(f"⚠ SET file not found: {set_file}")
        return None
    
    # 读取 EEG 数据
    raw = mne.io.read_raw_eeglab(set_file, preload=False, verbose='ERROR')
    
    # 读取 CSV
    segments = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        for row in reader:
            if len(row) >= 3:
                original_start = float(row[0])
                original_end = float(row[1])
                channels = row[2]
                segments.append((original_start, original_end, channels))
    
    if not segments:
        print(f"⚠ No segments found in: {csv_path}")
        return None
    
    # 转换时间
    converted_segments = []
    for original_start, original_end, channels in segments:
        cropped_start, in_boundary_start = calculate_cropped_time_from_original(original_start, raw)
        cropped_end, in_boundary_end = calculate_cropped_time_from_original(original_end, raw)
        
        if in_boundary_start or in_boundary_end:
            print(f"⚠ Warning: Segment {original_start:.2f}-{original_end:.2f}s overlaps with deleted boundary")
        
        converted_segments.append((cropped_start, cropped_end, channels))
    
    # 保存转换后的 CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['开始时间_postICA', '结束时间_postICA', '异常通道列表'])
        for start, end, channels in converted_segments:
            writer.writerow([start, end, channels])
    
    print(f"✓ Converted time saved to: {output_csv_path}")
    return converted_segments


def extract_and_concatenate_segments(set_file, segments, output_set_file):
    """
    根据标记的时间段裁剪并拼接 postICA 数据
    
    参数:
        set_file: 输入的 _postICA.set 文件路径
        segments: [(start, end, channels), ...] postICA 时间的片段列表
        output_set_file: 输出的 _result.set 文件路径
    """
    if not segments:
        print("⚠ No segments to extract")
        return False
    
    # 读取 EEG 数据
    print(f"Loading: {set_file}")
    raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose='ERROR')
    sfreq = raw.info['sfreq']
    
    # 按时间排序片段
    segments_sorted = sorted(segments, key=lambda x: x[0])
    
    # 提取所有片段的数据
    extracted_data_list = []
    for start, end, channels in segments_sorted:
        start_sample = int(start * sfreq)
        end_sample = int(end * sfreq)
        
        # 确保不越界
        start_sample = max(0, start_sample)
        end_sample = min(len(raw.times), end_sample)
        
        if start_sample >= end_sample:
            print(f"⚠ Invalid segment: {start:.2f}-{end:.2f}s")
            continue
        
        # 提取数据片段
        segment_data = raw.get_data(start=start_sample, stop=end_sample)
        extracted_data_list.append(segment_data)
        print(f"  Extracted: {start:.2f}s - {end:.2f}s ({end_sample - start_sample} samples)")
    
    if not extracted_data_list:
        print("⚠ No valid segments extracted")
        return False
    
    # 拼接所有片段
    concatenated_data = np.concatenate(extracted_data_list, axis=1)
    print(f"✓ Concatenated {len(extracted_data_list)} segments, total samples: {concatenated_data.shape[1]}")
    
    # 创建新的 Raw 对象
    info = raw.info.copy()
    new_raw = mne.io.RawArray(concatenated_data, info, verbose='ERROR')
    
    # 添加注释说明拼接的片段
    descriptions = []
    onsets = []
    durations = []
    cumulative_time = 0
    for i, (start, end, channels) in enumerate(segments_sorted):
        descriptions.append(f"Segment {i+1}: {channels}")
        onsets.append(cumulative_time)
        duration = end - start
        durations.append(duration)
        cumulative_time += duration
    
    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    new_raw.set_annotations(annotations)
    
    # 保存为 EEGLAB 格式
    print(f"Saving to: {output_set_file}")
    mne.export.export_raw(output_set_file, new_raw, fmt='eeglab', overwrite=True, verbose='ERROR')
    print(f"✓ Saved concatenated data to: {output_set_file}")
    
    return True


def process_single_result_dir(result_dir, extract_data=False):
    """
    处理单个结果目录
    
    参数:
        result_dir: 结果目录路径（包含 marked_abnormal_segments.csv）
        extract_data: 是否提取并拼接数据
    """
    csv_path = os.path.join(result_dir, 'marked_abnormal_segments.csv')
    
    if not os.path.exists(csv_path):
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
    
    print(f"\n处理: {result_dir_name}")
    print("-" * 80)
    
    # 1. 转换时间
    output_csv_path = os.path.join(result_dir, 'marked_abnormal_segments_postICA.csv')
    segments = convert_marked_csv_to_postica_time(csv_path, set_file, output_csv_path)
    
    if segments is None:
        return False
    
    # 2. 可选：提取并拼接数据
    if extract_data:
        output_set_name = set_name.replace('.set', '_result.set')
        output_set_file = os.path.join(parent_dir, output_set_name)
        success = extract_and_concatenate_segments(set_file, segments, output_set_file)
        if success:
            print(f"✓ Data extraction completed: {output_set_name}")
    
    return True


def process_all_results(root_dir, extract_data=False):
    """
    递归处理所有包含 marked_abnormal_segments.csv 的结果目录
    
    参数:
        root_dir: 根目录
        extract_data: 是否提取并拼接数据
    """
    processed_count = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'marked_abnormal_segments.csv' in filenames:
            success = process_single_result_dir(dirpath, extract_data)
            if success:
                processed_count += 1
    
    return processed_count


def main():
    parser = argparse.ArgumentParser(
        description="处理标记的异常时间段：转换时间并可选提取数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 只转换时间
  python process_marked_segments.py --root_dir "E:\\data"
  
  # 转换时间并提取拼接数据
  python process_marked_segments.py --root_dir "E:\\data" --extract
  
  # 处理单个结果目录
  python process_marked_segments.py --result_dir "E:\\data\\patient\\SZ1_results" --extract
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--root_dir',
        help="根目录，递归处理所有包含 marked_abnormal_segments.csv 的目录"
    )
    group.add_argument(
        '--result_dir',
        help="单个结果目录路径（包含 marked_abnormal_segments.csv）"
    )
    
    parser.add_argument(
        '--extract',
        action='store_true',
        help="提取并拼接标记的数据段，保存为 _result.set 文件"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("处理标记的异常时间段")
    print("=" * 80)
    
    if args.root_dir:
        print(f"\n扫描目录: {args.root_dir}")
        if args.extract:
            print("模式: 转换时间 + 提取数据")
        else:
            print("模式: 仅转换时间")
        print()
        
        processed_count = process_all_results(args.root_dir, args.extract)
        print("\n" + "=" * 80)
        print(f"✅ 完成！处理了 {processed_count} 个结果目录")
    
    elif args.result_dir:
        print(f"\n处理单个目录: {args.result_dir}")
        if args.extract:
            print("模式: 转换时间 + 提取数据")
        else:
            print("模式: 仅转换时间")
        print()
        
        success = process_single_result_dir(args.result_dir, args.extract)
        print("\n" + "=" * 80)
        if success:
            print("✅ 完成！")
        else:
            print("❌ 处理失败")


if __name__ == "__main__":
    # 如果不提供命令行参数，使用默认值
    import sys
    if len(sys.argv) == 1:
        print("使用默认参数运行...")
        sys.argv.extend([
            '--root_dir', r'E:\DataSet\EEG\EEG dataset_SUAT_processed',
            '--extract'
        ])
    
    main()

