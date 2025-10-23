#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_selected_segments.py

从_fast_reject_文件中提取被reject的数据段（boundary标记的区域）
并在原始postICA数据中找到对应的时间范围，保存为_selected.set文件

重要说明:
    - 在EEGLAB中reject某个片段后，该片段被删除，在删除位置插入boundary标记
    - boundary的onset是删除点在fast_reject文件中的位置
    - boundary的duration是被删除的数据长度
    - 本脚本提取的是boundary标记的那些被reject的片段（不是保留的片段！）

文件命名规则:
    输入: SZ3_postICA_fast_reject_Sph_R_F8_F7.set (reject后的文件)
    原始: SZ3_postICA.set (原始文件)
    输出: SZ3_postICA_selected_Sph_R_F8_F7.set (被reject的片段)
    
示例:
    如果在EEGLAB中对SZ3_postICA.set的100-130s片段做了reject：
    1. 保存为SZ3_postICA_fast_reject_Sph_R_F8_F7.set
    2. 运行本脚本后，提取100-130s的数据
    3. 保存为SZ3_postICA_selected_Sph_R_F8_F7.set
    
使用方法:
    python extract_selected_segments.py --source_dir "E:\DataSet\EEG\source" --target_dir "E:\DataSet\EEG\selected"
"""

import os
import argparse
import numpy as np
import mne
from pathlib import Path
from tqdm import tqdm

# 抑制MNE警告
mne.set_log_level('ERROR')


def parse_channel_names_from_filename(filename):
    """
    从文件名中解析被标记的通道名称
    
    例如: SZ2_postICA_fast_reject_T4_F8_Sph_R.set
    -> ['T4', 'F8', 'Sph-R']
    
    参数:
        filename: 文件名
    
    返回:
        channels: 通道名称列表
    """
    # 移除.set后缀
    name = filename.replace('.set', '')
    
    # 找到_fast_reject_的位置
    if '_fast_reject_' not in name:
        return []
    
    # 提取_fast_reject_之后的部分
    suffix = name.split('_fast_reject_')[1]
    
    # 按下划线分割得到通道名
    # 注意: Sph_R 需要恢复为 Sph-R
    parts = suffix.split('_')
    
    channels = []
    i = 0
    while i < len(parts):
        # 检查是否是 Sph/R 这种模式（应该合并为Sph-R）
        if i + 1 < len(parts) and parts[i] in ['Sph', 'Sphe'] and parts[i+1] in ['L', 'R']:
            channels.append(f"{parts[i]}-{parts[i+1]}")
            i += 2
        else:
            channels.append(parts[i])
            i += 1
    
    return channels


def get_original_filename(fast_reject_filename):
    """
    从fast_reject文件名获取原始postICA文件名
    
    例如: SZ2_postICA_fast_reject_T4_F8_Sph_R.set -> SZ2_postICA.set
    
    参数:
        fast_reject_filename: _fast_reject_文件名
    
    返回:
        original_filename: 原始postICA文件名
    """
    if '_fast_reject_' not in fast_reject_filename:
        return None
    
    # 提取_fast_reject_之前的部分
    base_name = fast_reject_filename.split('_fast_reject_')[0]
    return base_name + '.set'


def get_selected_filename(fast_reject_filename):
    """
    生成selected文件名
    
    例如: SZ2_postICA_fast_reject_T4_F8_Sph_R.set 
         -> SZ2_postICA_selected_T4_F8_Sph_R.set
    
    参数:
        fast_reject_filename: _fast_reject_文件名
    
    返回:
        selected_filename: _selected文件名
    """
    if '_fast_reject_' not in fast_reject_filename:
        return None
    
    return fast_reject_filename.replace('_fast_reject_', '_selected_')


def extract_rejected_segments(raw):
    """
    从MNE Raw对象中提取被reject的数据段（boundary标记的区域）
    
    在_fast_reject_文件中，boundary标记了被删除的区域
    我们需要提取的就是这些boundary本身（被reject的片段）
    
    参数:
        raw: MNE Raw对象
    
    返回:
        segments: list of dict, 每个segment包含:
            - start: 开始时间（秒）
            - end: 结束时间（秒）
            - start_sample: 开始样本索引
            - end_sample: 结束样本索引
            - duration: 持续时间（秒）
    """
    sfreq = raw.info['sfreq']
    
    # 获取所有boundary annotations - 这些就是被reject的片段
    boundaries = []
    for ann in raw.annotations:
        if ann['description'] == 'boundary':
            boundaries.append({
                'onset': ann['onset'],
                'duration': ann['duration']
            })
    
    # 按onset排序
    boundaries = sorted(boundaries, key=lambda x: x['onset'])
    
    segments = []
    
    # 如果没有boundary，说明没有被reject的数据
    if len(boundaries) == 0:
        print("  ⚠ 警告: 未找到boundary标记，可能没有被reject的数据")
        return segments
    
    # 提取每个boundary标记的区域（被reject的片段）
    for boundary in boundaries:
        start = boundary['onset']
        duration = boundary['duration']
        end = start + duration
        
        segments.append({
            'start': start,
            'end': end,
            'start_sample': int(start * sfreq),
            'end_sample': int(end * sfreq),
            'duration': duration
        })
    
    return segments


def map_segments_to_original(segments, fast_reject_raw, original_raw):
    """
    将fast_reject数据中被reject的片段映射到原始postICA数据的时间
    
    segments本身就是boundaries（被删除的区域）
    需要将boundary在fast_reject中的位置映射回原始数据的位置
    
    参数:
        segments: fast_reject数据中被reject的segments（boundary标记）
        fast_reject_raw: fast_reject的MNE Raw对象
        original_raw: 原始postICA的MNE Raw对象
    
    返回:
        original_segments: 映射到原始数据时间的segments
    """
    # 获取fast_reject中的所有boundaries（按onset排序）
    boundaries = []
    for ann in fast_reject_raw.annotations:
        if ann['description'] == 'boundary':
            boundaries.append({
                'onset': ann['onset'],
                'duration': ann['duration']
            })
    boundaries = sorted(boundaries, key=lambda x: x['onset'])
    
    original_segments = []
    
    # 对于每个被reject的片段（boundary）
    for seg in segments:
        # 计算在这个boundary之前有多少被删除的时间
        # 因为每次删除都会影响后续的时间轴
        cumulative_deleted = 0
        for boundary in boundaries:
            # 如果这个boundary在当前segment之前（用onset比较）
            if boundary['onset'] < seg['start']:
                cumulative_deleted += boundary['duration']
        
        # 原始时间 = fast_reject中的onset + 之前被删除的总时长
        # duration保持不变（就是boundary的duration）
        original_start = seg['start'] + cumulative_deleted
        original_end = original_start + seg['duration']  # 使用duration而不是end
        
        original_segments.append({
            'start': original_start,
            'end': original_end,
            'duration': seg['duration']
        })
    
    return original_segments


def extract_and_save_segments(original_raw, segments, output_file, marked_channels):
    """
    从原始数据中提取指定的segments并拼接保存
    
    参数:
        original_raw: 原始postICA的MNE Raw对象
        segments: 要提取的时间段列表
        output_file: 输出文件路径
        marked_channels: 被标记的通道列表
    """
    sfreq = original_raw.info['sfreq']
    
    # 提取所有segments的数据
    extracted_data_list = []
    total_samples = 0
    
    print(f"\n  提取数据段:")
    for i, seg in enumerate(segments):
        start_sample = int(seg['start'] * sfreq)
        end_sample = int(seg['end'] * sfreq)
        
        # 确保不越界
        start_sample = max(0, start_sample)
        end_sample = min(len(original_raw.times), end_sample)
        
        if start_sample >= end_sample:
            print(f"    ⚠ 片段{i+1}: 无效 ({seg['start']:.2f}s - {seg['end']:.2f}s)")
            continue
        
        # 提取数据
        segment_data = original_raw.get_data(start=start_sample, stop=end_sample)
        extracted_data_list.append(segment_data)
        n_samples = end_sample - start_sample
        total_samples += n_samples
        
        print(f"    ✓ 片段{i+1}: {seg['start']:.2f}s - {seg['end']:.2f}s "
              f"(持续 {seg['duration']:.2f}s, {n_samples} 采样点)")
    
    if not extracted_data_list:
        print("  ✗ 没有有效的数据段")
        return False
    
    # 拼接所有segments
    concatenated_data = np.concatenate(extracted_data_list, axis=1)
    print(f"\n  ✓ 拼接完成: {len(extracted_data_list)} 个片段, "
          f"总共 {total_samples} 采样点 ({total_samples/sfreq:.2f}秒)")
    
    # 创建新的Raw对象
    info = original_raw.info.copy()
    new_raw = mne.io.RawArray(concatenated_data, info, verbose='ERROR')
    
    # 添加annotations标记每个segment
    descriptions = []
    onsets = []
    durations = []
    cumulative_time = 0
    
    for i, seg in enumerate(segments):
        if seg['duration'] > 0:
            descriptions.append(f"Segment_{i+1}_Channels_{' '.join(marked_channels)}")
            onsets.append(cumulative_time)
            durations.append(seg['duration'])
            cumulative_time += seg['duration']
    
    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    new_raw.set_annotations(annotations)
    
    # 保存为EEGLAB格式
    print(f"\n  保存到: {output_file}")
    mne.export.export_raw(output_file, new_raw, fmt='eeglab', overwrite=True, verbose='ERROR')
    
    return True


def process_single_file(fast_reject_file, source_dir, target_dir):
    """
    处理单个_fast_reject_文件
    
    参数:
        fast_reject_file: _fast_reject_文件的完整路径
        source_dir: 源目录（包含原始postICA文件）
        target_dir: 目标目录（保存selected文件）
    """
    fast_reject_path = Path(fast_reject_file)
    filename = fast_reject_path.name
    
    # 解析通道名称
    marked_channels = parse_channel_names_from_filename(filename)
    if not marked_channels:
        print(f"  ✗ 无法从文件名解析通道: {filename}")
        return False
    
    print(f"  标记通道: {marked_channels}")
    
    # 获取原始文件名
    original_filename = get_original_filename(filename)
    if not original_filename:
        print(f"  ✗ 无法确定原始文件名")
        return False
    
    # 构建原始文件路径（保持相对目录结构）
    try:
        rel_path = fast_reject_path.relative_to(source_dir)
        original_path = Path(source_dir) / rel_path.parent / original_filename
    except ValueError:
        # 如果fast_reject_file不在source_dir下，就在同目录查找
        original_path = fast_reject_path.parent / original_filename
    
    if not original_path.exists():
        print(f"  ✗ 原始文件不存在: {original_path}")
        return False
    
    print(f"  原始文件: {original_path.name}")
    
    # 生成输出文件名
    selected_filename = get_selected_filename(filename)
    if not selected_filename:
        print(f"  ✗ 无法生成selected文件名")
        return False
    
    # 构建输出文件路径（保持相对目录结构）
    try:
        rel_dir = fast_reject_path.relative_to(source_dir).parent
        output_path = Path(target_dir) / rel_dir / selected_filename
    except ValueError:
        output_path = Path(target_dir) / selected_filename
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"  输出文件: {selected_filename}")
    
    # 读取fast_reject文件
    print(f"\n  读取fast_reject文件...")
    fast_reject_raw = mne.io.read_raw_eeglab(str(fast_reject_file), preload=False, verbose='ERROR')
    
    # 提取被reject的segments（boundary标记的区域）
    print(f"  提取被reject的数据段（boundary区域）...")
    segments = extract_rejected_segments(fast_reject_raw)
    
    if not segments:
        print(f"  ✗ 未找到有效的数据段")
        return False
    
    print(f"  ✓ 找到 {len(segments)} 个被reject的片段（boundary）:")
    for i, seg in enumerate(segments):
        print(f"    Boundary{i+1}: onset={seg['start']:.2f}s, duration={seg['duration']:.2f}s (在fast_reject文件中)")
    
    # 读取原始文件
    print(f"\n  读取原始postICA文件...")
    original_raw = mne.io.read_raw_eeglab(str(original_path), preload=True, verbose='ERROR')
    
    # 映射segments到原始数据的时间
    print(f"  映射到原始数据时间...")
    original_segments = map_segments_to_original(segments, fast_reject_raw, original_raw)
    
    print(f"  ✓ 映射到原始数据的时间段:")
    for i, seg in enumerate(original_segments):
        print(f"    被reject片段{i+1}: {seg['start']:.2f}s - {seg['end']:.2f}s (持续 {seg['duration']:.2f}s, 在原始数据中)")
    
    # 提取并保存
    success = extract_and_save_segments(original_raw, original_segments, str(output_path), marked_channels)
    
    if success:
        print(f"\n  ✓ 成功保存: {output_path}")
        return True
    else:
        print(f"\n  ✗ 保存失败")
        return False


def process_directory(source_dir, target_dir, pattern="*_fast_reject_*.set"):
    """
    递归处理目录中所有的_fast_reject_文件
    
    参数:
        source_dir: 源目录
        target_dir: 目标目录
        pattern: 文件匹配模式
    """
    source_path = Path(source_dir)
    
    # 查找所有匹配的文件
    fast_reject_files = list(source_path.rglob(pattern))
    
    if not fast_reject_files:
        print(f"\n未找到匹配的文件: {pattern}")
        return 0
    
    print(f"\n找到 {len(fast_reject_files)} 个文件\n")
    
    success_count = 0
    failed_count = 0
    
    for fast_reject_file in tqdm(fast_reject_files, desc="处理文件"):
        print(f"\n{'='*80}")
        print(f"处理: {fast_reject_file.name}")
        print(f"{'='*80}")
        
        try:
            success = process_single_file(str(fast_reject_file), source_dir, target_dir)
            if success:
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
    
    return success_count, failed_count


def main():
    parser = argparse.ArgumentParser(
        description="从_fast_reject_文件中提取被保留的数据段并保存为_selected文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理整个目录
  python extract_selected_segments.py \\
      --source_dir "E:\\DataSet\\EEG\\source" \\
      --target_dir "E:\\DataSet\\EEG\\selected"
  
  # 处理单个文件
  python extract_selected_segments.py \\
      --source_dir "E:\\DataSet\\EEG\\source" \\
      --target_dir "E:\\DataSet\\EEG\\selected" \\
      --single_file "E:\\DataSet\\EEG\\source\\SZ2_postICA_fast_reject_T4_F8_Sph_R.set"

说明:
  1. 脚本会查找所有*_fast_reject_*.set文件
  2. 从文件名中解析标记的异常通道
  3. 提取boundary标记的被reject数据段
  4. 在原始postICA文件中找到对应的时间范围
  5. 保存为*_selected_*.set文件
  
注意:
  - 提取的是被reject的片段，不是保留的片段！
  - boundary标记了被删除的区域
  - 这些片段包含异常活动，用于训练异常通道检测模型
        """
    )
    
    parser.add_argument(
        '--source_dir',
        required=True,
        help="源目录，包含_fast_reject_文件和原始postICA文件"
    )
    parser.add_argument(
        '--target_dir',
        required=True,
        help="目标目录，保存_selected文件"
    )
    parser.add_argument(
        '--single_file',
        default=None,
        help="只处理指定的单个文件"
    )
    parser.add_argument(
        '--pattern',
        default='*_fast_reject_*.set',
        help="文件匹配模式，默认: *_fast_reject_*.set"
    )
    
    args = parser.parse_args()
    
    # 检查源目录
    if not os.path.exists(args.source_dir):
        print(f"错误: 源目录不存在: {args.source_dir}")
        return 1
    
    # 创建目标目录
    Path(args.target_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("提取选定的数据段")
    print("="*80)
    print(f"源目录: {args.source_dir}")
    print(f"目标目录: {args.target_dir}")
    print(f"文件模式: {args.pattern}")
    
    # 处理文件
    if args.single_file:
        # 处理单个文件
        print(f"\n处理单个文件: {args.single_file}")
        if not os.path.exists(args.single_file):
            print(f"错误: 文件不存在: {args.single_file}")
            return 1
        
        success = process_single_file(args.single_file, args.source_dir, args.target_dir)
        
        print("\n" + "="*80)
        if success:
            print("✅ 处理完成！")
        else:
            print("❌ 处理失败")
    else:
        # 处理整个目录
        success_count, failed_count = process_directory(args.source_dir, args.target_dir, args.pattern)
        
        print("\n" + "="*80)
        print("处理完成！")
        print("="*80)
        print(f"成功: {success_count} 个文件")
        print(f"失败: {failed_count} 个文件")
        print(f"总计: {success_count + failed_count} 个文件")
    
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--source_dir', r'E:\DataSet\EEG\EEG dataset_SUAT_processed',
            '--target_dir', r'E:\DataSet\EEG\EEG dataset_SUAT_processed_selected',
        ])

    sys.exit(main())

