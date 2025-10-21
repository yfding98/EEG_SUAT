#!/usr/bin/env python3
"""
简化版EDF通道检测脚本
直接修改脚本中的路径即可使用
"""

import mne
import numpy as np
from pathlib import Path
import warnings
from collections import Counter

# 忽略警告
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ==================== 配置区域 ====================
# 修改这里的路径为您的EDF文件目录
EDF_DIRECTORY = "E:/DataSet/EEG/EEG dataset_SUAT"
# ==================================================

# 标准EEG通道配置
STANDARD_EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'Fz', 'Cz', 'Pz','Sph-R', 'Sph-L','Oz'
]

AUXILIARY_CHANNELS = [
    'A1', 'A2', 'ECG', 'EMG1', 'EMG2',
    '27', '28', '29', '30', '31', '32', 'DC', 'OSat', 'PR'
]

def check_edf_file(edf_path):
    """检查单个EDF文件"""
    try:
        print(f"\n检测文件: {edf_path.name}")
        print(f"路径: {edf_path}")
        
        # 读取EDF文件
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        
        # 获取通道信息
        channels = raw.ch_names
        channel_count = len(channels)
        sampling_rate = raw.info['sfreq']
        duration = raw.times[-1] if len(raw.times) > 0 else 0
        
        print(f"✓ 通道数量: {channel_count}")
        print(f"✓ 采样频率: {sampling_rate} Hz")
        print(f"✓ 录制时长: {duration:.1f} 秒")
        print(f"✓ 所有通道: {channels}")
        
        # 分析通道类型
        standard_channels = [ch for ch in channels if ch in STANDARD_EEG_CHANNELS]
        auxiliary_channels = [ch for ch in channels if ch in AUXILIARY_CHANNELS]
        unknown_channels = [ch for ch in channels if ch not in STANDARD_EEG_CHANNELS + AUXILIARY_CHANNELS]
        
        print(f"\n通道分类:")
        print(f"  标准EEG通道 ({len(standard_channels)}): {standard_channels}")
        print(f"  辅助通道 ({len(auxiliary_channels)}): {auxiliary_channels}")
        if unknown_channels:
            print(f"  未知通道 ({len(unknown_channels)}): {unknown_channels}")
        
        # 检查缺失的标准通道
        missing_standard = [ch for ch in STANDARD_EEG_CHANNELS if ch not in channels]
        if missing_standard:
            print(f"  ⚠️ 缺失的标准通道 ({len(missing_standard)}): {missing_standard}")
        else:
            print(f"  ✓ 所有标准EEG通道都存在!")
        
        # 评估数据质量
        eeg_coverage = len(standard_channels) / len(STANDARD_EEG_CHANNELS) * 100
        print(f"\nEEG通道覆盖率: {eeg_coverage:.1f}%")
        
        if eeg_coverage >= 90:
            print("✓ 数据质量: 优秀 - 几乎所有标准通道都存在")
        elif eeg_coverage >= 70:
            print("⚠️ 数据质量: 良好 - 大部分标准通道存在")
        elif eeg_coverage >= 50:
            print("⚠️ 数据质量: 一般 - 部分标准通道缺失")
        else:
            print("❌ 数据质量: 较差 - 许多标准通道缺失")
        
        raw.close()
        
        return {
            'success': True,
            'channels': channels,
            'standard_channels': standard_channels,
            'missing_channels': missing_standard,
            'channel_count': channel_count,
            'sampling_rate': sampling_rate,
            'duration': duration,
            'coverage': eeg_coverage
        }
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """主函数"""
    print("="*80)
    print("EDF文件通道检测工具")
    print("="*80)
    
    # 检查目录是否存在
    directory = Path(EDF_DIRECTORY)
    if not directory.exists():
        print(f"❌ 错误: 目录不存在 - {directory}")
        print("请修改脚本开头的 EDF_DIRECTORY 变量为正确的路径")
        return
    
    # 查找所有EDF文件
    edf_files = list(directory.rglob("*.edf"))
    print(f"在 {directory} 中找到 {len(edf_files)} 个EDF文件")
    
    if len(edf_files) == 0:
        print("未找到EDF文件")
        return
    
    # 检测结果统计
    results = []
    all_channels = []
    
    # 检测每个文件
    for i, edf_file in enumerate(edf_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(edf_files)}] {edf_file.relative_to(directory)}")
        
        result = check_edf_file(edf_file)
        results.append(result)
        
        if result['success']:
            all_channels.extend(result['channels'])
    
    # 生成汇总报告
    print(f"\n{'='*80}")
    print("汇总报告")
    print("="*80)
    
    successful_results = [r for r in results if r['success']]
    
    print(f"总文件数: {len(edf_files)}")
    print(f"成功解析: {len(successful_results)}")
    print(f"解析失败: {len(edf_files) - len(successful_results)}")
    
    if successful_results:
        # 通道统计
        channel_counts = [r['channel_count'] for r in successful_results]
        coverages = [r['coverage'] for r in successful_results]
        
        print(f"\n通道数量统计:")
        print(f"  最少: {min(channel_counts)} 个")
        print(f"  最多: {max(channel_counts)} 个")
        print(f"  平均: {np.mean(channel_counts):.1f} 个")
        
        print(f"\nEEG通道覆盖率统计:")
        print(f"  最低: {min(coverages):.1f}%")
        print(f"  最高: {max(coverages):.1f}%")
        print(f"  平均: {np.mean(coverages):.1f}%")
        
        # 通道出现频率
        channel_frequency = Counter(all_channels)
        print(f"\n通道出现频率:")
        for channel, count in channel_frequency.most_common():
            percentage = count / len(successful_results) * 100
            if channel in STANDARD_EEG_CHANNELS:
                ch_type = "[标准EEG]"
            elif channel in AUXILIARY_CHANNELS:
                ch_type = "[辅助]"
            else:
                ch_type = "[未知]"
            print(f"  {channel:8s}: {count:3d}/{len(successful_results)} ({percentage:5.1f}%) {ch_type}")
        
        # 推荐配置
        print(f"\n推荐处理配置:")
        print(f"  保留通道: {[ch for ch, count in channel_frequency.most_common() if ch in STANDARD_EEG_CHANNELS and count > len(successful_results) * 0.8]}")
        print(f"  删除通道: {[ch for ch, count in channel_frequency.most_common() if ch in AUXILIARY_CHANNELS and count > 0]}")

if __name__ == "__main__":
    main() 