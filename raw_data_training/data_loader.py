"""
原始EEG数据加载器
功能：读取.set文件，按6秒窗口提取数据，跳过包含boundary的窗口
"""

import numpy as np
import mne
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


class EEGWindowExtractor:
    """EEG窗口提取器"""
    
    def __init__(self, window_size: float = 6.0, overlap: float = 0.0):
        """
        Args:
            window_size: 窗口大小（秒）
            overlap: 窗口重叠比例（0-1），0表示无重叠
        """
        self.window_size = window_size
        self.overlap = overlap
        
    def load_set_file(self, file_path: str) -> Tuple[np.ndarray, float, List]:
        """
        加载.set文件
        
        Args:
            file_path: .set文件路径
            
        Returns:
            data: EEG数据 (n_channels, n_times)
            sfreq: 采样率
            events: 事件列表
        """
        try:
            # 使用MNE读取EEGLAB .set文件
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            
            # 获取数据和采样率
            data = raw.get_data()  # shape: (n_channels, n_times)
            sfreq = raw.info['sfreq']
            
            # 获取事件信息
            events, event_dict = mne.events_from_annotations(raw, verbose=False)
            
            return data, sfreq, events
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            raise
    
    def has_boundary_in_window(self, events: np.ndarray, start_sample: int, 
                               end_sample: int, event_dict: dict) -> bool:
        """
        检查窗口内是否有boundary事件
        
        Args:
            events: MNE事件数组 (n_events, 3)
            start_sample: 窗口起始样本点
            end_sample: 窗口结束样本点
            event_dict: 事件字典
            
        Returns:
            是否包含boundary事件
        """
        if len(events) == 0:
            return False
        
        # 查找boundary事件的ID
        boundary_ids = [v for k, v in event_dict.items() if 'boundary' in k.lower()]
        
        # 检查窗口内的事件
        window_events = events[(events[:, 0] >= start_sample) & 
                               (events[:, 0] < end_sample)]
        
        if len(window_events) == 0:
            return False
        
        # 检查是否有boundary事件
        for event_id in boundary_ids:
            if event_id in window_events[:, 2]:
                return True
                
        return False
    
    def extract_windows(self, file_path: str) -> Tuple[np.ndarray, dict]:
        """
        从.set文件提取无boundary的窗口
        
        Args:
            file_path: .set文件路径
            
        Returns:
            windows: 窗口数据数组 (n_windows, n_channels, n_samples)
            info: 信息字典，包含采样率等
        """
        # 加载数据
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        data = raw.get_data()  # (n_channels, n_times)
        sfreq = raw.info['sfreq']
        
        # 获取事件
        try:
            events, event_dict = mne.events_from_annotations(raw, verbose=False)
        except:
            events = np.array([])
            event_dict = {}
        
        # 计算窗口参数
        window_samples = int(self.window_size * sfreq)
        step_samples = int(window_samples * (1 - self.overlap))
        
        n_channels, n_times = data.shape
        
        # 提取窗口
        windows = []
        current_pos = 0
        
        while current_pos + window_samples <= n_times:
            window_start = current_pos
            window_end = current_pos + window_samples
            
            # 检查是否包含boundary
            if len(events) > 0 and self.has_boundary_in_window(
                events, window_start, window_end, event_dict
            ):
                # 跳到boundary之后
                boundary_events = events[(events[:, 0] >= window_start) & 
                                        (events[:, 0] < window_end)]
                if len(boundary_events) > 0:
                    # 移动到第一个boundary之后
                    current_pos = int(boundary_events[0, 0]) + 1
                    continue
            
            # 提取窗口数据
            window_data = data[:, window_start:window_end]
            windows.append(window_data)
            
            # 移动到下一个窗口（无重叠）
            current_pos += step_samples
        
        if len(windows) == 0:
            print(f"Warning: No valid windows extracted from {file_path}")
            return np.array([]), {'sfreq': sfreq, 'n_channels': n_channels}
        
        windows = np.stack(windows, axis=0)  # (n_windows, n_channels, n_samples)
        
        info = {
            'sfreq': sfreq,
            'n_channels': n_channels,
            'n_windows': len(windows),
            'window_size': self.window_size,
            'window_samples': window_samples
        }
        
        return windows, info


def test_extractor():
    """测试窗口提取器"""
    import os
    
    # 测试文件路径
    test_file = r"E:\DataSet\EEG\EEG dataset_SUAT_processed\头皮数据-6例\曾静君\SZ1_preICA_reject_1_postICA_merged_F7_Fp1_Sph_L.set"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    
    # 创建提取器
    extractor = EEGWindowExtractor(window_size=6.0, overlap=0.0)
    
    # 提取窗口
    print(f"Loading: {test_file}")
    windows, info = extractor.extract_windows(test_file)
    
    print(f"\nExtraction results:")
    print(f"  Sampling rate: {info['sfreq']} Hz")
    print(f"  Number of channels: {info['n_channels']}")
    print(f"  Number of windows: {info['n_windows']}")
    print(f"  Window shape: {windows.shape}")
    print(f"  Window size: {info['window_size']} seconds ({info['window_samples']} samples)")
    
    return windows, info


if __name__ == "__main__":
    test_extractor()

