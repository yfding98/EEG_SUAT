#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frequency_band_dataset.py

频段特定的数据集类
为每个频段创建独立的数据集
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pickle
import json
from typing import List, Dict, Tuple, Optional
import random
from scipy import signal
from scipy.signal import butter, filtfilt


class FrequencyBandDataset(Dataset):
    """频段特定的数据集"""
    
    def __init__(
        self,
        data_root: str,
        frequency_band: str,
        window_size: float = 6.0,
        window_stride: float = 3.0,
        sampling_rate: int = 250,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42
    ):
        self.data_root = Path(data_root)
        self.frequency_band = frequency_band
        self.window_size = window_size
        self.window_stride = window_stride
        self.sampling_rate = sampling_rate
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 频段定义
        self.frequency_bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 100.0)
        }
        
        if frequency_band not in self.frequency_bands:
            raise ValueError(f"不支持的频段: {frequency_band}")
        
        self.band_range = self.frequency_bands[frequency_band]
        
        # 加载和预处理数据
        self._load_data()
        
        # 分割数据集
        self._split_data(val_split, test_split)
        
        print(f"频段 {frequency_band} 数据集加载完成:")
        print(f"  总样本数: {len(self.all_data)}")
        print(f"  训练集: {len(self.train_data)}")
        print(f"  验证集: {len(self.val_data)}")
        print(f"  测试集: {len(self.test_data)}")
        print(f"  频段范围: {self.band_range[0]}-{self.band_range[1]} Hz")
    
    def _load_data(self):
        """加载原始数据"""
        self.all_data = []
        self.channel_names = None
        
        # 查找所有数据文件
        data_files = list(self.data_root.glob("*.pkl"))
        if not data_files:
            raise FileNotFoundError(f"在 {self.data_root} 中未找到数据文件")
        
        print(f"找到 {len(data_files)} 个数据文件")
        
        for file_path in data_files:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # 解析文件名获取异常通道信息
                abnormal_channels = self._parse_abnormal_channels(file_path.name)
                
                # 提取频段数据
                band_data = self._extract_frequency_band(data, self.frequency_band)
                
                if band_data is not None:
                    # 创建窗口
                    windows = self._create_windows(band_data, abnormal_channels)
                    self.all_data.extend(windows)
                    
                    if self.channel_names is None:
                        self.channel_names = data.get('channel_names', [f'ch_{i}' for i in range(band_data.shape[1])])
                
            except Exception as e:
                print(f"警告: 加载文件 {file_path} 失败: {e}")
                continue
        
        if not self.all_data:
            raise ValueError("未成功加载任何数据")
        
        print(f"成功加载 {len(self.all_data)} 个样本")
    
    def _parse_abnormal_channels(self, filename: str) -> List[int]:
        """从文件名解析异常通道"""
        # 假设文件名格式包含异常通道信息
        # 例如: "patient_001_channels_1_3_5.pkl"
        abnormal_channels = []
        
        try:
            # 提取通道信息
            if 'channels_' in filename:
                channel_part = filename.split('channels_')[1].split('.')[0]
                channel_indices = channel_part.split('_')
                abnormal_channels = [int(ch) for ch in channel_indices if ch.isdigit()]
        except:
            # 如果解析失败，返回空列表
            pass
        
        return abnormal_channels
    
    def _extract_frequency_band(self, data: Dict, frequency_band: str) -> Optional[np.ndarray]:
        """提取特定频段的数据"""
        try:
            # 获取EEG数据
            eeg_data = data.get('eeg_data')
            if eeg_data is None:
                return None
            
            # 确保数据是numpy数组
            if not isinstance(eeg_data, np.ndarray):
                eeg_data = np.array(eeg_data)
            
            # 设计带通滤波器
            low_freq, high_freq = self.frequency_bands[frequency_band]
            nyquist = self.sampling_rate / 2
            
            # 确保频率在奈奎斯特频率范围内
            low_freq = max(low_freq, 0.1)
            high_freq = min(high_freq, nyquist - 1)
            
            if low_freq >= high_freq:
                return None
            
            # 设计巴特沃斯滤波器
            b, a = butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
            
            # 应用滤波器
            filtered_data = np.zeros_like(eeg_data)
            for ch in range(eeg_data.shape[1]):
                filtered_data[:, ch] = filtfilt(b, a, eeg_data[:, ch])
            
            return filtered_data
            
        except Exception as e:
            print(f"警告: 频段 {frequency_band} 数据提取失败: {e}")
            return None
    
    def _create_windows(self, data: np.ndarray, abnormal_channels: List[int]) -> List[Dict]:
        """创建时间窗口"""
        windows = []
        
        window_samples = int(self.window_size * self.sampling_rate)
        stride_samples = int(self.window_stride * self.sampling_rate)
        
        for start_idx in range(0, data.shape[0] - window_samples + 1, stride_samples):
            end_idx = start_idx + window_samples
            
            # 提取窗口数据
            window_data = data[start_idx:end_idx]
            
            # 创建标签（多标签）
            n_channels = window_data.shape[1]
            labels = np.zeros(n_channels, dtype=np.float32)
            
            # 标记异常通道
            for ch_idx in abnormal_channels:
                if 0 <= ch_idx < n_channels:
                    labels[ch_idx] = 1.0
            
            # 检查是否有异常通道
            if np.sum(labels) > 0:
                windows.append({
                    'data': window_data,
                    'labels': labels,
                    'abnormal_channels': abnormal_channels
                })
        
        return windows
    
    def _split_data(self, val_split: float, test_split: float):
        """分割数据集"""
        # 随机打乱数据
        random.shuffle(self.all_data)
        
        n_total = len(self.all_data)
        n_val = int(n_total * val_split)
        n_test = int(n_total * test_split)
        n_train = n_total - n_val - n_test
        
        self.train_data = self.all_data[:n_train]
        self.val_data = self.all_data[n_train:n_train + n_val]
        self.test_data = self.all_data[n_train + n_val:]
        
        print(f"数据分割: 训练集={n_train}, 验证集={n_val}, 测试集={n_test}")
    
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        sample = self.train_data[idx]
        
        # 转换为张量
        data_tensor = torch.FloatTensor(sample['data']).T  # (channels, time)
        labels_tensor = torch.FloatTensor(sample['labels'])
        
        return {
            'data': data_tensor,
            'labels': labels_tensor,
            'abnormal_channels': sample['abnormal_channels']
        }
    
    def get_val_data(self):
        """获取验证集数据"""
        val_data = []
        for sample in self.val_data:
            data_tensor = torch.FloatTensor(sample['data']).T
            labels_tensor = torch.FloatTensor(sample['labels'])
            val_data.append({
                'data': data_tensor,
                'labels': labels_tensor,
                'abnormal_channels': sample['abnormal_channels']
            })
        return val_data
    
    def get_test_data(self):
        """获取测试集数据"""
        test_data = []
        for sample in self.test_data:
            data_tensor = torch.FloatTensor(sample['data']).T
            labels_tensor = torch.FloatTensor(sample['labels'])
            test_data.append({
                'data': data_tensor,
                'labels': labels_tensor,
                'abnormal_channels': sample['abnormal_channels']
            })
        return test_data


def create_frequency_band_dataloaders(
    data_root: str,
    frequency_band: str,
    batch_size: int = 8,
    window_size: float = 6.0,
    window_stride: float = 3.0,
    sampling_rate: int = 250,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """创建频段特定的数据加载器"""
    
    # 创建数据集
    dataset = FrequencyBandDataset(
        data_root=data_root,
        frequency_band=frequency_band,
        window_size=window_size,
        window_stride=window_stride,
        sampling_rate=sampling_rate,
        val_split=val_split,
        test_split=test_split,
        seed=seed
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 验证集和测试集数据加载器
    val_data = dataset.get_val_data()
    test_data = dataset.get_test_data()
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, dataset.channel_names

