"""
原始EEG数据的PyTorch数据集
支持时空特征的训练
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
from tqdm import tqdm
import pickle
import os

from data_loader import EEGWindowExtractor


class RawEEGDataset(Dataset):
    """原始EEG数据集"""
    
    def __init__(
        self, 
        data_root: str,
        labels_csv: str,
        window_size: float = 6.0,
        overlap: float = 0.0,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        transform=None,
        normalization: str = 'window_robust'  # 'none', 'window_zscore', 'window_robust', 'channel_zscore'
    ):
        """
        Args:
            data_root: 数据根目录
            labels_csv: 标签CSV文件路径
            window_size: 窗口大小（秒）
            overlap: 窗口重叠比例
            cache_dir: 缓存目录
            use_cache: 是否使用缓存
            transform: 数据增强/预处理
        """
        self.data_root = Path(data_root)
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform
        self.normalization = normalization
        
        # 加载标签信息
        self.labels_df = pd.read_csv(labels_csv)
        print(f"Loaded {len(self.labels_df)} entries from labels CSV")
        
        # 提取器
        self.extractor = EEGWindowExtractor(window_size=window_size, overlap=overlap)
        
        # 缓存设置
        self.use_cache = use_cache
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        self.samples = []
        self._prepare_dataset()
        
    def _extract_label_from_filename(self, filename: str) -> int:
        """
        从文件名提取标签
        例如: SZ1 -> 1, SZ4 -> 4
        
        Args:
            filename: 文件名
            
        Returns:
            标签值
        """
        # 提取SZ后面的数字
        match = re.search(r'SZ(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            # 如果没有找到，默认返回0
            return 0
    
    def _get_cache_path(self, file_path: str) -> Path:
        """获取缓存文件路径"""
        # 创建基于文件路径的唯一缓存名
        cache_name = file_path.replace('\\', '_').replace('/', '_').replace(':', '') + '.pkl'
        return self.cache_dir / cache_name
    
    def _load_windows_cached(self, file_path: str) -> Tuple[np.ndarray, dict]:
        """加载窗口数据（支持缓存）"""
        cache_path = self._get_cache_path(file_path)
        
        # 尝试从缓存加载
        if self.use_cache and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data['windows'], cached_data['info']
            except Exception as e:
                print(f"Failed to load cache for {file_path}: {e}")
        
        # 提取窗口
        full_path = self.data_root / file_path
        windows, info = self.extractor.extract_windows(str(full_path))
        
        # 保存到缓存
        if self.use_cache and len(windows) > 0:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({'windows': windows, 'info': info}, f)
            except Exception as e:
                print(f"Failed to save cache for {file_path}: {e}")
        
        return windows, info
    
    def _prepare_dataset(self):
        """准备数据集"""
        print("Preparing dataset...")
        
        for idx, row in tqdm(self.labels_df.iterrows(), total=len(self.labels_df)):
            file_path = row['data_file_path']
            
            # 检查文件是否存在
            full_path = self.data_root / file_path
            if not full_path.exists():
                print(f"File not found: {full_path}")
                continue
            
            # 提取标签
            label = self._extract_label_from_filename(file_path)
            
            # 加载窗口数据
            try:
                windows, info = self._load_windows_cached(file_path)
                
                if len(windows) == 0:
                    continue
                
                # 为每个窗口创建一个样本
                for window_idx in range(len(windows)):
                    self.samples.append({
                        'file_path': file_path,
                        'window_idx': window_idx,
                        'label': label,
                        'n_channels': info['n_channels'],
                        'sfreq': info['sfreq']
                    })
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        print(f"Dataset prepared: {len(self.samples)} windows from {len(self.labels_df)} files")
        
        # 统计标签分布
        labels = [s['label'] for s in self.samples]
        unique_labels = np.unique(labels)
        print(f"Label distribution:")
        for label in unique_labels:
            count = labels.count(label)
            print(f"  Label {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        归一化数据
        
        Args:
            data: (n_channels, n_samples)
        
        Returns:
            归一化后的数据
        """
        if self.normalization == 'none':
            return data
        
        elif self.normalization == 'window_zscore':
            # 窗口级Z-score归一化（所有通道一起）
            mean = data.mean()
            std = data.std()
            return (data - mean) / (std + 1e-8)
        
        elif self.normalization == 'window_robust':
            # 窗口级Robust归一化（对异常值更鲁棒）
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            return (data - median) / (mad * 1.4826 + 1e-8)  # 1.4826使MAD等价于标准差
        
        elif self.normalization == 'channel_zscore':
            # 通道独立Z-score归一化
            normalized = np.zeros_like(data)
            for ch in range(data.shape[0]):
                mean = data[ch].mean()
                std = data[ch].std()
                normalized[ch] = (data[ch] - mean) / (std + 1e-8)
            return normalized
        
        elif self.normalization == 'channel_robust':
            # 通道独立Robust归一化
            normalized = np.zeros_like(data)
            for ch in range(data.shape[0]):
                median = np.median(data[ch])
                mad = np.median(np.abs(data[ch] - median))
                normalized[ch] = (data[ch] - median) / (mad * 1.4826 + 1e-8)
            return normalized
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本
        
        Returns:
            dict with keys:
                - 'data': EEG窗口数据 (n_channels, n_samples)
                - 'label': 标签
                - 'file_path': 文件路径
                - 'window_idx': 窗口索引
        """
        sample_info = self.samples[idx]
        
        # 加载窗口数据
        windows, _ = self._load_windows_cached(sample_info['file_path'])
        window_data = windows[sample_info['window_idx']]  # (n_channels, n_samples)
        
        # 归一化
        window_data = self._normalize_data(window_data)
        
        # 转换为张量
        data = torch.FloatTensor(window_data)
        label = torch.LongTensor([sample_info['label']])[0]
        
        # 应用变换
        if self.transform is not None:
            data = self.transform(data)
        
        return {
            'data': data,
            'label': label,
            'file_path': sample_info['file_path'],
            'window_idx': sample_info['window_idx']
        }


def create_dataloaders(
    data_root: str,
    labels_csv: str,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    window_size: float = 6.0,
    num_workers: int = 4,
    seed: int = 42,
    normalization: str = 'window_robust'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        data_root: 数据根目录
        labels_csv: 标签CSV文件
        batch_size: 批量大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        window_size: 窗口大小
        num_workers: 数据加载工作进程数
        seed: 随机种子
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建完整数据集
    full_dataset = RawEEGDataset(
        data_root=data_root,
        labels_csv=labels_csv,
        window_size=window_size,
        use_cache=True,
        normalization=normalization
    )
    
    # 计算分割大小
    n_samples = len(full_dataset)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    # 随机分割
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Data split: Train={n_train}, Val={n_val}, Test={n_test}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集
    data_root = r"E:\DataSet\EEG\EEG dataset_SUAT_processed"
    labels_csv = r"E:\output\connectivity_features\labels.csv"
    
    print("Creating dataset...")
    dataset = RawEEGDataset(
        data_root=data_root,
        labels_csv=labels_csv,
        window_size=6.0,
        use_cache=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        print("\nTesting data loading...")
        sample = dataset[0]
        print(f"Data shape: {sample['data'].shape}")
        print(f"Label: {sample['label']}")
        print(f"File: {sample['file_path']}")

