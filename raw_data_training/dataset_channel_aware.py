"""
通道感知的数据集
从labels.csv读取通道组合信息，创建通道掩码
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import re
from tqdm import tqdm
import pickle
import json
import ast

from data_loader import EEGWindowExtractor


class ChannelAwareEEGDataset(Dataset):
    """
    通道感知的EEG数据集
    
    关键改进：
    1. 从labels.csv读取channel_combination
    2. 创建通道掩码标记活跃通道
    3. 保留所有21个通道的数据
    """
    
    def __init__(
        self,
        data_root: str,
        labels_csv: str,
        window_size: float = 6.0,
        overlap: float = 0.0,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        transform=None,
        normalization: str = 'window_robust'
    ):
        self.data_root = Path(data_root)
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform
        self.normalization = normalization
        
        # 加载标签信息
        self.labels_df = pd.read_csv(labels_csv)
        print(f"加载了 {len(self.labels_df)} 条记录")
        
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
        self.channel_name_to_idx = {}  # 通道名到索引的映射
        self._prepare_dataset()
        
    def _parse_channel_combination(self, channel_str: str) -> List[str]:
        """
        解析channel_combination字符串
        
        Args:
            channel_str: 如 "[F7,Fp1,Sph_L]" 或 "['F7','Fp1','Sph_L']"
        
        Returns:
            通道名列表
        """
        try:
            # 尝试作为Python列表解析
            channels = ast.literal_eval(channel_str)
            return channels
        except:
            # 手动解析
            channel_str = channel_str.strip('[]')
            channels = [ch.strip().strip("'\"") for ch in channel_str.split(',')]
            return channels
    
    def _extract_label_from_filename(self, filename: str) -> int:
        """从文件名提取标签"""
        match = re.search(r'SZ(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    def _get_cache_path(self, file_path: str) -> Path:
        """获取缓存文件路径"""
        cache_name = file_path.replace('\\', '_').replace('/', '_').replace(':', '') + '.pkl'
        return self.cache_dir / cache_name
    
    def _load_windows_cached(self, file_path: str) -> Tuple[np.ndarray, dict]:
        """加载窗口数据（支持缓存）"""
        cache_path = self._get_cache_path(file_path)
        
        if self.use_cache and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data['windows'], cached_data['info']
            except Exception as e:
                print(f"缓存加载失败 {file_path}: {e}")
        
        # 提取窗口
        full_path = self.data_root / file_path
        windows, info = self.extractor.extract_windows(str(full_path))
        
        # 保存到缓存
        if self.use_cache and len(windows) > 0:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({'windows': windows, 'info': info}, f)
            except Exception as e:
                print(f"缓存保存失败 {file_path}: {e}")
        
        return windows, info
    
    def _build_channel_mapping(self, file_path: str) -> Dict[str, int]:
        """
        构建通道名到索引的映射
        通过读取.set文件的通道信息
        """
        try:
            import mne
            full_path = self.data_root / file_path
            raw = mne.io.read_raw_eeglab(str(full_path), preload=False, verbose=False)
            
            # 获取通道名
            ch_names = raw.ch_names
            
            # 创建映射
            mapping = {name: idx for idx, name in enumerate(ch_names)}
            
            return mapping
        except Exception as e:
            print(f"构建通道映射失败 {file_path}: {e}")
            return {}
    
    def _prepare_dataset(self):
        """准备数据集"""
        print("准备通道感知数据集...")
        
        for idx, row in tqdm(self.labels_df.iterrows(), total=len(self.labels_df)):
            file_path = row['data_file_path']
            
            # 检查文件是否存在
            full_path = self.data_root / file_path
            if not full_path.exists():
                print(f"文件不存在: {full_path}")
                continue
            
            # 提取标签
            label = self._extract_label_from_filename(file_path)
            
            # 解析活跃通道
            try:
                active_channels = self._parse_channel_combination(row['channel_combination'])
            except Exception as e:
                print(f"解析通道组合失败 {file_path}: {e}")
                continue
            
            # 构建通道映射（只在第一次时构建）
            if not self.channel_name_to_idx:
                self.channel_name_to_idx = self._build_channel_mapping(file_path)
                print(f"通道映射: {self.channel_name_to_idx}")
            
            # 加载窗口数据
            try:
                windows, info = self._load_windows_cached(file_path)
                
                if len(windows) == 0:
                    continue
                
                # 为每个窗口创建样本
                for window_idx in range(len(windows)):
                    self.samples.append({
                        'file_path': file_path,
                        'window_idx': window_idx,
                        'label': label,
                        'active_channels': active_channels,
                        'n_channels': info['n_channels'],
                        'sfreq': info['sfreq']
                    })
                    
            except Exception as e:
                print(f"处理文件出错 {file_path}: {e}")
                continue
        
        print(f"数据集准备完成: {len(self.samples)} 个窗口")
        
        # 统计标签分布
        labels = [s['label'] for s in self.samples]
        unique_labels = np.unique(labels)
        print(f"标签分布:")
        for label in unique_labels:
            count = labels.count(label)
            print(f"  Label {label}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        # 统计活跃通道数量分布
        n_active_channels = [len(s['active_channels']) for s in self.samples]
        unique_counts = np.unique(n_active_channels)
        print(f"\n活跃通道数量分布:")
        for count in unique_counts:
            n_samples = n_active_channels.count(count)
            print(f"  {count}个活跃通道: {n_samples} samples ({n_samples/len(n_active_channels)*100:.1f}%)")
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """归一化数据"""
        if self.normalization == 'none':
            return data
        
        elif self.normalization == 'window_zscore':
            mean = data.mean()
            std = data.std()
            return (data - mean) / (std + 1e-8)
        
        elif self.normalization == 'window_robust':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            return (data - median) / (mad * 1.4826 + 1e-8)
        
        elif self.normalization == 'channel_zscore':
            normalized = np.zeros_like(data)
            for ch in range(data.shape[0]):
                mean = data[ch].mean()
                std = data[ch].std()
                normalized[ch] = (data[ch] - mean) / (std + 1e-8)
            return normalized
        
        elif self.normalization == 'channel_robust':
            normalized = np.zeros_like(data)
            for ch in range(data.shape[0]):
                median = np.median(data[ch])
                mad = np.median(np.abs(data[ch] - median))
                normalized[ch] = (data[ch] - median) / (mad * 1.4826 + 1e-8)
            return normalized
        
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")
    
    def _create_channel_mask(self, active_channels: List[str], n_channels: int) -> np.ndarray:
        """
        创建通道掩码
        
        Args:
            active_channels: 活跃通道名列表
            n_channels: 总通道数
        
        Returns:
            mask: (n_channels,) binary数组，1表示活跃通道
        """
        mask = np.zeros(n_channels, dtype=np.float32)
        
        for ch_name in active_channels:
            if ch_name in self.channel_name_to_idx:
                idx = self.channel_name_to_idx[ch_name]
                if idx < n_channels:
                    mask[idx] = 1.0
            else:
                ch_name = ch_name.replace('_', '-')
                if ch_name in self.channel_name_to_idx:
                    idx = self.channel_name_to_idx[ch_name]
                    if idx < n_channels:
                        mask[idx] = 1.0
                else:
                    print(f"警告: 通道 {ch_name} 不在映射中")
        
        return mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本
        
        Returns:
            dict:
                - 'data': (n_channels, n_samples)
                - 'label': int
                - 'channel_mask': (n_channels,) binary mask
                - 'file_path': str
                - 'window_idx': int
                - 'n_active_channels': int
        """
        sample_info = self.samples[idx]
        
        # 加载窗口数据
        windows, _ = self._load_windows_cached(sample_info['file_path'])
        window_data = windows[sample_info['window_idx']]  # (n_channels, n_samples)
        
        # 归一化
        window_data = self._normalize_data(window_data)
        
        # 创建通道掩码
        channel_mask = self._create_channel_mask(
            sample_info['active_channels'],
            sample_info['n_channels']
        )
        
        # 转换为张量
        data = torch.FloatTensor(window_data)
        label = torch.LongTensor([sample_info['label']])[0]
        channel_mask = torch.FloatTensor(channel_mask)
        
        # 应用变换
        if self.transform is not None:
            data = self.transform(data)
        
        return {
            'data': data,
            'label': label,
            'channel_mask': channel_mask,
            'file_path': sample_info['file_path'],
            'window_idx': sample_info['window_idx'],
            'n_active_channels': len(sample_info['active_channels'])
        }


def create_channel_aware_dataloaders(
    data_root: str,
    labels_csv: str,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    window_size: float = 6.0,
    num_workers: int = 0,
    seed: int = 42,
    normalization: str = 'window_robust'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建通道感知的数据加载器"""
    
    # 创建数据集
    full_dataset = ChannelAwareEEGDataset(
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
    
    print(f"数据分割: Train={n_train}, Val={n_val}, Test={n_test}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试
    data_root = r'E:\DataSet\EEG\EEG dataset_SUAT_processed'
    labels_csv = r'E:\output\connectivity_features_v2\labels.csv'
    
    print("创建数据集...")
    dataset = ChannelAwareEEGDataset(
        data_root=data_root,
        labels_csv=labels_csv,
        window_size=6.0,
        use_cache=True,
        normalization='window_robust'
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        print("\n测试数据加载...")
        sample = dataset[0]
        print(f"Data shape: {sample['data'].shape}")
        print(f"Label: {sample['label']}")
        print(f"Channel mask shape: {sample['channel_mask'].shape}")
        print(f"Active channels: {sample['channel_mask'].sum().item()} / {sample['channel_mask'].shape[0]}")
        print(f"File: {sample['file_path']}")

