"""
过滤后的数据集 - 排除badcase
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
from typing import Optional, Set, Tuple

from dataset import RawEEGDataset


class FilteredRawEEGDataset(RawEEGDataset):
    """过滤后的原始EEG数据集 - 排除质量问题窗口"""
    
    def __init__(
        self,
        data_root: str,
        labels_csv: str,
        window_size: float = 6.0,
        overlap: float = 0.0,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        transform=None,
        bad_windows_file: Optional[str] = None,
        normalization: str = 'window_robust'
    ):
        """
        Args:
            bad_windows_file: JSON文件路径，包含要排除的窗口列表
        """
        # 加载bad windows
        self.bad_windows_set: Set[Tuple[str, int]] = set()
        
        if bad_windows_file and Path(bad_windows_file).exists():
            print(f"加载badcase过滤列表: {bad_windows_file}")
            with open(bad_windows_file, 'r') as f:
                data = json.load(f)
                
                if isinstance(data, dict) and 'bad_windows' in data:
                    bad_list = data['bad_windows']
                elif isinstance(data, dict) and 'filtered_dataset' in data:
                    # 旧格式兼容
                    bad_list = data.get('bad_windows', [])
                else:
                    bad_list = []
                
                # 转换为集合
                for item in bad_list:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        self.bad_windows_set.add((item[0], item[1]))
                    elif isinstance(item, dict):
                        self.bad_windows_set.add((item['file_path'], item['window_idx']))
            
            print(f"  已加载 {len(self.bad_windows_set)} 个badcase窗口")
        
        # 调用父类初始化
        super().__init__(
            data_root=data_root,
            labels_csv=labels_csv,
            window_size=window_size,
            overlap=overlap,
            cache_dir=cache_dir,
            use_cache=use_cache,
            transform=transform,
            normalization=normalization
        )
    
    def _prepare_dataset(self):
        """准备数据集 - 过滤bad windows"""
        print("准备过滤后的数据集...")
        
        # 首先调用父类方法准备所有样本
        super()._prepare_dataset()
        
        # 然后过滤
        if len(self.bad_windows_set) > 0:
            original_count = len(self.samples)
            
            self.samples = [
                sample for sample in self.samples
                if (sample['file_path'], sample['window_idx']) not in self.bad_windows_set
            ]
            
            filtered_count = original_count - len(self.samples)
            print(f"过滤了 {filtered_count} 个badcase窗口 "
                  f"({filtered_count/original_count*100:.1f}%)")
            print(f"剩余 {len(self.samples)} 个有效窗口")
            
            # 重新统计标签分布
            if len(self.samples) > 0:
                labels = [s['label'] for s in self.samples]
                unique_labels = np.unique(labels)
                print(f"过滤后标签分布:")
                for label in unique_labels:
                    count = labels.count(label)
                    print(f"  Label {label}: {count} samples ({count/len(labels)*100:.1f}%)")


def create_filtered_dataloaders(
    data_root: str,
    labels_csv: str,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    window_size: float = 6.0,
    num_workers: int = 0,
    seed: int = 42,
    bad_windows_file: Optional[str] = None,
    normalization: str = 'window_robust'
):
    """创建过滤后的数据加载器"""
    
    from torch.utils.data import DataLoader, random_split
    
    # 创建过滤后的数据集
    full_dataset = FilteredRawEEGDataset(
        data_root=data_root,
        labels_csv=labels_csv,
        window_size=window_size,
        use_cache=True,
        bad_windows_file=bad_windows_file,
        normalization=normalization
    )
    
    if len(full_dataset) == 0:
        raise ValueError("过滤后数据集为空！")
    
    # 计算分割大小
    n_samples = len(full_dataset)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    # 随机分割
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
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

