#!/usr/bin/env python3
"""
过滤版多标签数据集 - 只使用有足够样本的通道

自动过滤掉 support < min_samples 的通道
"""

import numpy as np
import pandas as pd
import torch

from datasets_multilabel import (
    MultiLabelConnectivityDataset,
    discover_all_channels,
    load_labels_csv
)
from typing import List


def filter_channels_by_support(
    labels_df: pd.DataFrame,
    min_samples: int = 10,
    verbose: bool = True
) -> List[str]:
    """
    过滤掉样本数太少的通道
    
    Args:
        labels_df: 标签DataFrame
        min_samples: 最小样本数阈值
        verbose: 是否打印详细信息
    
    Returns:
        valid_channels: 有效通道列表
    """
    from collections import Counter
    
    # 发现所有通道
    all_channels = discover_all_channels(labels_df)
    
    # 统计每个通道的出现次数
    channel_counts = Counter()
    
    for combo_str in labels_df['channel_combination']:
        if not isinstance(combo_str, str):
            continue
        
        # 解析通道（复用相同的清理逻辑）
        cleaned = combo_str.strip()
        cleaned = cleaned.replace('[', '').replace(']', '')
        cleaned = cleaned.replace("'", '').replace('"', '')
        cleaned = cleaned.replace('(', '').replace(')', '')
        
        channels = []
        for sep in ['-', ',', ';', ' ', '|']:
            if sep in cleaned:
                channels = [ch.strip() for ch in cleaned.split(sep) if ch.strip()]
                break
        else:
            if cleaned.strip():
                channels = [cleaned.strip()]
        
        for ch in channels:
            ch = ''.join(c for c in ch if c.isalnum() or c in ['-', '_'])
            if ch and any(c.isalpha() for c in ch) and ch in all_channels:
                channel_counts[ch] += 1
    
    # 过滤
    valid_channels = [ch for ch in all_channels if channel_counts[ch] >= min_samples]
    filtered_channels = [ch for ch in all_channels if channel_counts[ch] < min_samples]
    
    if verbose:
        print(f"\n{'='*80}")
        print("Channel Filtering Results")
        print(f"{'='*80}")
        print(f"\nTotal channels before filtering: {len(all_channels)}")
        print(f"Channels with >= {min_samples} samples: {len(valid_channels)}")
        print(f"Filtered out: {len(filtered_channels)}")
        
        print(f"\n✓ Valid channels ({len(valid_channels)}):")
        for ch in sorted(valid_channels):
            count = channel_counts[ch]
            print(f"  {ch}: {count} samples")
        
        if filtered_channels:
            print(f"\n✗ Filtered channels ({len(filtered_channels)}):")
            for ch in sorted(filtered_channels):
                count = channel_counts[ch]
                print(f"  {ch}: {count} samples (< {min_samples})")
        
        print(f"\n{'='*80}")
    
    return valid_channels


class FilteredMultiLabelDataset(MultiLabelConnectivityDataset):
    """
    过滤版多标签数据集
    
    自动过滤掉样本太少的通道，提高训练效果
    
    重要: 这个数据集会过滤掉低频通道，但仍然能正确处理包含这些通道的样本
    """
    
    def __init__(
        self,
        npz_paths: List[str],
        labels_df: pd.DataFrame,
        min_samples: int = 10,
        **kwargs
    ):
        """
        Args:
            npz_paths: NPZ文件路径列表
            labels_df: 标签DataFrame
            min_samples: 最小样本数阈值（通道样本数少于此值将被过滤）
            **kwargs: 其他参数传递给父类
        """
        # 先发现所有通道（包括低频的）
        all_channels_before_filter = discover_all_channels(labels_df)
        
        # 过滤通道
        valid_channels = filter_channels_by_support(
            labels_df,
            min_samples=min_samples,
            verbose=True
        )
        
        # 记录被过滤的通道（用于警告抑制）
        self.filtered_channels = set(all_channels_before_filter) - set(valid_channels)
        
        # 用过滤后的通道初始化父类
        super().__init__(
            npz_paths=npz_paths,
            labels_df=labels_df,
            all_channels=valid_channels,
            **kwargs
        )
        
        self.min_samples = min_samples
        print(f"\n✓ Created FilteredMultiLabelDataset with {self.num_channels} channels")
        if self.filtered_channels:
            print(f"  Filtered out {len(self.filtered_channels)} low-frequency channels")
    
    def _parse_channel_combination(self, combo_str: str) -> torch.Tensor:
        """
        重写: 解析通道组合，忽略已被过滤的通道
        """
        label_vector = np.zeros(self.num_channels, dtype=np.float32)
        
        channels = self._parse_channels(combo_str)
        
        # 标记存在的通道（只标记未被过滤的）
        for ch in channels:
            if ch in self.channel_to_idx:
                # 在有效通道列表中
                label_vector[self.channel_to_idx[ch]] = 1.0
            elif ch in self.filtered_channels:
                # 被过滤的通道，静默忽略
                pass
            elif ch:  # 非空但既不在有效列表也不在过滤列表
                # 真正未知的通道，打印警告
                if not hasattr(self, '_warned_unknown_channels'):
                    self._warned_unknown_channels = set()
                if ch not in self._warned_unknown_channels and len(self._warned_unknown_channels) < 10:
                    print(f"Warning: Truly unknown channel '{ch}' (not in original data)")
                    self._warned_unknown_channels.add(ch)
        
        return torch.from_numpy(label_vector)


if __name__ == "__main__":
    print("Testing Filtered Multi-Label Dataset...")
    
    labels_csv = r'E:\output\connectivity_features\labels.csv'
    
    from pathlib import Path
    if Path(labels_csv).exists():
        labels_df = load_labels_csv(labels_csv)
        
        # 测试不同的阈值
        for min_samples in [5, 10, 20]:
            print(f"\n{'='*80}")
            print(f"Testing with min_samples={min_samples}")
            print(f"{'='*80}")
            
            valid_channels = filter_channels_by_support(
                labels_df,
                min_samples=min_samples,
                verbose=True
            )
            
            print(f"\nResult: {len(valid_channels)} valid channels")
    else:
        print(f"Labels file not found: {labels_csv}")

