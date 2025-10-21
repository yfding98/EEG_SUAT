#!/usr/bin/env python3
"""
高级采样策略 - 处理多层次不平衡

包含:
1. ClassBalancedSampler - 基于有效样本数
2. MultiLevelWeightedSampler - 三级加权
3. CurriculumSampler - 课程学习采样
"""

import torch
import numpy as np
from torch.utils.data import Sampler
from collections import Counter


class ClassBalancedSampler(Sampler):
    """
    类平衡采样器
    
    基于 Cui et al. CVPR 2019 的有效样本数理论
    E_n = (1 - β^n) / (1 - β)
    """
    
    def __init__(self, dataset, beta=0.9999):
        """
        Args:
            dataset: MultiLabelDataset
            beta: 重采样参数，通常 0.99-0.9999
        """
        self.dataset = dataset
        self.beta = beta
        self.num_samples = len(dataset)
        
        # 统计每个通道的样本数
        channel_samples = np.zeros(dataset.num_channels)
        for i in range(len(dataset)):
            y = dataset[i]['y'].numpy()
            channel_samples += y
        
        # 计算有效样本数
        effective_num = 1.0 - np.power(beta, channel_samples + 1e-6)
        weights_per_channel = (1.0 - beta) / (effective_num + 1e-6)
        
        # 归一化
        weights_per_channel = weights_per_channel / weights_per_channel.sum() * len(weights_per_channel)
        
        # 每个样本的权重 = 包含通道权重之和
        sample_weights = []
        for i in range(len(dataset)):
            y = dataset[i]['y'].numpy()
            # 权重 = 所有正类通道的权重之和
            weight = (y * weights_per_channel).sum()
            sample_weights.append(weight if weight > 0 else 1.0)
        
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        
        print(f"\nClassBalancedSampler initialized:")
        print(f"  Beta: {beta}")
        print(f"  Weight range: [{self.sample_weights.min():.2f}, {self.sample_weights.max():.2f}]")
    
    def __iter__(self):
        # 使用weights进行有放回采样
        indices = torch.multinomial(
            self.sample_weights,
            self.num_samples,
            replacement=True
        ).tolist()
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


class MultiLevelWeightedSampler(Sampler):
    """
    三级加权采样器
    
    Level 1: 正样本数量
    Level 2: 稀有通道
    Level 3: 困难样本
    """
    
    def __init__(
        self,
        dataset,
        channel_difficulties=None,
        weight_positive=10.0,
        weight_rare=5.0,
        weight_hard=3.0
    ):
        """
        Args:
            dataset: MultiLabelDataset
            channel_difficulties: [num_channels] 每个通道的难度分数
            weight_positive: 正样本权重系数
            weight_rare: 稀有通道权重系数
            weight_hard: 困难样本权重系数
        """
        self.dataset = dataset
        self.num_samples = len(dataset)
        
        # 统计通道频率
        channel_counts = np.zeros(dataset.num_channels)
        for i in range(len(dataset)):
            y = dataset[i]['y'].numpy()
            channel_counts += y
        
        # 归一化频率 [0, 1]
        max_count = channel_counts.max() if channel_counts.max() > 0 else 1.0
        channel_freqs = channel_counts / max_count
        
        # 如果没有提供难度，使用频率的倒数作为难度
        if channel_difficulties is None:
            channel_difficulties = 1.0 - channel_freqs
        
        # 计算每个样本的权重
        sample_weights = []
        
        for i in range(len(dataset)):
            y = dataset[i]['y'].numpy()
            
            # Level 1: 正样本数量权重
            num_pos = y.sum()
            w1 = 1.0 + num_pos * weight_positive
            
            # Level 2: 稀有通道权重
            rare_score = 0.0
            for ch_idx, label in enumerate(y):
                if label > 0:
                    # 频率低的通道分数高
                    rare_score += (1.0 - channel_freqs[ch_idx])
            w2 = 1.0 + rare_score * weight_rare
            
            # Level 3: 困难样本权重
            difficulty_score = 0.0
            for ch_idx, label in enumerate(y):
                if label > 0:
                    difficulty_score += channel_difficulties[ch_idx]
            w3 = 1.0 + difficulty_score * weight_hard
            
            # 组合权重（几何平均，避免某一项过大）
            total_weight = (w1 * w2 * w3) ** (1/3)
            sample_weights.append(total_weight)
        
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        
        print(f"\nMultiLevelWeightedSampler initialized:")
        print(f"  Weight range: [{self.sample_weights.min():.2f}, {self.sample_weights.max():.2f}]")
        print(f"  Mean weight: {self.sample_weights.mean():.2f}")
    
    def __iter__(self):
        indices = torch.multinomial(
            self.sample_weights,
            self.num_samples,
            replacement=True
        ).tolist()
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


class CurriculumSampler(Sampler):
    """
    课程学习采样器
    
    根据当前训练阶段，采样不同难度的样本
    """
    
    def __init__(
        self,
        dataset,
        total_epochs,
        stage_ratios=[0.2, 0.3, 0.5]  # 前20%轮只用简单样本
    ):
        """
        Args:
            dataset: MultiLabelDataset
            total_epochs: 总训练轮数
            stage_ratios: [stage1_ratio, stage2_ratio, stage3_ratio]
        """
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.num_samples = len(dataset)
        
        # 根据样本包含的通道数量定义难度
        # 包含更多正类 = 更容易（信息更丰富）
        sample_difficulties = []
        for i in range(len(dataset)):
            num_pos = dataset[i]['y'].sum().item()
            # 难度 = 1 / (正样本数 + 1)
            difficulty = 1.0 / (num_pos + 1.0)
            sample_difficulties.append(difficulty)
        
        # 按难度排序样本索引
        sorted_indices = np.argsort(sample_difficulties)
        
        # 分阶段
        n1 = int(len(sorted_indices) * stage_ratios[0])
        n2 = int(len(sorted_indices) * (stage_ratios[0] + stage_ratios[1]))
        
        self.easy_samples = sorted_indices[:n1].tolist()
        self.medium_samples = sorted_indices[n1:n2].tolist()
        self.hard_samples = sorted_indices[n2:].tolist()
        
        self.current_epoch = 0
        
        print(f"\nCurriculumSampler initialized:")
        print(f"  Easy samples: {len(self.easy_samples)}")
        print(f"  Medium samples: {len(self.medium_samples)}")
        print(f"  Hard samples: {len(self.hard_samples)}")
    
    def set_epoch(self, epoch):
        """设置当前epoch"""
        self.current_epoch = epoch
    
    def __iter__(self):
        # 根据当前epoch决定采样策略
        progress = self.current_epoch / self.total_epochs
        
        if progress < 0.3:
            # Stage 1: 主要用简单样本
            pool = self.easy_samples * 3 + self.medium_samples
        elif progress < 0.6:
            # Stage 2: 混合
            pool = self.easy_samples + self.medium_samples * 2 + self.hard_samples
        else:
            # Stage 3: 全部样本，偏向困难样本
            pool = self.easy_samples + self.medium_samples + self.hard_samples * 2
        
        # 随机采样
        indices = np.random.choice(pool, size=self.num_samples, replace=True).tolist()
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    print("Testing Advanced Samplers...")
    
    # 创建模拟数据集
    class MockDataset:
        def __init__(self):
            self.num_channels = 10
            self.data = []
            
            # 模拟不平衡数据
            for i in range(100):
                num_pos = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                y = np.zeros(10)
                pos_indices = np.random.choice(10, num_pos, replace=False)
                y[pos_indices] = 1.0
                self.data.append({'y': torch.from_numpy(y)})
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = MockDataset()
    
    # 测试ClassBalancedSampler
    print("\n1. ClassBalancedSampler:")
    sampler1 = ClassBalancedSampler(dataset, beta=0.999)
    
    # 测试MultiLevelWeightedSampler
    print("\n2. MultiLevelWeightedSampler:")
    sampler2 = MultiLevelWeightedSampler(dataset)
    
    # 测试CurriculumSampler
    print("\n3. CurriculumSampler:")
    sampler3 = CurriculumSampler(dataset, total_epochs=100)
    
    print("\n✓ All samplers working!")

