#!/usr/bin/env python3
"""
图数据增强方法

包含:
1. Graph SMOTE - 合成少数类样本
2. Graph Mixup - 图混合增强
3. Edge/Node Dropout - 结构增强
4. Feature Noise - 特征增强
"""

import torch
import numpy as np
from typing import Dict, List


class GraphSMOTE:
    """
    图数据的SMOTE（Synthetic Minority Over-sampling Technique）
    
    为稀有通道生成合成样本
    """
    
    def __init__(self, dataset, channel_idx, k_neighbors=5):
        """
        Args:
            dataset: MultiLabelDataset
            channel_idx: 目标通道索引
            k_neighbors: KNN的k值
        """
        self.dataset = dataset
        self.channel_idx = channel_idx
        self.k_neighbors = k_neighbors
        
        # 找出该通道的所有正样本
        self.positive_indices = []
        for i in range(len(dataset)):
            if dataset[i]['y'][channel_idx] > 0:
                self.positive_indices.append(i)
        
        print(f"GraphSMOTE: Found {len(self.positive_indices)} positive samples for channel {channel_idx}")
    
    def generate(self, n_synthetic):
        """
        生成n_synthetic个合成样本
        
        Returns:
            List of synthetic samples
        """
        if len(self.positive_indices) < 2:
            print(f"Warning: Not enough positive samples ({len(self.positive_indices)})")
            return []
        
        synthetic_samples = []
        
        for _ in range(n_synthetic):
            # 随机选择一个正样本
            idx1 = np.random.choice(self.positive_indices)
            
            # 找到k个最近邻（简化版：随机选择）
            neighbors = [i for i in self.positive_indices if i != idx1]
            if not neighbors:
                continue
            
            idx2 = np.random.choice(neighbors)
            
            # 获取样本
            sample1 = self.dataset[idx1]
            sample2 = self.dataset[idx2]
            
            # 插值系数
            alpha = np.random.uniform(0.2, 0.8)
            
            # 合成新样本
            new_sample = self._interpolate(sample1, sample2, alpha)
            synthetic_samples.append(new_sample)
        
        return synthetic_samples
    
    def _interpolate(self, sample1, sample2, alpha):
        """插值生成新样本"""
        new_adj = alpha * sample1['adj'] + (1 - alpha) * sample2['adj']
        new_x = alpha * sample1['x'] + (1 - alpha) * sample2['x']
        
        # 标签：取并集（conservative）或交集（aggressive）
        # 这里使用并集
        new_y = torch.maximum(sample1['y'], sample2['y'])
        
        return {
            'adj': new_adj,
            'x': new_x,
            'y': new_y,
            'n': sample1['n'],
            'path': f"synthetic_{sample1['path']}_{sample2['path']}",
            'combo': f"{sample1.get('combo', '')}+{sample2.get('combo', '')}"
        }


class GraphMixup:
    """
    图数据的Mixup增强
    
    混合两个图样本，生成新的训练样本
    """
    
    def __init__(self, alpha=0.2):
        """
        Args:
            alpha: Beta分布参数，alpha越大混合越均匀
        """
        self.alpha = alpha
    
    def __call__(self, batch1, batch2):
        """
        混合两个batch
        
        Args:
            batch1, batch2: dict with 'adj', 'x', 'y'
        
        Returns:
            mixed batch
        """
        # 采样lambda
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 混合邻接矩阵
        mixed_adj = lam * batch1['adj'] + (1 - lam) * batch2['adj']
        
        # 混合节点特征
        mixed_x = lam * batch1['x'] + (1 - lam) * batch2['x']
        
        # 混合标签（软标签）
        mixed_y = lam * batch1['y'] + (1 - lam) * batch2['y']
        
        result = {
            'adj': mixed_adj,
            'x': mixed_x,
            'y': mixed_y
        }
        
        # 如果batch1有'n'键，也包含它（向后兼容）
        if 'n' in batch1:
            result['n'] = batch1['n']
        
        return result


class GraphAugmentor:
    """
    图增强器 - 集成多种增强方法
    """
    
    def __init__(
        self,
        edge_dropout_rate=0.2,
        node_dropout_rate=0.1,
        feature_noise_std=0.05,
        use_mixup=True,
        mixup_alpha=0.2
    ):
        self.edge_dropout_rate = edge_dropout_rate
        self.node_dropout_rate = node_dropout_rate
        self.feature_noise_std = feature_noise_std
        self.use_mixup = use_mixup
        self.mixup = GraphMixup(alpha=mixup_alpha) if use_mixup else None
    
    def edge_dropout(self, adj):
        """边dropout"""
        if self.edge_dropout_rate > 0:
            mask = torch.rand_like(adj) > self.edge_dropout_rate
            adj = adj * mask.float()
            # 保持对称性
            adj = (adj + adj.transpose(-1, -2)) / 2.0
        return adj
    
    def node_dropout(self, x):
        """节点dropout"""
        if self.node_dropout_rate > 0:
            mask = torch.rand(x.size(0), x.size(1), 1) > self.node_dropout_rate
            mask = mask.to(x.device)
            x = x * mask.float()
        return x
    
    def feature_noise(self, x):
        """特征噪声"""
        if self.feature_noise_std > 0:
            noise = torch.randn_like(x) * self.feature_noise_std
            x = x + noise
        return x
    
    def augment(self, adj, x, y=None):
        """
        应用增强
        
        Args:
            adj: [B, N, N] 或 [N, N]
            x: [B, N, F] 或 [N, F]
            y: [B, num_channels] 标签（可选）
        
        Returns:
            augmented adj, x, y
        """
        # Edge dropout
        adj = self.edge_dropout(adj)
        
        # Node dropout
        x = self.node_dropout(x)
        
        # Feature noise
        x = self.feature_noise(x)
        
        return adj, x, y
    
    def augment_batch(self, batch, mixup_batch=None):
        """
        增强一个batch，可选mixup
        
        Args:
            batch: 当前batch
            mixup_batch: 用于mixup的另一个batch（可选）
        """
        adj, x, y = self.augment(batch['adj'], batch['x'], batch['y'])
        
        # Mixup
        if mixup_batch is not None and self.use_mixup and np.random.rand() < 0.5:
            # 检查batch size是否一致
            if batch['adj'].shape[0] == mixup_batch['adj'].shape[0]:
                # 构造mixup输入（不需要'n'键）
                current_batch = {'adj': adj, 'x': x, 'y': y}
                mixup_result = self.mixup(current_batch, mixup_batch)
                adj, x, y = mixup_result['adj'], mixup_result['x'], mixup_result['y']
            # else: batch size不一致，跳过mixup
        
        return adj, x, y


def create_augmented_dataset(
    base_dataset,
    rare_channel_threshold=15,
    n_synthetic_per_channel=50
):
    """
    为稀有通道创建增强数据集
    
    Args:
        base_dataset: 原始数据集
        rare_channel_threshold: 少于此值的通道被认为是稀有的
        n_synthetic_per_channel: 每个稀有通道生成多少合成样本
    
    Returns:
        augmented_dataset: 包含原始+合成样本的数据集
    """
    from torch.utils.data import ConcatDataset
    
    # 统计每个通道的样本数
    channel_counts = np.zeros(base_dataset.num_channels)
    for i in range(len(base_dataset)):
        y = base_dataset[i]['y'].numpy()
        channel_counts += y
    
    # 找出稀有通道
    rare_channels = [i for i, count in enumerate(channel_counts) 
                     if 0 < count < rare_channel_threshold]
    
    print(f"\nCreating augmented dataset:")
    print(f"  Rare channels ({len(rare_channels)}): {rare_channels}")
    
    # 为每个稀有通道生成合成样本
    all_synthetic = []
    for ch_idx in rare_channels:
        smote = GraphSMOTE(base_dataset, ch_idx, k_neighbors=5)
        synthetic = smote.generate(n_synthetic_per_channel)
        all_synthetic.extend(synthetic)
        print(f"    Channel {ch_idx}: generated {len(synthetic)} samples")
    
    print(f"  Total synthetic samples: {len(all_synthetic)}")
    
    # 创建合成数据集
    if all_synthetic:
        from torch.utils.data import Dataset
        
        class SyntheticDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        synthetic_dataset = SyntheticDataset(all_synthetic)
        augmented_dataset = ConcatDataset([base_dataset, synthetic_dataset])
        
        print(f"  Final dataset size: {len(augmented_dataset)} (original: {len(base_dataset)})")
        return augmented_dataset
    else:
        print(f"  No synthetic samples generated")
        return base_dataset


if __name__ == "__main__":
    print("Testing Graph Augmentations...")
    
    # 测试GraphAugmentor
    print("\n1. GraphAugmentor:")
    augmentor = GraphAugmentor(
        edge_dropout_rate=0.2,
        node_dropout_rate=0.1,
        feature_noise_std=0.05,
        use_mixup=True
    )
    
    # 创建测试数据
    batch_size, n_nodes, n_features = 4, 20, 2
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    x = torch.randn(batch_size, n_nodes, n_features)
    y = torch.randint(0, 2, (batch_size, 10)).float()
    
    # 增强
    aug_adj, aug_x, aug_y = augmentor.augment(adj, x, y)
    
    print(f"  Original adj: {adj.shape}")
    print(f"  Augmented adj: {aug_adj.shape}")
    print(f"  ✓ Basic augmentation works")
    
    # 测试Mixup
    print("\n2. Graph Mixup:")
    mixup = GraphMixup(alpha=0.2)
    batch1 = {'adj': adj[:2], 'x': x[:2], 'y': y[:2], 'n': torch.tensor([20, 20])}
    batch2 = {'adj': adj[2:], 'x': x[2:], 'y': y[2:], 'n': torch.tensor([20, 20])}
    
    mixed = mixup(batch1, batch2)
    print(f"  Mixed adj: {mixed['adj'].shape}")
    print(f"  Mixed y: {mixed['y'].shape}")
    print(f"  ✓ Mixup works")
    
    print("\n✓ All augmentations working!")

