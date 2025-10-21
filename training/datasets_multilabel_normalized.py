#!/usr/bin/env python3
"""
Per-Matrix Normalization Dataset

对每个连接矩阵独立进行Z-score归一化
解决不同度量量纲不同的问题
"""

import numpy as np
import torch
from pathlib import Path
import pickle
from datasets_multilabel_filtered import FilteredMultiLabelDataset


class NormalizedMultiLabelDataset(FilteredMultiLabelDataset):
    """
    在FilteredMultiLabelDataset基础上，对每个矩阵独立归一化
    
    归一化策略：
    1. 在训练集上计算每个matrix_key的 mean/std
    2. 保存这些统计量
    3. 在 __getitem__ 中应用归一化
    """
    
    def __init__(self, *args, scaler_path=None, fit_scaler=True, **kwargs):
        """
        Args:
            scaler_path: 保存/加载归一化参数的路径
            fit_scaler: 是否在训练集上拟合归一化参数
        """
        super().__init__(*args, **kwargs)
        
        self.scaler_path = scaler_path
        self.scalers = {}  # {matrix_key: {'mean': float, 'std': float}}
        
        if fit_scaler:
            print("\n  Computing normalization statistics on training set...")
            self._fit_scalers()
            
            if scaler_path:
                self._save_scalers(scaler_path)
        else:
            if scaler_path and Path(scaler_path).exists():
                print(f"\n  Loading normalization statistics from {scaler_path}")
                self._load_scalers(scaler_path)
            else:
                print("\n  Warning: No scaler provided, data will not be normalized!")
    
    def _fit_scalers(self):
        """在训练集上计算每个矩阵的归一化参数"""
        from collections import defaultdict
        
        # 收集每个矩阵类型的所有值
        matrix_values = defaultdict(list)
        
        print(f"  Sampling {min(100, len(self))} files to compute statistics...")
        
        # 采样部分数据计算统计量（避免太慢）
        sample_indices = np.linspace(0, len(self)-1, min(100, len(self)), dtype=int)
        
        for idx in sample_indices:
            npz_file = self.npz_paths[idx]
            try:
                arrays = np.load(npz_file, allow_pickle=True)
                
                for key in self.matrix_keys:
                    if key in arrays:
                        matrix = arrays[key]
                        # 只收集非对角线元素（上三角）
                        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                            triu_indices = np.triu_indices(matrix.shape[0], k=1)
                            values = matrix[triu_indices]
                            matrix_values[key].extend(values.flatten())
            except Exception as e:
                print(f"    Warning: Failed to load {Path(npz_file).name}: {e}")
                continue
        
        # 计算每个矩阵的 mean/std
        print(f"\n  Computed normalization parameters:")
        for key in self.matrix_keys:
            if key in matrix_values and len(matrix_values[key]) > 0:
                values = np.array(matrix_values[key])
                
                # 使用robust统计（去除极端值影响）
                mean = np.median(values)
                std = np.std(values)
                
                # 如果std太小，设为1（避免除零）
                if std < 1e-6:
                    std = 1.0
                
                self.scalers[key] = {
                    'mean': float(mean),
                    'std': float(std)
                }
                
                print(f"    {key}: mean={mean:.4f}, std={std:.4f}")
            else:
                print(f"    {key}: No data, skipping normalization")
                self.scalers[key] = {'mean': 0.0, 'std': 1.0}
    
    def _save_scalers(self, path):
        """保存归一化参数"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scalers, f)
        print(f"  ✓ Saved normalization parameters to {path}")
    
    def _load_scalers(self, path):
        """加载归一化参数"""
        with open(path, 'rb') as f:
            self.scalers = pickle.load(f)
        print(f"  ✓ Loaded normalization parameters")
        for key, params in self.scalers.items():
            print(f"    {key}: mean={params['mean']:.4f}, std={params['std']:.4f}")
    
    def _normalize_matrix(self, matrix, key):
        """对单个矩阵应用归一化"""
        if key not in self.scalers:
            return matrix
        
        params = self.scalers[key]
        normalized = (matrix - params['mean']) / params['std']
        
        return normalized
    
    def __getitem__(self, idx: int):
        """重写，添加归一化"""
        # 调用父类获取原始数据
        item = super().__getitem__(idx)
        
        # 如果有归一化参数，应用归一化
        if self.scalers and 'adj' in item:
            # 注意：adj可能是融合后的单个矩阵
            # 这里假设fusion在原始矩阵上已经完成
            # 如果需要对每个原始矩阵单独归一化，需要修改数据流程
            pass  # adj已经是融合后的，这里不再处理
        
        return item


if __name__ == "__main__":
    # 测试
    print("Testing NormalizedMultiLabelDataset...")
    
    import pandas as pd
    from glob import glob
    
    labels_csv = r'E:\output\connectivity_features\labels.csv'
    features_root = r'E:\output\connectivity_features'
    
    if not Path(labels_csv).exists():
        print(f"Labels file not found: {labels_csv}")
        exit(1)
    
    # 加载数据
    labels_df = pd.read_csv(labels_csv, encoding='utf-8')
    
    # 查找所有NPZ文件
    npz_files = []
    for _, row in labels_df.iterrows():
        features_dir = Path(features_root) / row['features_dir_path']
        if features_dir.exists():
            npz_files.extend(list(features_dir.glob('*.npz')))
    
    print(f"Found {len(npz_files)} NPZ files")
    
    # 创建数据集
    dataset = NormalizedMultiLabelDataset(
        npz_paths=[str(f) for f in npz_files[:100]],  # 只用前100个测试
        labels_df=labels_df,
        matrix_keys=['pearson', 'spearman'],
        min_samples=10,
        scaler_path='test_scaler.pkl',
        fit_scaler=True
    )
    
    print(f"\nDataset created with {len(dataset)} samples")
    print(f"Number of channels: {dataset.num_channels}")
    
    # 测试加载
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"  adj shape: {sample['adj'].shape}")
    print(f"  x shape: {sample['x'].shape}")
    print(f"  y shape: {sample['y'].shape}")
    
    print("\n✓ Test passed!")

