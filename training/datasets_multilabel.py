#!/usr/bin/env python3
"""
多标签数据集 - 用于通道级别的异常检测
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Union
from pathlib import Path


# 标准10-20系统通道
STANDARD_CHANNELS = [
    'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]


def _safe_load_npz(npz_file: str) -> Dict[str, np.ndarray]:
    """安全加载NPZ文件"""
    arrays = np.load(npz_file, allow_pickle=True)
    return {k: arrays[k] for k in arrays.files}


def build_graph_from_matrix(matrix: np.ndarray, topk_ratio: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从连接矩阵构建图，考虑个体差异进行归一化
    
    Args:
        matrix: 连接矩阵 [N, N]
        topk_ratio: 保留的边的比例
    
    Returns:
        adj: 邻接矩阵 [N, N]
        node_feat: 节点特征 [N, 2] (归一化的度和强度)
    """
    n = matrix.shape[0]
    mat = matrix.copy()
    
    # 1. 个体归一化：标准化到[0, 1]（考虑个体差异）
    # 使用min-max归一化，保留连接的相对强度
    mat_abs = np.abs(mat)
    mat_min = mat_abs.min()
    mat_max = mat_abs.max()
    
    if mat_max > mat_min:
        mat = (mat_abs - mat_min) / (mat_max - mat_min)
    else:
        mat = mat_abs
    
    np.fill_diagonal(mat, 0.0)
    
    # 2. Top-k稀疏化（基于归一化后的值）
    flat = mat.flatten()
    if len(flat) > 0:
        kth = np.percentile(flat, (1.0 - topk_ratio) * 100.0)
        mat[mat < kth] = 0.0
    
    # 3. 节点特征：度和强度（标准化）
    degree = (mat > 0).sum(axis=1, dtype=np.float32)
    strength = mat.sum(axis=1, dtype=np.float32)
    
    # 标准化度（相对于最大可能度）
    max_degree = n - 1
    if max_degree > 0:
        degree = degree / max_degree
    
    # 标准化强度（Z-score归一化）
    strength_mean = strength.mean()
    strength_std = strength.std()
    if strength_std > 1e-6:
        strength = (strength - strength_mean) / strength_std
    else:
        strength = strength - strength_mean
    
    node_feat = np.stack([degree, strength], axis=1)  # [N, 2]
    
    return torch.from_numpy(mat).float(), torch.from_numpy(node_feat).float()


def discover_all_channels(labels_df: pd.DataFrame) -> List[str]:
    """
    从labels.csv中发现所有使用的通道，清理无关字符
    
    Args:
        labels_df: 包含channel_combination列的DataFrame
    
    Returns:
        sorted list of unique channels
    """
    all_channels = set()
    
    for combo_str in labels_df['channel_combination'].unique():
        if not isinstance(combo_str, str):
            continue
        
        # 1. 移除方括号、引号等无关字符
        cleaned = combo_str.strip()
        cleaned = cleaned.replace('[', '').replace(']', '')
        cleaned = cleaned.replace("'", '').replace('"', '')
        cleaned = cleaned.replace('(', '').replace(')', '')
        
        # 2. 尝试不同的分隔符
        channels = []
        for sep in ['-', ',', ';', ' ', '|']:
            if sep in cleaned:
                channels = [ch.strip() for ch in cleaned.split(sep) if ch.strip()]
                break
        else:
            # 单个通道
            if cleaned.strip():
                channels = [cleaned.strip()]
        
        # 3. 清理每个通道名称
        for ch in channels:
            ch = ch.strip()
            
            # 标准化通道名中的下划线（如 Sph_L, Sph_R）
            # 但移除其他分隔符（避免 F7-T3 被当成一个通道）
            # 只保留：字母、数字、下划线
            cleaned_ch = ''.join(c if c.isalnum() or c == '_' else '' for c in ch)
            
            # 验证通道名格式：
            # 1. 必须包含字母
            # 2. 长度在 2-10 之间（避免太长的组合被当成通道）
            # 3. 不能有多个下划线连续（如 Sph_L 可以，F7__T3 不行）
            if (cleaned_ch and 
                any(c.isalpha() for c in cleaned_ch) and 
                2 <= len(cleaned_ch) <= 10 and
                '__' not in cleaned_ch):
                all_channels.add(cleaned_ch)
    
    # 移除空字符串和无效值
    all_channels.discard('')
    
    # 过滤明显无效的通道名
    invalid_patterns = ['[]', '()', '{}', 'nan', 'none', 'null']
    all_channels = {ch for ch in all_channels 
                    if ch.lower() not in invalid_patterns and len(ch) >= 2}
    
    return sorted(all_channels)


class MultiLabelConnectivityDataset(Dataset):
    """
    多标签连接性数据集
    
    将通道组合转换为多标签向量
    例如: "Fp1-F3-C3" → [1,0,1,1,0,0,...]
    """
    
    def __init__(
        self,
        npz_paths: List[str],
        labels_df: pd.DataFrame,
        all_channels: List[str] = None,
        matrix_keys: Union[str, List[str]] = "plv_alpha",
        fusion_method: str = "weighted",
        topk_ratio: float = 0.2,
        augment: bool = False,
    ):
        """
        Args:
            npz_paths: NPZ文件路径列表
            labels_df: 标签DataFrame（包含channel_combination列）
            all_channels: 所有可能的通道列表（None则自动发现）
            matrix_keys: 使用的矩阵键
            fusion_method: 矩阵融合方法
            topk_ratio: Top-k稀疏化比例
            augment: 是否进行数据增强（暂未实现）
        """
        self.npz_paths = npz_paths
        self.labels_df = labels_df
        self.matrix_keys = matrix_keys if isinstance(matrix_keys, list) else [matrix_keys]
        self.fusion_method = fusion_method
        self.topk_ratio = topk_ratio
        self.augment = augment
        
        # 通道映射
        if all_channels is None:
            # 自动从数据中发现
            all_channels = discover_all_channels(labels_df)
            print(f"Auto-discovered {len(all_channels)} channels from labels")
        
        self.all_channels = sorted(all_channels)
        self.channel_to_idx = {ch: idx for idx, ch in enumerate(self.all_channels)}
        self.num_channels = len(self.all_channels)
        
        print(f"MultiLabel Dataset: {self.num_channels} channels as multi-label targets")
        print(f"Channels: {', '.join(self.all_channels)}")
        
        # 统计标签分布
        self._analyze_label_distribution()
    
    def _analyze_label_distribution(self):
        """分析标签分布"""
        channel_counts = {ch: 0 for ch in self.all_channels}
        
        for combo_str in self.labels_df['channel_combination']:
            channels = self._parse_channels(combo_str)
            for ch in channels:
                if ch in channel_counts:
                    channel_counts[ch] += 1
        
        print(f"\nChannel occurrence statistics:")
        sorted_counts = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)
        for ch, count in sorted_counts[:10]:  # Top 10
            print(f"  {ch}: {count} times")
        
        # 计算正样本权重（用于处理不平衡）
        total_samples = len(self.labels_df)
        pos_counts = np.array([channel_counts[ch] for ch in self.all_channels])
        neg_counts = total_samples - pos_counts
        self.pos_weight = torch.from_numpy(neg_counts / (pos_counts + 1e-6)).float()
    
    def get_pos_weight(self) -> torch.Tensor:
        """获取正样本权重（用于BCEWithLogitsLoss）"""
        return self.pos_weight
    
    def __len__(self) -> int:
        return len(self.npz_paths)
    
    def _parse_channels(self, combo_str: str) -> List[str]:
        """
        解析通道组合字符串，清理所有无关字符
        
        处理各种格式：
        - "Fp1-F3-C3" → ['Fp1', 'F3', 'C3']
        - "Fp1, F3, C3" → ['Fp1', 'F3', 'C3']
        - "['Fp1', 'F3']" → ['Fp1', 'F3']
        - "[]" → []
        """
        if not isinstance(combo_str, str):
            return []
        
        # 1. 移除方括号、引号等无关字符
        cleaned = combo_str.strip()
        cleaned = cleaned.replace('[', '').replace(']', '')
        cleaned = cleaned.replace("'", '').replace('"', '')
        cleaned = cleaned.replace('(', '').replace(')', '')
        
        # 2. 尝试不同的分隔符
        channels = []
        for sep in ['-', ',', ';', ' ', '|']:
            if sep in cleaned:
                channels = [ch.strip() for ch in cleaned.split(sep) if ch.strip()]
                break
        else:
            # 没有分隔符，可能是单个通道
            channels = [cleaned.strip()] if cleaned.strip() else []
        
        # 3. 进一步清理每个通道名称
        cleaned_channels = []
        for ch in channels:
            ch = ch.strip()
            
            # 只保留字母、数字、下划线
            # 移除分隔符（避免 F7-T3 被当成单个通道）
            cleaned_ch = ''.join(c if c.isalnum() or c == '_' else '' for c in ch)
            
            # 验证通道名：
            # 1. 必须包含字母
            # 2. 长度合理（2-10字符，避免组合被误识别）
            # 3. 不能有连续下划线
            if (cleaned_ch and 
                any(c.isalpha() for c in cleaned_ch) and 
                2 <= len(cleaned_ch) <= 10 and
                '__' not in cleaned_ch):
                cleaned_channels.append(cleaned_ch)
        
        return cleaned_channels
    
    def _parse_channel_combination(self, combo_str: str) -> torch.Tensor:
        """
        将通道组合字符串转换为多标签向量
        
        Args:
            combo_str: "Fp1-F3-C3" 或 "Fp1, F3, C3"
        
        Returns:
            label_vector: [num_channels] 的二值向量
        """
        label_vector = np.zeros(self.num_channels, dtype=np.float32)
        
        channels = self._parse_channels(combo_str)
        
        # 标记存在的通道
        for ch in channels:
            if ch in self.channel_to_idx:
                label_vector[self.channel_to_idx[ch]] = 1.0
            elif ch:  # 非空但不在列表中
                print(f"Warning: Unknown channel '{ch}' in '{combo_str}'")
        
        return torch.from_numpy(label_vector)
    
    def _get_channel_combination(self, npz_file: str) -> str:
        """
        从CSV获取通道组合
        
        Args:
            npz_file: NPZ文件的完整路径
        
        Returns:
            channel_combination字符串，如果找不到则返回空字符串
        """
        # 标准化路径分隔符
        npz_file_normalized = npz_file.replace('\\', '/')
        
        for _, row in self.labels_df.iterrows():
            features_dir_path = str(row['features_dir_path']).replace('\\', '/')
            
            # 尝试多种匹配方式
            if (features_dir_path in npz_file_normalized or 
                npz_file_normalized.endswith(features_dir_path) or
                features_dir_path in Path(npz_file).parts):
                
                return row['channel_combination']
        
        # 如果没找到，打印警告（只在前几次）
        if not hasattr(self, '_warned_files'):
            self._warned_files = set()
        
        if npz_file not in self._warned_files and len(self._warned_files) < 5:
            print(f"Warning: No label found for {Path(npz_file).name}")
            self._warned_files.add(npz_file)
        
        return ""
    
    def __getitem__(self, idx: int):
        npz_file = self.npz_paths[idx]
        arrays = _safe_load_npz(npz_file)
        
        # 加载矩阵
        matrices = []
        available_keys = []
        for key in self.matrix_keys:
            if key in arrays:
                matrices.append(arrays[key])
                available_keys.append(key)
        
        # 回退到默认键
        if not matrices:
            fallback_keys = arrays.keys()
            for key in fallback_keys:
                if key in arrays:
                    matrices.append(arrays[key])
                    available_keys.append(key)
                    break
        
        if not matrices:
            raise ValueError(f"No valid matrices found in {npz_file}")
        
        # 融合矩阵（简化版：使用第一个或平均）
        if len(matrices) == 1:
            matrix = matrices[0]
        else:
            # 简单平均
            matrix = np.mean(matrices, axis=0)
        
        adj, node_feat = build_graph_from_matrix(matrix, self.topk_ratio)
        
        # 获取多标签
        channel_combo = self._get_channel_combination(npz_file)
        multi_labels = self._parse_channel_combination(channel_combo)
        
        # 调试：检查是否所有标签都是0
        if not hasattr(self, '_checked_labels'):
            self._checked_labels = 0
        
        if self._checked_labels < 5:  # 只检查前5个样本
            label_sum = multi_labels.sum().item()
            if label_sum == 0 and channel_combo:
                print(f"\n⚠ DEBUG: Sample {idx} has all-zero labels!")
                print(f"  File: {Path(npz_file).name}")
                print(f"  Combo string: '{channel_combo}'")
                parsed = self._parse_channels(channel_combo)
                print(f"  Parsed channels: {parsed}")
                print(f"  Known channels: {self.all_channels[:10]}...")
                
                # 检查哪些通道在已知列表中
                for ch in parsed:
                    if ch in self.channel_to_idx:
                        print(f"    ✓ '{ch}' found at index {self.channel_to_idx[ch]}")
                    else:
                        print(f"    ✗ '{ch}' NOT in channel list!")
            elif label_sum > 0:
                print(f"\n✓ DEBUG: Sample {idx} has {int(label_sum)} positive labels")
                parsed = self._parse_channels(channel_combo)
                print(f"  Parsed channels: {parsed}")
            
            self._checked_labels += 1
        
        return {
            "adj": adj,                    # [N, N] 邻接矩阵
            "x": node_feat,                # [N, 2] 节点特征
            "y": multi_labels,             # [num_channels] 多标签向量
            "n": torch.tensor(adj.shape[0], dtype=torch.long),
            "path": npz_file,
            "combo": channel_combo,        # 原始通道组合字符串
            "matrix_keys": available_keys,
        }


def load_labels_csv(labels_csv_path: str) -> pd.DataFrame:
    """加载labels.csv"""
    return pd.read_csv(labels_csv_path, encoding='utf-8')


def discover_patient_segments_from_csv(labels_csv_path: str, features_root: str) -> Dict[str, List[str]]:
    """
    从CSV发现患者的NPZ文件
    
    Returns:
        Dict[patient_name, List[npz_file_paths]]
    """
    labels_df = load_labels_csv(labels_csv_path)
    patient_to_files = {}
    
    for _, row in labels_df.iterrows():
        features_dir = Path(features_root) / row['features_dir_path']
        if features_dir.exists():
            npz_files = sorted(features_dir.glob('*.npz'))
            
            # 提取患者名（假设在路径中）
            patient_name = row['features_dir_path'].split('/')[0]
            
            if patient_name not in patient_to_files:
                patient_to_files[patient_name] = []
            
            patient_to_files[patient_name].extend([str(f) for f in npz_files])
    
    return patient_to_files


def make_patient_splits(
    patient_to_files: Dict[str, List[str]],
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    按患者划分数据集
    
    注意：确保每个集合至少有1个患者（如果可能）
    """
    import random
    random.seed(seed)
    
    patients = list(patient_to_files.keys())
    random.shuffle(patients)
    
    n = len(patients)
    
    # 确保至少有3个患者才能分割
    if n < 3:
        print(f"WARNING: Only {n} patients, cannot create proper train/val/test split")
        print(f"  Using all data for training, no validation/test sets")
        all_files = []
        for p in patients:
            all_files.extend(patient_to_files[p])
        return {
            'train': all_files,
            'val': [],
            'test': []
        }
    
    # 计算分割数量，确保至少1个患者在测试集
    n_test = max(1, int(n * test_ratio))
    n_val = max(0, int(n * val_ratio)) if n > 3 else 0  # 患者太少时不用验证集
    
    # 确保分割合理
    if n_test + n_val >= n:
        # 调整比例
        n_test = max(1, n // 5)  # 至少20%给测试
        n_val = max(0, (n - n_test) // 10) if n > 5 else 0
        print(f"Adjusted split: n_test={n_test}, n_val={n_val} (due to small sample size)")
    
    test_patients = patients[:n_test]
    val_patients = patients[n_test:n_test + n_val]
    train_patients = patients[n_test + n_val:]
    
    # 确保训练集不为空
    if not train_patients:
        print("WARNING: No patients in training set! Adjusting...")
        train_patients = patients[n_test:]
        val_patients = []
    
    train_files = []
    val_files = []
    test_files = []
    
    for p in train_patients:
        train_files.extend(patient_to_files[p])
    for p in val_patients:
        val_files.extend(patient_to_files[p])
    for p in test_patients:
        test_files.extend(patient_to_files[p])
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_patients)} patients, {len(train_files)} files")
    print(f"  Val:   {len(val_patients)} patients, {len(val_files)} files")
    print(f"  Test:  {len(test_patients)} patients, {len(test_files)} files")
    
    if len(val_files) == 0 and val_ratio > 0:
        print(f"  ⚠ Warning: Validation set is empty (val_ratio={val_ratio}, but n_patients={n})")
    
    return {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }


# Collate函数
def collate_graph_multilabel(batch):
    """多标签版本的collate函数"""
    max_n = max(int(item['n']) for item in batch)
    B = len(batch)
    
    adjs = torch.zeros(B, max_n, max_n)
    xs = torch.zeros(B, max_n, batch[0]['x'].shape[-1])
    ys = torch.stack([item['y'] for item in batch])  # [B, num_channels]
    paths = [item['path'] for item in batch]
    combos = [item['combo'] for item in batch]
    
    for i, item in enumerate(batch):
        n = int(item['n'])
        adjs[i, :n, :n] = item['adj']
        xs[i, :n] = item['x']
    
    return {
        "adj": adjs,
        "x": xs,
        "y": ys,
        "paths": paths,
        "combos": combos
    }


if __name__ == "__main__":
    # 测试
    print("Testing MultiLabelConnectivityDataset...")
    
    labels_csv = r'E:\output\connectivity_features\labels.csv'
    features_root = r'E:\output\connectivity_features'
    
    if Path(labels_csv).exists():
        labels_df = load_labels_csv(labels_csv)
        patient_to_files = discover_patient_segments_from_csv(labels_csv, features_root)
        
        # 测试数据集
        all_files = []
        for files in patient_to_files.values():
            all_files.extend(files[:2])  # 每个患者取2个文件测试
        
        dataset = MultiLabelConnectivityDataset(
            all_files[:10],
            labels_df,
            matrix_keys=['plv_alpha']
        )
        
        print(f"\nDataset size: {len(dataset)}")
        print(f"Number of channels: {dataset.num_channels}")
        
        # 测试加载
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  adj shape: {sample['adj'].shape}")
        print(f"  x shape: {sample['x'].shape}")
        print(f"  y shape: {sample['y'].shape}")
        print(f"  combo: {sample['combo']}")
        print(f"  y sum: {sample['y'].sum().item()} (number of abnormal channels)")
    else:
        print(f"Labels file not found: {labels_csv}")

