import os
import glob
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ConnectivitySegment:
    """Container for one 30s segment connectivity features and metadata."""

    def __init__(self, npz_path: str, features: Dict[str, np.ndarray], meta: Dict):
        self.npz_path = npz_path
        self.features = features
        self.meta = meta


def _safe_load_npz(npz_file: str) -> Dict[str, np.ndarray]:
    arrays = np.load(npz_file, allow_pickle=True)
    return {k: arrays[k] for k in arrays.files}


def load_labels_csv(labels_csv_path: str) -> pd.DataFrame:
    """Load the labels CSV file with patient and channel combination information."""
    return pd.read_csv(labels_csv_path, encoding='utf-8')


def discover_patient_segments_from_csv(labels_csv_path: str, features_root: str) -> Dict[str, List[str]]:
    """
    Discover all connectivity npz files grouped by patient name using the labels CSV.
    
    Args:
        labels_csv_path: Path to labels.csv file
        features_root: Root directory for connectivity features
        
    Returns:
        Dict mapping patient names to lists of npz file paths
    """
    df = load_labels_csv(labels_csv_path)
    patient_to_files: Dict[str, List[str]] = {}
    
    for _, row in df.iterrows():
        patient_name = row['patient_name']
        features_dir_path = row['features_dir_path']
        
        # Construct full path to features directory
        features_dir = os.path.join(features_root, features_dir_path)
        
        # Find all npz files in this directory
        npz_pattern = os.path.join(features_dir, "connectivity_matrices_seg*.npz")
        npz_files = glob.glob(npz_pattern)
        
        if npz_files:
            patient_to_files.setdefault(patient_name, []).extend(sorted(npz_files))
    
    return patient_to_files


def discover_patient_segments(root_dir: str) -> Dict[str, List[str]]:
    """
    Legacy function for backward compatibility.
    Discover all connectivity npz files grouped by patient name.

    Assumes outputs are saved by `extract_connectivity_features.py`, one directory
    per original .set with many `connectivity_matrices_segXXX.npz` files. Patient
    id is inferred from the .set parent path two levels up or directory name.
    """
    npz_files = glob.glob(os.path.join(root_dir, "**", "connectivity_matrices_seg*.npz"), recursive=True)
    patient_to_files: Dict[str, List[str]] = {}
    for f in npz_files:
        # infer patient from upstream directory structure, fallback to parent name
        parts = os.path.normpath(f).split(os.sep)
        patient = parts[-5] if len(parts) >= 5 else parts[-3]
        patient_to_files.setdefault(patient, []).append(f)
    # stable sort
    for k in patient_to_files:
        patient_to_files[k] = sorted(patient_to_files[k])
    return patient_to_files


def build_graph_from_matrix(matrix: np.ndarray, topk_ratio: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build dense adjacency and node features from a connectivity matrix.
    Uses individual-level normalization to handle inter-subject variability.
    
    Args:
        matrix: connectivity matrix [N, N]
        topk_ratio: ratio of edges to keep
    
    Returns:
        adjacency: normalized adjacency matrix [N, N]
        node_features: normalized degree/strength features [N, 2]
    """
    n = matrix.shape[0]
    mat = matrix.copy()
    
    # Individual-level min-max normalization (handles inter-subject differences)
    mat_abs = np.abs(mat)
    mat_min = mat_abs.min()
    mat_max = mat_abs.max()
    
    if mat_max > mat_min:
        mat = (mat_abs - mat_min) / (mat_max - mat_min)
    else:
        mat = mat_abs
    
    np.fill_diagonal(mat, 0.0)
    
    # Sparsify by top-k ratio
    flat = mat.flatten()
    if len(flat) > 0:
        kth = np.percentile(flat, (1.0 - topk_ratio) * 100.0)
        mat[mat < kth] = 0.0
    
    # Node features: normalized degree and strength
    degree = (mat > 0).sum(axis=1, dtype=np.float32)
    strength = mat.sum(axis=1, dtype=np.float32)
    
    # Normalize degree (relative to max possible degree)
    max_degree = n - 1
    if max_degree > 0:
        degree = degree / max_degree
    
    # Normalize strength (Z-score)
    strength_mean = strength.mean()
    strength_std = strength.std()
    if strength_std > 1e-6:
        strength = (strength - strength_mean) / strength_std
    else:
        strength = strength - strength_mean
    
    node_feat = np.stack([degree, strength], axis=1)  # [N, 2]
    return torch.from_numpy(mat).float(), torch.from_numpy(node_feat).float()


class ConnectivityGraphDataset(Dataset):
    """
    Dataset reading connectivity matrices npz per segment and a label per segment.
    Label is derived from the labels CSV that maps channel combinations to target classes.
    Supports multiple matrix keys with attention-based fusion.
    """

    def __init__(
        self,
        npz_paths: List[str],
        labels_df: pd.DataFrame,
        matrix_keys: List[str] = ["plv_alpha"],
        augment: bool = False,
        topk_ratio: float = 0.2,
        fusion_method: str = "attention",
    ):
        self.npz_paths = npz_paths
        self.labels_df = labels_df
        self.matrix_keys = matrix_keys if isinstance(matrix_keys, list) else [matrix_keys]
        self.augment = augment
        self.topk_ratio = topk_ratio
        self.fusion_method = fusion_method
        
        # Build label mapping from unique channel combinations
        self._build_label_mapping()

    def _build_label_mapping(self):
        """Build a mapping from channel combinations to integer labels."""
        unique_combos = self.labels_df['channel_combination'].unique()
        self.label_to_int = {combo: idx for idx, combo in enumerate(sorted(unique_combos))}
        self.int_to_label = {idx: combo for combo, idx in self.label_to_int.items()}
        self.num_classes = len(self.label_to_int)
        print(f"Dataset: Found {self.num_classes} unique channel combinations (classes)")
    
    def get_num_classes(self) -> int:
        """Get the number of unique classes in this dataset."""
        return self.num_classes

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int):
        npz_file = self.npz_paths[idx]
        arrays = _safe_load_npz(npz_file)
        
        # Load multiple matrices
        matrices = []
        available_keys = []
        for key in self.matrix_keys:
            if key in arrays:
                matrices.append(arrays[key])
                available_keys.append(key)
        
        # Fallback to default keys if none found
        if not matrices:
            fallback_keys = ["pearson_corr", "coherence_alpha", "wpli_alpha", "plv_alpha"]
            for key in fallback_keys:
                if key in arrays:
                    matrices.append(arrays[key])
                    available_keys.append(key)
                    break
        
        if not matrices:
            raise ValueError(f"No valid matrices found in {npz_file}")
        
        # Fuse matrices based on method
        if self.fusion_method == "attention" and len(matrices) > 1:
            # For attention fusion, return multiple matrices as dictionary
            # (fusion will be handled in the model)
            matrices_dict = {}
            for key, mat in zip(available_keys, matrices):
                adj, feat = build_graph_from_matrix(mat, self.topk_ratio)
                matrices_dict[key] = adj
            
            # Use first matrix for node features
            fused_adj, node_feat = build_graph_from_matrix(matrices[0], self.topk_ratio)
            
            # infer label from CSV based on npz file path
            label = self._resolve_label(npz_file)
            return {
                "adj": fused_adj,  # [N, N] - for backward compatibility
                "matrices": matrices_dict,  # Dict of matrices for attention fusion
                "x": node_feat,  # [N, 2]
                "y": torch.tensor(label, dtype=torch.long),
                "n": torch.tensor(fused_adj.shape[0], dtype=torch.long),
                "path": npz_file,
                "matrix_keys": available_keys,  # For debugging
            }
        elif self.fusion_method == "concat":
            fused_adj, node_feat = self._concat_fusion(matrices)
        elif self.fusion_method == "weighted":
            fused_adj, node_feat = self._weighted_fusion(matrices)
        else:
            # Default: use first matrix
            fused_adj, node_feat = build_graph_from_matrix(matrices[0], self.topk_ratio)
        
        # infer label from CSV based on npz file path
        label = self._resolve_label(npz_file)
        return {
            "adj": fused_adj,  # [N, N]
            "x": node_feat,  # [N, 2]
            "y": torch.tensor(label, dtype=torch.long),
            "n": torch.tensor(fused_adj.shape[0], dtype=torch.long),
            "path": npz_file,
            "matrix_keys": available_keys,  # For debugging
        }

    def _resolve_label(self, npz_file: str) -> int:
        """
        Resolve label from CSV based on npz file path.
        The npz file path should contain the features_dir_path from the CSV.
        """
        # Extract the relative path from the npz file
        # Find matching row in labels_df based on features_dir_path
        for _, row in self.labels_df.iterrows():
            features_dir_path = row['features_dir_path']
            # Check if the npz file is in this features directory
            if features_dir_path in npz_file:
                channel_combo = row['channel_combination']
                # Use the pre-built mapping
                if channel_combo in self.label_to_int:
                    return self.label_to_int[channel_combo]
                else:
                    # Should not happen if mapping was built correctly
                    print(f"Warning: Unknown channel combination: {channel_combo}")
                    return 0
        
        # Fallback: return 0 if no match found
        print(f"Warning: No matching label found for {npz_file}")
        return 0
    
    def _attention_fusion(self, matrices: List[np.ndarray], keys: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attention-based fusion of multiple matrices.
        For now, use simple weighted average with learnable weights.
        """
        if len(matrices) == 1:
            return build_graph_from_matrix(matrices[0], self.topk_ratio)
        
        # Simple weighted average (will be replaced by learned attention in model)
        weights = np.ones(len(matrices)) / len(matrices)
        fused_matrix = np.zeros_like(matrices[0])
        for i, (matrix, weight) in enumerate(zip(matrices, weights)):
            fused_matrix += weight * matrix
        
        return build_graph_from_matrix(fused_matrix, self.topk_ratio)
    
    def _concat_fusion(self, matrices: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate multiple matrices along feature dimension.
        For graph adjacency, we'll use weighted average instead of concatenation.
        """
        if len(matrices) == 1:
            return build_graph_from_matrix(matrices[0], self.topk_ratio)
        
        # For adjacency matrices, use weighted average instead of concatenation
        # to maintain square matrix property
        weights = np.ones(len(matrices)) / len(matrices)
        fused_matrix = np.average(matrices, axis=0, weights=weights)
        return build_graph_from_matrix(fused_matrix, self.topk_ratio)
    
    def _weighted_fusion(self, matrices: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Weighted fusion of multiple matrices.
        """
        if len(matrices) == 1:
            return build_graph_from_matrix(matrices[0], self.topk_ratio)
        
        # Equal weights for now
        weights = np.ones(len(matrices)) / len(matrices)
        fused_matrix = np.average(matrices, axis=0, weights=weights)
        return build_graph_from_matrix(fused_matrix, self.topk_ratio)


def make_patient_splits(patient_to_files: Dict[str, List[str]], test_ratio: float = 0.2, val_ratio: float = 0.1, seed: int = 42):
    rng = np.random.RandomState(seed)
    patients = sorted(patient_to_files.keys())
    rng.shuffle(patients)
    n = len(patients)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    test_pat = set(patients[:n_test])
    val_pat = set(patients[n_test:n_test + n_val])
    train_pat = [p for p in patients if p not in test_pat and p not in val_pat]

    def gather(pats: List[str]) -> List[str]:
        out: List[str] = []
        for p in pats:
            out.extend(patient_to_files[p])
        return out

    return {
        "train": gather(train_pat),
        "val": gather(list(val_pat)),
        "test": gather(list(test_pat)),
        "train_patients": train_pat,
        "val_patients": list(val_pat),
        "test_patients": list(test_pat),
    }


