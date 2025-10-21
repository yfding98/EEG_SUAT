#!/usr/bin/env python3
"""
Debug script to identify the exact issue in training.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add training module to path
sys.path.append(str(Path(__file__).parent))

def test_graph_augment_detailed():
    """Test graph_augment with detailed debugging."""
    print("Testing graph_augment with detailed debugging...")
    
    # Test case 1: Normal 3D tensor
    print("\n1. Testing 3D tensor (batch_size, n_nodes, n_nodes)")
    batch_size, n_nodes = 2, 10
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    x = torch.randn(batch_size, n_nodes, 2)
    
    print(f"Input shapes: adj={adj.shape}, x={x.shape}")
    print(f"Adj dimensions: {adj.dim()}")
    print(f"Is square: {adj.shape[-1] == adj.shape[-2]}")
    
    try:
        from contrastive_pretrain import graph_augment
        adj1, x1 = graph_augment(adj, x, drop_edge=0.2, feat_noise=0.05)
        print(f"✓ Success: adj1={adj1.shape}, x1={x1.shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test case 2: Non-square matrix (this might be the issue)
    print("\n2. Testing non-square matrix")
    adj_nonsquare = torch.randn(batch_size, n_nodes, n_nodes + 1)
    print(f"Non-square adj shape: {adj_nonsquare.shape}")
    print(f"Is square: {adj_nonsquare.shape[-1] == adj_nonsquare.shape[-2]}")
    
    try:
        adj1, x1 = graph_augment(adj_nonsquare, x, drop_edge=0.2, feat_noise=0.05)
        print(f"✓ Success: adj1={adj1.shape}, x1={x1.shape}")
    except Exception as e:
        print(f"✗ Failed (expected): {e}")
    
    return True

def test_dataset_loading():
    """Test dataset loading to see what shapes we get."""
    print("\nTesting dataset loading...")
    
    try:
        from datasets import ConnectivityGraphDataset, load_labels_csv, discover_patient_segments_from_csv
        
        # Try to load a small dataset
        features_root = r'E:\output\connectivity_features'
        labels_csv = r'E:\output\connectivity_features\labels.csv'
        
        print(f"Loading labels from: {labels_csv}")
        labels_df = load_labels_csv(labels_csv)
        print(f"Labels shape: {labels_df.shape}")
        
        print(f"Discovering segments from: {features_root}")
        patient_to_files = discover_patient_segments_from_csv(labels_csv, features_root)
        print(f"Found {len(patient_to_files)} patients")
        
        # Get first patient's files
        first_patient = list(patient_to_files.keys())[0]
        first_files = patient_to_files[first_patient][:1]  # Just one file
        
        print(f"Testing with patient: {first_patient}, files: {len(first_files)}")
        
        # Create dataset
        dataset = ConnectivityGraphDataset(
            first_files, labels_df,
            matrix_keys=['plv_alpha'],
            fusion_method='weighted'
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Get one sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Adj shape: {sample['adj'].shape}")
        print(f"X shape: {sample['x'].shape}")
        print(f"Y: {sample['y']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run debug tests."""
    print("Debug Training Issues")
    print("=" * 50)
    
    # Test 1: Graph augmentation
    test_graph_augment_detailed()
    
    # Test 2: Dataset loading
    test_dataset_loading()
    
    print("\n" + "=" * 50)
    print("Debug complete!")

if __name__ == "__main__":
    main()
