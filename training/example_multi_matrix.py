#!/usr/bin/env python3
"""
Example script demonstrating multi-matrix attention fusion for EEG connectivity analysis.

This script shows how to use the attention-based multi-matrix fusion system
with different connectivity matrices (PLV, coherence, wPLI, etc.).
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add training module to path
sys.path.append(str(Path(__file__).parent))

from datasets import ConnectivityGraphDataset, discover_patient_segments_from_csv, make_patient_splits, load_labels_csv
from attention_fusion import MatrixAttentionFusion, MultiMatrixGCNEncoder
from models import ContrastiveModel, SupervisedModel


def test_attention_fusion():
    """Test the attention fusion mechanism."""
    print("Testing Matrix Attention Fusion...")
    
    # Create dummy data
    batch_size, n_nodes = 2, 10
    matrix_keys = ["plv_alpha", "coherence_alpha", "wpli_alpha"]
    
    # Create dummy matrices
    matrices = {
        key: torch.randn(batch_size, n_nodes, n_nodes) 
        for key in matrix_keys
    }
    
    # Test attention fusion
    fusion = MatrixAttentionFusion(
        matrix_keys=matrix_keys, 
        hidden_dim=64, 
        num_heads=2,
        fusion_type="cross_attention"
    )
    
    fused_matrix = fusion(matrices)
    print(f"✓ Fused matrix shape: {fused_matrix.shape}")
    
    # Test different fusion types
    for fusion_type in ["cross_attention", "self_attention", "gated"]:
        fusion = MatrixAttentionFusion(
            matrix_keys=matrix_keys,
            hidden_dim=64,
            num_heads=2,
            fusion_type=fusion_type
        )
        fused = fusion(matrices)
        print(f"✓ {fusion_type} fusion: {fused.shape}")
    
    return True


def test_multi_matrix_gcn():
    """Test the multi-matrix GCN encoder."""
    print("\nTesting Multi-Matrix GCN Encoder...")
    
    batch_size, n_nodes = 2, 10
    matrix_keys = ["plv_alpha", "coherence_alpha", "wpli_alpha"]
    
    # Create dummy data
    x = torch.randn(batch_size, n_nodes, 2)  # Node features
    matrices = {
        key: torch.randn(batch_size, n_nodes, n_nodes) 
        for key in matrix_keys
    }
    
    # Test multi-matrix GCN
    gcn = MultiMatrixGCNEncoder(
        matrix_keys=matrix_keys,
        in_dim=2,
        hidden_dim=64,
        out_dim=32,
        fusion_type="cross_attention"
    )
    
    output = gcn(x, matrices)
    print(f"✓ Multi-matrix GCN output shape: {output.shape}")
    
    return True


def test_contrastive_model():
    """Test the contrastive model with multi-matrix support."""
    print("\nTesting Contrastive Model with Multi-Matrix...")
    
    batch_size, n_nodes = 2, 10
    matrix_keys = ["plv_alpha", "coherence_alpha", "wpli_alpha"]
    
    # Create dummy data
    x = torch.randn(batch_size, n_nodes, 2)
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    matrices = {
        key: torch.randn(batch_size, n_nodes, n_nodes) 
        for key in matrix_keys
    }
    
    # Test contrastive model
    model = ContrastiveModel(
        in_dim=2,
        hidden_dim=64,
        proj_dim=32,
        matrix_keys=matrix_keys,
        fusion_type="attention"
    )
    
    z, g = model(x, adj, matrices)
    print(f"✓ Contrastive model output shapes: z={z.shape}, g={g.shape}")
    
    return True


def test_supervised_model():
    """Test the supervised model with multi-matrix support."""
    print("\nTesting Supervised Model with Multi-Matrix...")
    
    batch_size, n_nodes = 2, 10
    matrix_keys = ["plv_alpha", "coherence_alpha", "wpli_alpha"]
    
    # Create dummy data
    x = torch.randn(batch_size, n_nodes, 2)
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    matrices = {
        key: torch.randn(batch_size, n_nodes, n_nodes) 
        for key in matrix_keys
    }
    
    # Test supervised model
    model = SupervisedModel(
        in_dim=2,
        hidden_dim=64,
        num_classes=3,
        matrix_keys=matrix_keys,
        fusion_type="attention"
    )
    
    logits, g = model(x, adj, matrices)
    print(f"✓ Supervised model output shapes: logits={logits.shape}, g={g.shape}")
    
    return True


def test_dataset_loading():
    """Test dataset loading with multi-matrix support."""
    print("\nTesting Dataset Loading with Multi-Matrix...")
    
    # This would require actual data files
    print("Note: This test requires actual connectivity feature files")
    print("To test with real data, use:")
    print("python -m training.contrastive_pretrain \\")
    print("    --features_root E:\\output\\connectivity_features \\")
    print("    --labels_csv E:\\output\\connectivity_features\\labels.csv \\")
    print("    --matrix_keys plv_alpha coherence_alpha wpli_alpha \\")
    print("    --fusion_method attention")
    
    return True


def main():
    """Run all tests."""
    print("Multi-Matrix Attention Fusion Test Suite")
    print("=" * 50)
    
    tests = [
        test_attention_fusion,
        test_multi_matrix_gcn,
        test_contrastive_model,
        test_supervised_model,
        test_dataset_loading
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
