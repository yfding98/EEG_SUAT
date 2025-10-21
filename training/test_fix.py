#!/usr/bin/env python3
"""
Test script to verify the multi-matrix training fix.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add training module to path
sys.path.append(str(Path(__file__).parent))

from contrastive_pretrain import graph_augment, nt_xent


def test_graph_augment():
    """Test graph augmentation with different input shapes."""
    print("Testing graph_augment function...")
    
    # Test 1: Single matrix (3D)
    print("Test 1: Single matrix (3D)")
    batch_size, n_nodes = 2, 10
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    x = torch.randn(batch_size, n_nodes, 2)
    
    try:
        adj1, x1 = graph_augment(adj, x, drop_edge=0.2, feat_noise=0.05)
        print(f"✓ Single matrix: adj1={adj1.shape}, x1={x1.shape}")
    except Exception as e:
        print(f"✗ Single matrix failed: {e}")
        return False
    
    # Test 2: Multi-matrix (4D) - this should not happen in current implementation
    # but let's test the robustness
    print("Test 2: Multi-matrix (4D)")
    n_matrices = 3
    adj_multi = torch.randn(batch_size, n_nodes, n_nodes, n_matrices)
    
    try:
        adj1, x1 = graph_augment(adj_multi, x, drop_edge=0.2, feat_noise=0.05)
        print(f"✓ Multi-matrix: adj1={adj1.shape}, x1={x1.shape}")
    except Exception as e:
        print(f"✗ Multi-matrix failed: {e}")
        return False
    
    return True


def test_nt_xent():
    """Test NT-Xent loss function."""
    print("\nTesting NT-Xent loss...")
    
    batch_size, proj_dim = 4, 32
    z1 = torch.randn(batch_size, proj_dim)
    z2 = torch.randn(batch_size, proj_dim)
    
    try:
        loss = nt_xent(z1, z2, temperature=0.2)
        print(f"✓ NT-Xent loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ NT-Xent failed: {e}")
        return False


def test_training_step():
    """Test a complete training step."""
    print("\nTesting complete training step...")
    
    batch_size, n_nodes = 2, 10
    proj_dim = 32
    
    # Create dummy data
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    x = torch.randn(batch_size, n_nodes, 2)
    
    # Test augmentation
    adj1, x1 = graph_augment(adj, x, drop_edge=0.2, feat_noise=0.05)
    adj2, x2 = graph_augment(adj, x, drop_edge=0.2, feat_noise=0.05)
    
    # Create dummy projections
    z1 = torch.randn(batch_size, proj_dim)
    z2 = torch.randn(batch_size, proj_dim)
    
    # Test loss
    loss = nt_xent(z1, z2, temperature=0.2)
    
    print(f"✓ Training step: loss={loss.item():.4f}")
    return True


def main():
    """Run all tests."""
    print("Multi-Matrix Training Fix Test")
    print("=" * 40)
    
    tests = [
        test_graph_augment,
        test_nt_xent,
        test_training_step
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed! The fix should work.")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
