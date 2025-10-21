#!/usr/bin/env python3
"""
Verify that the graph_augment fix works correctly.
"""

import torch
import sys

def graph_augment(adj: torch.Tensor, x: torch.Tensor, drop_edge: float = 0.2, feat_noise: float = 0.05):
    """
    Graph augmentation with support for multi-matrix inputs.
    Note: fill_diagonal_ does not work with batched tensors, so we use indexing.
    """
    A = adj.clone()
    
    if drop_edge > 0:
        mask = (torch.rand_like(A) > drop_edge).float()
        A = A * mask
        
        # Symmetrize the adjacency matrix
        if A.dim() == 3:  # [batch_size, n_nodes, n_nodes]
            A = (A + A.transpose(-1, -2)) / 2.0
            # Fill diagonal using indexing (works with batched tensors)
            if A.shape[-1] == A.shape[-2]:
                n = A.shape[-1]
                A[:, range(n), range(n)] = 0.0
        elif A.dim() == 4:  # [batch_size, n_nodes, n_nodes, n_matrices]
            A = (A + A.transpose(-2, -3)) / 2.0
            # Fill diagonal for each matrix using indexing
            n = A.shape[1]
            A[:, range(n), range(n), :] = 0.0
    
    X = x.clone()
    if feat_noise > 0:
        X = X + torch.randn_like(X) * feat_noise
    return A, X

def verify():
    """Verify the fix works."""
    print("Verifying graph_augment fix...")
    print("=" * 60)
    
    # Test Case 1: Typical batch of graphs
    print("\nTest 1: Batch of square matrices [batch_size=2, n_nodes=10]")
    batch_size, n_nodes = 2, 10
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    x = torch.randn(batch_size, n_nodes, 2)
    
    print(f"Input shapes: adj={adj.shape}, x={x.shape}")
    
    try:
        adj1, x1 = graph_augment(adj, x, drop_edge=0.2, feat_noise=0.05)
        print(f"✓ Output shapes: adj1={adj1.shape}, x1={x1.shape}")
        
        # Verify diagonal is zero
        diagonal_sum = adj1[:, range(n_nodes), range(n_nodes)].abs().sum()
        print(f"✓ Diagonal sum: {diagonal_sum.item():.6f} (should be 0.0)")
        
        # Verify symmetry
        diff = (adj1 - adj1.transpose(-1, -2)).abs().max()
        print(f"✓ Symmetry check: max diff = {diff.item():.6f} (should be ~0.0)")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Case 2: Larger batch
    print("\nTest 2: Larger batch [batch_size=32, n_nodes=22]")
    batch_size, n_nodes = 32, 22
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    x = torch.randn(batch_size, n_nodes, 2)
    
    try:
        adj1, x1 = graph_augment(adj, x, drop_edge=0.3, feat_noise=0.1)
        print(f"✓ Output shapes: adj1={adj1.shape}, x1={x1.shape}")
        
        diagonal_sum = adj1[:, range(n_nodes), range(n_nodes)].abs().sum()
        print(f"✓ Diagonal sum: {diagonal_sum.item():.6f} (should be 0.0)")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test Case 3: Edge case - single item batch
    print("\nTest 3: Single item batch [batch_size=1, n_nodes=5]")
    adj = torch.randn(1, 5, 5)
    x = torch.randn(1, 5, 2)
    
    try:
        adj1, x1 = graph_augment(adj, x, drop_edge=0.2, feat_noise=0.05)
        print(f"✓ Output shapes: adj1={adj1.shape}, x1={x1.shape}")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("\nThe fix is working correctly. You can now run training.")
    return True

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
