#!/usr/bin/env python3
"""
Simple test to verify the graph_augment fix.
"""

import torch
import numpy as np

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

def test():
    print("Testing graph_augment fix...")
    
    # Test with square matrix
    batch_size, n_nodes = 2, 10
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    x = torch.randn(batch_size, n_nodes, 2)
    
    try:
        adj1, x1 = graph_augment(adj, x, drop_edge=0.2, feat_noise=0.05)
        print(f"✓ Success: adj1={adj1.shape}, x1={x1.shape}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

if __name__ == "__main__":
    test()
