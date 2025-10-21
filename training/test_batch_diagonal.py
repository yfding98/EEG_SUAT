#!/usr/bin/env python3
"""
Test the batch diagonal fill fix.
"""

import torch

def test_fill_diagonal():
    """Test different ways to fill diagonal in batched tensors."""
    print("Testing fill_diagonal with batched tensors...")
    
    # Test 1: Single 2D matrix (works)
    print("\n1. Single 2D matrix:")
    A = torch.randn(5, 5)
    print(f"Shape: {A.shape}")
    A.fill_diagonal_(0.0)
    print(f"✓ Success, diagonal sum: {A.diagonal().sum()}")
    
    # Test 2: Batched 3D tensor (fails with direct fill_diagonal_)
    print("\n2. Batched 3D tensor (direct fill_diagonal_):")
    A = torch.randn(2, 5, 5)
    print(f"Shape: {A.shape}")
    try:
        A.fill_diagonal_(0.0)
        print(f"✓ Success, diagonal sum: {A[:, range(5), range(5)].sum()}")
    except RuntimeError as e:
        print(f"✗ Failed (expected): {e}")
    
    # Test 3: Batched 3D tensor (loop fix)
    print("\n3. Batched 3D tensor (loop fix):")
    A = torch.randn(2, 5, 5)
    print(f"Shape: {A.shape}")
    try:
        for i in range(A.shape[0]):
            A[i].fill_diagonal_(0.0)
        print(f"✓ Success, diagonal sum: {A[:, range(5), range(5)].sum()}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 4: Using indexing (alternative)
    print("\n4. Batched 3D tensor (indexing alternative):")
    A = torch.randn(2, 5, 5)
    print(f"Shape: {A.shape}")
    try:
        n = A.shape[-1]
        A[:, range(n), range(n)] = 0.0
        print(f"✓ Success, diagonal sum: {A[:, range(5), range(5)].sum()}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\n" + "=" * 50)
    print("Conclusion: Loop method works for batched tensors!")

if __name__ == "__main__":
    test_fill_diagonal()
