#!/usr/bin/env python3
"""
Test script to verify the NT-Xent loss fix.
"""

import torch
import torch.nn.functional as F

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2):
    """NT-Xent loss for contrastive learning."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    B = z1.size(0)
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels])
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))
    loss = F.cross_entropy(sim, labels)
    return loss


def test_nt_xent():
    """Test NT-Xent loss with different batch sizes."""
    print("Testing NT-Xent Loss Function")
    print("=" * 60)
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    proj_dim = 128
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create random embeddings
        z1 = torch.randn(batch_size, proj_dim)
        z2 = torch.randn(batch_size, proj_dim)
        
        print(f"  z1 shape: {z1.shape}")
        print(f"  z2 shape: {z2.shape}")
        
        try:
            loss = nt_xent(z1, z2, temperature=0.2)
            print(f"  ✓ Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    return True


def test_model_output():
    """Test that model outputs have correct shape."""
    print("\nTesting Model Output Shapes")
    print("=" * 60)
    
    from training.models import GCNEncoder, ContrastiveModel
    
    batch_size = 8
    n_nodes = 20
    in_dim = 2
    hidden_dim = 64
    proj_dim = 32
    
    # Test GCNEncoder
    print("\n1. Testing GCNEncoder:")
    encoder = GCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
    
    # Batch input
    x = torch.randn(batch_size, n_nodes, in_dim)
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    
    print(f"   Input x: {x.shape}")
    print(f"   Input adj: {adj.shape}")
    
    g = encoder(x, adj)
    print(f"   Output g: {g.shape}")
    
    if g.shape[0] != batch_size:
        print(f"   ✗ FAIL: Expected batch dimension {batch_size}, got {g.shape[0]}")
        return False
    else:
        print(f"   ✓ PASS: Correct batch dimension")
    
    # Test ContrastiveModel
    print("\n2. Testing ContrastiveModel:")
    model = ContrastiveModel(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim
    )
    
    z, g = model(x, adj)
    print(f"   Output z: {z.shape}")
    print(f"   Output g: {g.shape}")
    
    if z.shape[0] != batch_size:
        print(f"   ✗ FAIL: Expected batch dimension {batch_size}, got {z.shape[0]}")
        return False
    else:
        print(f"   ✓ PASS: Correct batch dimension")
    
    # Test NT-Xent with model outputs
    print("\n3. Testing NT-Xent with model outputs:")
    z1, _ = model(x, adj)
    z2, _ = model(x, adj)
    
    try:
        loss = nt_xent(z1, z2, temperature=0.2)
        print(f"   ✓ PASS: Loss = {loss.item():.4f}")
    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All model tests passed!")
    return True


def test_multi_matrix_encoder():
    """Test multi-matrix encoder output shapes."""
    print("\nTesting Multi-Matrix Encoder")
    print("=" * 60)
    
    from training.attention_fusion import MultiMatrixGCNEncoder
    
    batch_size = 8
    n_nodes = 20
    in_dim = 2
    hidden_dim = 64
    
    print("\n1. Testing MultiMatrixGCNEncoder:")
    encoder = MultiMatrixGCNEncoder(
        matrix_keys=['plv_alpha', 'coherence_alpha', 'wpli_alpha'],
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=hidden_dim,
        fusion_type='cross_attention'
    )
    
    # Create multi-matrix input
    x = torch.randn(batch_size, n_nodes, in_dim)
    matrices = {
        'plv_alpha': torch.randn(batch_size, n_nodes, n_nodes),
        'coherence_alpha': torch.randn(batch_size, n_nodes, n_nodes),
        'wpli_alpha': torch.randn(batch_size, n_nodes, n_nodes),
    }
    
    print(f"   Input x: {x.shape}")
    for k, v in matrices.items():
        print(f"   Input {k}: {v.shape}")
    
    g = encoder(x, matrices)
    print(f"   Output g: {g.shape}")
    
    if g.shape[0] != batch_size:
        print(f"   ✗ FAIL: Expected batch dimension {batch_size}, got {g.shape[0]}")
        return False
    else:
        print(f"   ✓ PASS: Correct batch dimension")
    
    print("\n" + "=" * 60)
    print("✓ Multi-matrix encoder test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "🔍 NT-Xent Loss and Model Shape Tests".center(60))
    print()
    
    tests = [
        ("NT-Xent Loss", test_nt_xent),
        ("Model Outputs", test_model_output),
        ("Multi-Matrix Encoder", test_multi_matrix_encoder),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:30s}: {status}")
    
    print("\n" + "=" * 60)
    
    if all(p for _, p in results):
        print("✓ ALL TESTS PASSED!")
        print("\nYou can now run training without shape errors.")
        return 0
    else:
        print("✗ SOME TESTS FAILED!")
        print("\nPlease check the errors above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
