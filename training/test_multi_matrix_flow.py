#!/usr/bin/env python3
"""
Test the complete multi-matrix flow from data loading to model forward pass.
"""

import torch
import sys
from pathlib import Path

# Add training module to path
sys.path.append(str(Path(__file__).parent.parent))

def test_multi_matrix_flow():
    """Test the complete flow."""
    print("Testing Multi-Matrix Flow")
    print("=" * 60)
    
    # Step 1: Simulate dataset output
    print("\n1. Simulating dataset output (multi-matrix)...")
    batch_size, n_nodes = 2, 10
    
    sample_multi = {
        "adj": torch.randn(n_nodes, n_nodes),
        "matrices": {
            "plv_alpha": torch.randn(n_nodes, n_nodes),
            "coherence_alpha": torch.randn(n_nodes, n_nodes),
            "wpli_alpha": torch.randn(n_nodes, n_nodes),
        },
        "x": torch.randn(n_nodes, 2),
        "y": torch.tensor(0, dtype=torch.long),
        "n": torch.tensor(n_nodes, dtype=torch.long),
    }
    print(f"✓ Sample has matrices: {list(sample_multi['matrices'].keys())}")
    
    # Step 2: Test collate function
    print("\n2. Testing collate function...")
    from training.contrastive_pretrain import collate_graph
    
    batch = [sample_multi, sample_multi]  # Batch of 2
    collated = collate_graph(batch)
    
    print(f"✓ Collated batch keys: {collated.keys()}")
    print(f"  - adj shape: {collated['adj'].shape}")
    print(f"  - x shape: {collated['x'].shape}")
    if 'matrices' in collated:
        print(f"  - matrices keys: {list(collated['matrices'].keys())}")
        for k, v in collated['matrices'].items():
            print(f"    - {k}: {v.shape}")
    
    # Step 3: Test model forward pass
    print("\n3. Testing model forward pass...")
    from training.models import ContrastiveModel
    
    model = ContrastiveModel(
        in_dim=2,
        hidden_dim=64,
        proj_dim=32,
        matrix_keys=["plv_alpha", "coherence_alpha", "wpli_alpha"],
        fusion_type="attention"
    )
    
    x = collated['x']
    adj = collated['adj']
    matrices = collated.get('matrices', None)
    
    print(f"  Input shapes:")
    print(f"    - x: {x.shape}")
    print(f"    - adj: {adj.shape}")
    if matrices:
        for k, v in matrices.items():
            print(f"    - {k}: {v.shape}")
    
    try:
        z, g = model(x, adj, matrices)
        print(f"✓ Model forward pass successful!")
        print(f"  - z (projection): {z.shape}")
        print(f"  - g (graph embedding): {g.shape}")
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test graph augmentation with matrices
    print("\n4. Testing graph augmentation with matrices...")
    from training.contrastive_pretrain import graph_augment
    
    for k, v in matrices.items():
        try:
            aug_mat, _ = graph_augment(v, x, drop_edge=0.2, feat_noise=0.0)
            print(f"✓ Augmented {k}: {aug_mat.shape}")
        except Exception as e:
            print(f"✗ Augmentation failed for {k}: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("\nThe multi-matrix flow is working correctly.")
    return True

if __name__ == "__main__":
    success = test_multi_matrix_flow()
    sys.exit(0 if success else 1)
