#!/usr/bin/env python3
"""
测试对比学习模块
"""

import torch
from models_multilabel import MultiLabelGNNClassifier
from contrastive_pretrain_multilabel import ContrastiveEncoder, nt_xent_loss
from augmentations_graph import GraphAugmentor


def test_contrastive_encoder():
    """测试对比学习编码器"""
    print("=" * 60)
    print("Testing Contrastive Encoder")
    print("=" * 60)
    
    # 创建模型
    print("\n1. Creating MultiLabelGNNClassifier...")
    model = MultiLabelGNNClassifier(
        in_dim=2,
        hidden_dim=128,
        num_channels=10,
        num_layers=3,
        dropout=0.3
    )
    print(f"   Model hidden_dim: {model.hidden_dim}")
    
    # 创建对比学习编码器
    print("\n2. Creating ContrastiveEncoder...")
    contrastive_model = ContrastiveEncoder(model, projection_dim=64)
    print(f"   Encoder hidden_dim: {contrastive_model.hidden_dim}")
    
    # 测试前向传播
    print("\n3. Testing forward pass...")
    batch_size, n_nodes, n_features = 8, 20, 2
    
    x = torch.randn(batch_size, n_nodes, n_features)
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    
    print(f"   Input x: {x.shape}")
    print(f"   Input adj: {adj.shape}")
    
    try:
        z, g = contrastive_model(x, adj)
        print(f"   ✓ Output z (projection): {z.shape}")
        print(f"   ✓ Output g (embedding): {g.shape}")
        
        # 验证维度
        assert z.shape == (batch_size, 64), f"Expected z shape (8, 64), got {z.shape}"
        assert g.shape == (batch_size, 128), f"Expected g shape (8, 128), got {g.shape}"
        
        print(f"\n   ✓ Dimensions correct!")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试对比损失
    print("\n4. Testing NT-Xent loss...")
    
    # 创建两个增强
    augmentor = GraphAugmentor(
        edge_dropout_rate=0.2,
        node_dropout_rate=0.1,
        feature_noise_std=0.05
    )
    
    adj1, x1, _ = augmentor.augment(adj, x)
    adj2, x2, _ = augmentor.augment(adj, x)
    
    z1, _ = contrastive_model(x1, adj1)
    z2, _ = contrastive_model(x2, adj2)
    
    print(f"   z1: {z1.shape}")
    print(f"   z2: {z2.shape}")
    
    loss = nt_xent_loss(z1, z2, temperature=0.1)
    print(f"   ✓ NT-Xent loss: {loss.item():.4f}")
    
    # 测试反向传播
    print("\n5. Testing backward pass...")
    try:
        loss.backward()
        print(f"   ✓ Backward pass successful")
    except Exception as e:
        print(f"   ✗ Backward failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
    return True


def test_with_different_model_sizes():
    """测试不同大小的模型"""
    print("\n" + "=" * 60)
    print("Testing Different Model Sizes")
    print("=" * 60)
    
    configs = [
        {'hidden_dim': 128, 'num_layers': 3},
        {'hidden_dim': 256, 'num_layers': 4},
        {'hidden_dim': 512, 'num_layers': 5},
    ]
    
    for config in configs:
        print(f"\nTesting: hidden_dim={config['hidden_dim']}, layers={config['num_layers']}")
        
        model = MultiLabelGNNClassifier(
            in_dim=2,
            hidden_dim=config['hidden_dim'],
            num_channels=10,
            num_layers=config['num_layers']
        )
        
        contrastive_model = ContrastiveEncoder(model, projection_dim=128)
        
        # 测试
        x = torch.randn(4, 20, 2)
        adj = torch.randn(4, 20, 20)
        
        try:
            z, g = contrastive_model(x, adj)
            print(f"  ✓ z: {z.shape}, g: {g.shape}")
            
            assert z.shape[1] == 128, f"Projection dim should be 128, got {z.shape[1]}"
            assert g.shape[1] == config['hidden_dim'], f"Hidden dim should be {config['hidden_dim']}, got {g.shape[1]}"
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False
    
    print("\n✓ All configurations working!")
    return True


if __name__ == "__main__":
    print("\n" + "🔍 Contrastive Learning Module Test".center(60))
    print()
    
    success1 = test_contrastive_encoder()
    success2 = test_with_different_model_sizes()
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("=" * 60)
        print("\nContrastive pretraining is ready to use!")
        print("You can now run: training\\run_ultimate.bat")
    else:
        print("\n" + "=" * 60)
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease check the errors above.")

