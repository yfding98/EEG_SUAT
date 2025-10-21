#!/usr/bin/env python3
"""
测试课程学习与ChannelAdaptiveFocalLoss的兼容性
"""

import torch
import numpy as np
from losses_advanced import ChannelAdaptiveFocalLoss


def test_channel_adaptive_with_curriculum():
    """测试ChannelAdaptiveFocalLoss与课程学习的兼容性"""
    
    print("=" * 80)
    print("Testing ChannelAdaptiveFocalLoss with Curriculum Learning")
    print("=" * 80)
    
    # 模拟数据
    num_channels = 11
    batch_size = 8
    
    # 创建一些模拟的通道频率
    channel_frequencies = {
        0: 100,  # 高频
        1: 90,
        2: 80,
        3: 50,   # 中频
        4: 45,
        5: 40,
        6: 20,   # 低频
        7: 15,
        8: 10,
        9: 5,
        10: 3
    }
    
    # 创建损失函数
    print("\n1. Creating ChannelAdaptiveFocalLoss...")
    criterion = ChannelAdaptiveFocalLoss(
        channel_frequencies=channel_frequencies,
        base_gamma=2.0,
        base_alpha=0.25
    )
    
    print(f"\n   Channel gammas: {criterion.channel_gammas.tolist()}")
    print(f"   Channel alphas: {criterion.channel_alphas.tolist()}")
    
    # 测试1: 全部通道（正常情况）
    print("\n2. Testing with all channels...")
    logits_all = torch.randn(batch_size, num_channels)
    targets_all = torch.randint(0, 2, (batch_size, num_channels)).float()
    
    try:
        loss_all = criterion(logits_all, targets_all)
        print(f"   ✓ Loss (all channels): {loss_all.item():.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # 测试2: Stage 1（高频通道）
    print("\n3. Testing Stage 1 (high-frequency channels)...")
    stage1_channels = [0, 1, 2]  # 索引
    stage1_logits = logits_all[:, stage1_channels]
    stage1_targets = targets_all[:, stage1_channels]
    
    print(f"   Active channels: {stage1_channels}")
    print(f"   Logits shape: {stage1_logits.shape}")
    
    try:
        # 使用channel_indices
        loss_stage1 = criterion(stage1_logits, stage1_targets, channel_indices=stage1_channels)
        print(f"   ✓ Loss (stage 1 with indices): {loss_stage1.item():.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试3: Stage 2（高频+中频）
    print("\n4. Testing Stage 2 (high + mid frequency)...")
    stage2_channels = [0, 1, 2, 3, 4, 5]
    stage2_logits = logits_all[:, stage2_channels]
    stage2_targets = targets_all[:, stage2_channels]
    
    print(f"   Active channels: {stage2_channels}")
    print(f"   Logits shape: {stage2_logits.shape}")
    
    try:
        loss_stage2 = criterion(stage2_logits, stage2_targets, channel_indices=stage2_channels)
        print(f"   ✓ Loss (stage 2 with indices): {loss_stage2.item():.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试4: Stage 3（所有通道，包括低频）
    print("\n5. Testing Stage 3 (all channels including low-frequency)...")
    stage3_channels = list(range(num_channels))
    stage3_logits = logits_all
    stage3_targets = targets_all
    
    print(f"   Active channels: {stage3_channels}")
    print(f"   Logits shape: {stage3_logits.shape}")
    
    try:
        loss_stage3 = criterion(stage3_logits, stage3_targets, channel_indices=stage3_channels)
        print(f"   ✓ Loss (stage 3 with indices): {loss_stage3.item():.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试5: 反向传播
    print("\n6. Testing backward pass...")
    try:
        loss_stage1.backward()
        print(f"   ✓ Backward pass successful")
    except Exception as e:
        print(f"   ✗ Backward failed: {e}")
        return False
    
    # 测试6: 没有channel_indices（向后兼容）
    print("\n7. Testing without channel_indices (backward compatibility)...")
    try:
        loss_compat = criterion(stage1_logits, stage1_targets)
        print(f"   ✓ Loss (without indices): {loss_compat.item():.4f}")
        print(f"   ⚠ Warning: This assumes first N channels, may not be correct for curriculum!")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    
    return True


def test_non_sequential_channels():
    """测试非连续通道索引"""
    
    print("\n" + "=" * 80)
    print("Testing Non-Sequential Channel Indices")
    print("=" * 80)
    
    num_channels = 10
    batch_size = 4
    
    channel_frequencies = {i: 100 - i * 10 for i in range(num_channels)}
    
    criterion = ChannelAdaptiveFocalLoss(
        channel_frequencies=channel_frequencies,
        base_gamma=2.0
    )
    
    # 测试非连续索引（例如：[2, 5, 7]）
    print("\n1. Testing non-sequential indices [2, 5, 7]...")
    non_seq_channels = [2, 5, 7]
    
    logits = torch.randn(batch_size, len(non_seq_channels))
    targets = torch.randint(0, 2, (batch_size, len(non_seq_channels))).float()
    
    try:
        loss = criterion(logits, targets, channel_indices=non_seq_channels)
        print(f"   ✓ Loss: {loss.item():.4f}")
        print(f"   Used gammas for channels {non_seq_channels}: {criterion.channel_gammas[non_seq_channels].tolist()}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试倒序索引
    print("\n2. Testing reverse order indices [9, 5, 1]...")
    reverse_channels = [9, 5, 1]
    
    logits = torch.randn(batch_size, len(reverse_channels))
    targets = torch.randint(0, 2, (batch_size, len(reverse_channels))).float()
    
    try:
        loss = criterion(logits, targets, channel_indices=reverse_channels)
        print(f"   ✓ Loss: {loss.item():.4f}")
        print(f"   Used gammas for channels {reverse_channels}: {criterion.channel_gammas[reverse_channels].tolist()}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✓ Non-sequential tests passed!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    print("\n" + "🧪 Curriculum Learning + Channel Adaptive Loss Test".center(80))
    print()
    
    success1 = test_channel_adaptive_with_curriculum()
    success2 = test_non_sequential_channels()
    
    if success1 and success2:
        print("\n" + "=" * 80)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("=" * 80)
        print("\nCurriculum learning with ChannelAdaptiveFocalLoss is ready!")
        print("\nYou can now run: training\\run_ultimate.bat")
    else:
        print("\n" + "=" * 80)
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease check the errors above.")

