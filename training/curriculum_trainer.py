#!/usr/bin/env python3
"""
课程学习训练器

从易到难逐步训练:
Stage 1: 只训练高频通道（容易学习）
Stage 2: 加入中频通道
Stage 3: 训练所有通道
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict


class CurriculumTrainer:
    """
    课程学习训练策略
    
    原理:
        人类学习也是从易到难
        模型先学会"容易"的通道
        然后迁移知识到"困难"的通道
    """
    
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        channel_names,
        channel_frequencies,
        stage_epochs=[30, 50, 70],  # 每个阶段的epoch数
        device='cuda'
    ):
        """
        Args:
            model: 多标签分类模型
            criterion: 损失函数
            optimizer: 优化器
            channel_names: 通道名称列表
            channel_frequencies: [num_channels] 每个通道的样本数
            stage_epochs: [stage1_end, stage2_end, stage3_end]
            device: 设备
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.channel_names = channel_names
        self.device = device
        
        # 根据频率分组通道
        self.high_freq_channels, self.mid_freq_channels, self.low_freq_channels = \
            self._group_channels_by_frequency(channel_frequencies)
        
        self.stage_epochs = stage_epochs
        self.current_stage = 1
        
        print(f"\nCurriculumTrainer initialized:")
        print(f"  Stage 1 (0-{stage_epochs[0]}): High-freq channels ({len(self.high_freq_channels)})")
        for idx in self.high_freq_channels[:5]:
            print(f"    - {channel_names[idx]}")
        
        print(f"  Stage 2 ({stage_epochs[0]}-{stage_epochs[1]}): + Mid-freq channels ({len(self.mid_freq_channels)})")
        for idx in self.mid_freq_channels[:5]:
            print(f"    - {channel_names[idx]}")
        
        print(f"  Stage 3 ({stage_epochs[1]}-{stage_epochs[2]}): All channels ({len(self.high_freq_channels) + len(self.mid_freq_channels) + len(self.low_freq_channels)})")
    
    def _group_channels_by_frequency(self, channel_frequencies):
        """按频率分组通道"""
        # 排序
        sorted_indices = np.argsort(channel_frequencies)[::-1]  # 降序
        
        # 过滤掉0样本的通道
        non_zero_indices = [i for i in sorted_indices if channel_frequencies[i] > 0]
        
        n = len(non_zero_indices)
        
        # 分组：前33%高频，中33%中频，后33%低频
        split1 = n // 3
        split2 = 2 * n // 3
        
        high_freq = non_zero_indices[:split1]
        mid_freq = non_zero_indices[split1:split2]
        low_freq = non_zero_indices[split2:]
        
        return high_freq, mid_freq, low_freq
    
    def get_active_channels(self, epoch):
        """获取当前epoch应该训练的通道"""
        if epoch <= self.stage_epochs[0]:
            # Stage 1: 只训练高频通道
            self.current_stage = 1
            return self.high_freq_channels
        elif epoch <= self.stage_epochs[1]:
            # Stage 2: 高频+中频
            self.current_stage = 2
            return self.high_freq_channels + self.mid_freq_channels
        else:
            # Stage 3: 所有通道
            self.current_stage = 3
            return self.high_freq_channels + self.mid_freq_channels + self.low_freq_channels
    
    def train_step(self, batch, epoch):
        """
        单步训练，根据当前stage只计算部分通道的loss
        使用渐进式策略：新阶段开始时降低学习率，避免破坏已学习的通道
        
        Args:
            batch: 训练batch
            epoch: 当前epoch
        
        Returns:
            loss: 标量loss
        """
        adj = batch['adj'].to(self.device)
        x = batch['x'].to(self.device)
        targets = batch['y'].to(self.device)
        
        # 检测是否刚进入新阶段（需要降低学习率）
        if epoch in self.stage_epochs:
            # 刚进入新阶段，临时降低学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
            print(f"\n  ⚠ Entering new stage, reducing LR to {param_group['lr']:.6f}")
        
        # 前向传播（全部通道）
        logits = self.model(x, adj)
        
        # 获取当前应该训练的通道
        active_channels = self.get_active_channels(epoch)
        
        # 创建mask
        channel_mask = torch.zeros(logits.size(1), dtype=torch.bool, device=self.device)
        channel_mask[active_channels] = True
        
        # 只计算活跃通道的loss
        active_logits = logits[:, channel_mask]
        active_targets = targets[:, channel_mask]
        
        # 计算loss
        # 如果损失函数支持channel_indices参数（如ChannelAdaptiveFocalLoss），传递它
        try:
            import inspect
            sig = inspect.signature(self.criterion.forward)
            if 'channel_indices' in sig.parameters:
                # 支持channel_indices
                loss = self.criterion(active_logits, active_targets, channel_indices=active_channels)
            else:
                # 不支持，正常调用
                loss = self.criterion(active_logits, active_targets)
        except:
            # 出错时使用默认调用
            loss = self.criterion(active_logits, active_targets)
        
        return loss
    
    def print_stage_info(self, epoch):
        """打印当前阶段信息"""
        if epoch == 1:
            print(f"\n{'='*80}")
            print(f"Stage 1: Training on {len(self.high_freq_channels)} high-frequency channels")
            print(f"{'='*80}")
        elif epoch == self.stage_epochs[0] + 1:
            print(f"\n{'='*80}")
            print(f"Stage 2: Adding {len(self.mid_freq_channels)} mid-frequency channels")
            print(f"{'='*80}")
        elif epoch == self.stage_epochs[1] + 1:
            print(f"\n{'='*80}")
            print(f"Stage 3: Training on all {len(self.high_freq_channels) + len(self.mid_freq_channels) + len(self.low_freq_channels)} channels")
            print(f"{'='*80}")


if __name__ == "__main__":
    print("Testing Curriculum Trainer...")
    
    # 模拟设置
    from models_multilabel import MultiLabelGNNClassifier
    from losses import AsymmetricLoss
    
    num_channels = 10
    channel_frequencies = [50, 45, 30, 25, 15, 10, 5, 3, 1, 0]
    channel_names = [f"Ch{i}" for i in range(num_channels)]
    
    model = MultiLabelGNNClassifier(
        in_dim=2,
        hidden_dim=128,
        num_channels=num_channels
    )
    
    criterion = AsymmetricLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建curriculum trainer
    trainer = CurriculumTrainer(
        model, criterion, optimizer,
        channel_names, channel_frequencies,
        stage_epochs=[30, 60, 100]
    )
    
    # 测试不同epoch的活跃通道
    print("\nTesting active channels at different epochs:")
    for epoch in [1, 30, 31, 60, 61, 100]:
        active = trainer.get_active_channels(epoch)
        print(f"  Epoch {epoch}: {len(active)} active channels (Stage {trainer.current_stage})")
    
    print("\n✓ Curriculum trainer working!")

