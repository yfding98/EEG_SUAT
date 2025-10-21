#!/usr/bin/env python3
"""
对比学习预训练 for 多标签分类

先用对比学习学习好的图表征（无监督）
然后再做多标签分类微调（监督）

优点:
- 可以利用所有数据（不需要标签）
- 学习更鲁棒的表征
- 提升少样本通道的性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from augmentations_graph import GraphAugmentor


class ContrastiveEncoder(nn.Module):
    """
    对比学习编码器
    """
    
    def __init__(self, model, projection_dim=128):
        """
        Args:
            model: MultiLabelGNNClassifier 完整模型
            projection_dim: 投影维度
        """
        super().__init__()
        self.encoder = model.encoder  # GCNEncoder
        
        # 获取hidden_dim（从模型的分类器）
        # MultiLabelGNNClassifier.classifier 是 Sequential
        # 第一层是 Linear(hidden_dim, ...)
        if hasattr(model, 'hidden_dim'):
            hidden_dim = model.hidden_dim
        else:
            # 从classifier的第一层获取
            first_layer = model.classifier[0]
            if isinstance(first_layer, nn.Linear):
                hidden_dim = first_layer.in_features
            else:
                # 默认值
                hidden_dim = 128
        
        self.hidden_dim = hidden_dim
        
        # 投影头（用于对比学习）
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        print(f"  ContrastiveEncoder: hidden_dim={hidden_dim}, projection_dim={projection_dim}")
    
    def forward(self, x, adj):
        """
        Args:
            x: [B, N, F]
            adj: [B, N, N]
        
        Returns:
            z: [B, projection_dim] 投影特征
            g: [B, hidden_dim] 图嵌入
        """
        g = self.encoder(x, adj)  # [B, hidden_dim]
        z = self.projection(g)    # [B, projection_dim]
        z = F.normalize(z, dim=-1)  # L2归一化
        return z, g


def nt_xent_loss(z1, z2, temperature=0.1):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
    
    用于对比学习
    """
    batch_size = z1.size(0)
    
    # 拼接
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    
    # 计算相似度矩阵
    sim_matrix = torch.matmul(z, z.t()) / temperature  # [2B, 2B]
    
    # 创建标签：每个样本的正样本是其增强版本
    labels = torch.arange(batch_size).to(z.device)
    labels = torch.cat([labels + batch_size, labels])  # [2B]
    
    # Mask掉对角线（自己和自己的相似度）
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
    
    # Cross-entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss


def pretrain_contrastive(
    model,
    train_loader,
    epochs=50,
    lr=0.001,
    temperature=0.1,
    device='cuda',
    save_path='contrastive_pretrained.pt'
):
    """
    对比学习预训练
    
    Args:
        model: MultiLabelGNNClassifier
        train_loader: 数据加载器（不需要标签）
        epochs: 预训练轮数
        lr: 学习率
        temperature: 温度参数
        device: 设备
        save_path: 保存路径
    
    Returns:
        pretrained_model: 预训练后的模型
    """
    from tqdm import tqdm
    
    # 创建对比学习编码器
    contrastive_model = ContrastiveEncoder(model, projection_dim=128).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(contrastive_model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 图增强器
    augmentor = GraphAugmentor(
        edge_dropout_rate=0.3,
        node_dropout_rate=0.1,
        feature_noise_std=0.1,
        use_mixup=False  # 对比学习不用mixup
    )
    
    print(f"\n{'='*80}")
    print("Contrastive Pretraining")
    print(f"{'='*80}")
    print(f"Epochs: {epochs}")
    print(f"Temperature: {temperature}")
    print(f"{'='*80}\n")
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        contrastive_model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Pretrain Epoch {epoch}/{epochs}')
        
        for batch in pbar:
            adj = batch['adj'].to(device)
            x = batch['x'].to(device)
            
            # 两种不同的增强
            adj1, x1, _ = augmentor.augment(adj, x)
            adj2, x2, _ = augmentor.augment(adj, x)
            
            # 编码
            z1, _ = contrastive_model(x1, adj1)
            z2, _ = contrastive_model(x2, adj2)
            
            # 对比loss
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            
            # 优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(contrastive_model.parameters(), 5.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch}: Contrastive Loss = {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'loss': best_loss
            }, save_path)
            print(f"  ✓ Saved best pretrained model (loss={best_loss:.4f})")
    
    print(f"\n✓ Contrastive pretraining completed!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Model saved to: {save_path}")
    
    return model


if __name__ == "__main__":
    print("Testing Curriculum Trainer...")
    
    from models_multilabel import MultiLabelGNNClassifier
    from losses import AsymmetricLoss
    
    # 创建模型
    model = MultiLabelGNNClassifier(
        in_dim=2,
        hidden_dim=128,
        num_channels=10
    )
    
    criterion = AsymmetricLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # 模拟频率
    channel_names = [f"Ch{i}" for i in range(10)]
    channel_freqs = [50, 45, 30, 25, 15, 10, 5, 3, 1, 0]
    
    # 创建trainer
    trainer = CurriculumTrainer(
        model, criterion, optimizer,
        channel_names, channel_freqs,
        stage_epochs=[30, 60, 100],
        device='cpu'
    )
    
    print("\n✓ Curriculum trainer created successfully!")

