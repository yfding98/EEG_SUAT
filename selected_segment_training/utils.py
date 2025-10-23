"""
工具函数
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import shutil


class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state: dict, is_best: bool, save_dir: Path, 
                    filename: str = 'checkpoint.pth'):
    """
    保存检查点
    
    Args:
        state: 状态字典
        is_best: 是否是最佳模型
        save_dir: 保存目录
        filename: 文件名
    """
    save_path = save_dir / filename
    torch.save(state, save_path)
    
    if is_best:
        best_path = save_dir / 'best_model.pth'
        shutil.copyfile(save_path, best_path)


def load_checkpoint(checkpoint_path: Path, model: nn.Module, 
                    optimizer: Optional[optim.Optimizer] = None,
                    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None) -> dict:
    """
    加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        
    Returns:
        检查点字典
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


class EarlyStopping:
    """Early stopping"""
    
    def __init__(self, patience: int = 10, mode: str = 'min', delta: float = 0.0):
        """
        Args:
            patience: 等待多少个epoch没有改善就停止
            mode: 'min' 或 'max'
            delta: 最小改善量
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.delta
        else:
            return score > self.best_score + self.delta

