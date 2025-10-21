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


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
    """
    计算top-k准确率
    
    Args:
        output: 模型输出 (batch_size, n_classes)
        target: 真实标签 (batch_size,)
        topk: 要计算的top-k值
        
    Returns:
        top-k准确率列表
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
    # PyTorch 2.6+兼容性：weights_only=False
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


def get_lr_scheduler(optimizer: optim.Optimizer, 
                     scheduler_type: str = 'cosine',
                     **kwargs) -> optim.lr_scheduler._LRScheduler:
    """
    获取学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        **kwargs: 其他参数
        
    Returns:
        学习率调度器
    """
    if scheduler_type == 'cosine':
        n_epochs = kwargs.get('n_epochs', 100)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-6
        )
    elif scheduler_type == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def compute_class_weights(targets: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    计算类别权重（用于不平衡数据）
    
    Args:
        targets: 目标标签数组
        n_classes: 类别数
        
    Returns:
        类别权重张量
    """
    class_counts = np.bincount(targets, minlength=n_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * n_classes
    return torch.FloatTensor(class_weights)


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple:
    """
    Mixup数据增强
    
    Args:
        x: 输入数据
        y: 标签
        alpha: mixup参数
        
    Returns:
        混合后的数据、标签a、标签b、lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothing(nn.Module):
    """标签平滑"""
    
    def __init__(self, n_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测logits (batch_size, n_classes)
            target: 真实标签 (batch_size,)
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def get_confusion_matrix(predictions: np.ndarray, targets: np.ndarray, 
                        n_classes: int) -> np.ndarray:
    """
    计算混淆矩阵
    
    Args:
        predictions: 预测标签
        targets: 真实标签
        n_classes: 类别数
        
    Returns:
        混淆矩阵
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(targets, predictions):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray, class_names: Optional[list] = None):
    """打印混淆矩阵"""
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # 打印表头
    print("\nConfusion Matrix:")
    print(" " * 15 + " ".join(f"{name:>10}" for name in class_names))
    
    # 打印每一行
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>15}" + " ".join(f"{val:>10}" for val in row))
    
    # 计算每个类别的准确率
    print("\nPer-class Accuracy:")
    for i in range(n_classes):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum() * 100
            print(f"  {class_names[i]}: {acc:.2f}%")


if __name__ == "__main__":
    # 测试工具函数
    print("Testing utility functions...")
    
    # 测试accuracy
    output = torch.randn(4, 5)
    target = torch.tensor([0, 1, 2, 3])
    acc = accuracy(output, target)
    print(f"Accuracy: {acc[0].item():.2f}%")
    
    # 测试confusion matrix
    pred = np.array([0, 1, 2, 0, 1, 2])
    targ = np.array([0, 1, 1, 0, 2, 2])
    cm = get_confusion_matrix(pred, targ, 3)
    print_confusion_matrix(cm, ['Class 0', 'Class 1', 'Class 2'])

