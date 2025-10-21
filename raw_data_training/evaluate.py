"""
评估脚本
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import RawEEGDataset
from torch.utils.data import DataLoader
from model import create_model
from raw_data_training.model_channel_aware import create_channel_aware_model
from utils import (
    AverageMeter, accuracy, load_checkpoint,
    get_confusion_matrix, print_confusion_matrix
)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_classes: int
):
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        n_classes: 类别数
        
    Returns:
        结果字典
    """
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            data = batch['data'].to(device)
            target = batch['label'].to(device)
            
            # Forward
            output = model(data)
            loss = criterion(output, target)
            
            # 统计
            acc = accuracy(output, target)[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc.item(), data.size(0))
            
            # 保存结果
            probs = torch.softmax(output, dim=1)
            _, pred = output.max(1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{top1.avg:.2f}%'
            })
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # 计算混淆矩阵
    cm = get_confusion_matrix(all_preds, all_targets, n_classes)
    
    # 计算每个类别的指标
    class_metrics = {}
    for i in range(n_classes):
        # True Positives, False Positives, False Negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[i] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(cm[i, :].sum())
        }
    
    results = {
        'loss': losses.avg,
        'accuracy': top1.avg,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
        'confusion_matrix': cm,
        'class_metrics': class_metrics
    }
    
    return results


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, class_names=None):
    """绘制混淆矩阵"""
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_class_distribution(targets: np.ndarray, save_path: Path, class_names=None):
    """绘制类别分布"""
    unique, counts = np.unique(targets, return_counts=True)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in unique]
    
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class distribution saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate EEG Classification Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--data_root', type=str,
                        default=r'E:\DataSet\EEG\EEG dataset_SUAT_processed',
                        help='Data root directory')
    parser.add_argument('--labels_csv', type=str,
                        default=r'E:\output\connectivity_features\labels.csv',
                        help='Labels CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载检查点配置
    checkpoint_dir = Path(args.checkpoint).parent
    config_path = checkpoint_dir / 'config.json'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Loaded configuration from checkpoint directory")
    else:
        print("Warning: config.json not found, using default settings")
        config = {
            'model_type': 'lightweight',
            'n_classes': 5,
            'window_size': 6.0
        }
    
    # 创建数据集
    print("Loading dataset...")
    dataset = RawEEGDataset(
        data_root=args.data_root,
        labels_csv=args.labels_csv,
        window_size=config.get('window_size', 6.0),
        use_cache=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 获取数据形状
    sample_batch = next(iter(dataloader))
    n_channels = sample_batch['data'].shape[1]
    n_samples = sample_batch['data'].shape[2]
    n_classes = config['n_classes']
    
    print(f"Data shape: channels={n_channels}, samples={n_samples}")
    print(f"Number of classes: {n_classes}")
    
    # 创建模型
    if 'model_type' not in config:
        model = create_channel_aware_model(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            dropout=config['dropout']
        )

    else:
        print(f"Creating {config['model_type']} model...")
        model = create_model(
            model_type=config['model_type'],
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes
        )
    model = model.to(device)
    
    # 加载检查点
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(args.checkpoint, model)
    
    # 评估
    print("\nEvaluating model...")
    results = evaluate_model(model, dataloader, device, n_classes)
    
    # 打印结果
    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"{'='*50}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    
    print("\nPer-class Metrics:")
    for i, metrics in results['class_metrics'].items():
        print(f"\nClass {i}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Support: {metrics['support']}")
    
    # 打印混淆矩阵
    class_names = [f"Class {i}" for i in range(n_classes)]
    print_confusion_matrix(results['confusion_matrix'], class_names)
    
    # 保存结果
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数值结果
    np.savez(
        save_dir / 'evaluation_results.npz',
        predictions=results['predictions'],
        targets=results['targets'],
        probabilities=results['probabilities'],
        confusion_matrix=results['confusion_matrix'],
        accuracy=results['accuracy'],
        loss=results['loss']
    )
    
    # 保存类别指标
    metrics_dict = {
        'accuracy': float(results['accuracy']),
        'loss': float(results['loss']),
        'class_metrics': {
            str(k): {
                'precision': float(v['precision']),
                'recall': float(v['recall']),
                'f1': float(v['f1']),
                'support': int(v['support'])
            } for k, v in results['class_metrics'].items()
        }
    }
    
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # 绘制图表
    print("\nGenerating plots...")
    plot_confusion_matrix(
        results['confusion_matrix'],
        save_dir / 'confusion_matrix.png',
        class_names
    )
    
    plot_class_distribution(
        results['targets'],
        save_dir / 'class_distribution.png',
        class_names
    )
    
    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":

    import sys
    if len(sys.argv) < 2:
        sys.argv.extend([
            '--checkpoint', r'E:\code_learn\SUAT\workspace\EEG-projects\LaBraM\raw_data_training\checkpoints_channel_aware\channel_aware_20251020_214050\checkpoint.pth',
            '--data_root', r'E:\DataSet\EEG\EEG dataset_SUAT_processed',
            '--labels_csv', r'E:\output\connectivity_features_v2\labels.csv'
        ])
    main()

