#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_basic_eeg.py

基础EEG通道分类器推理脚本
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
import random
from typing import Dict, List, Tuple

# 导入模型和数据集
from basic_eeg_channel_classifier import create_basic_eeg_classifier, compute_multilabel_metrics
from dataset_selected import create_dataloaders


def load_model(checkpoint_path: str, n_channels: int, n_samples: int, n_bands: int, device: torch.device):
    """加载训练好的模型"""
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 创建模型
    model = create_basic_eeg_classifier(
        n_channels=n_channels,
        n_samples=n_samples,
        d_model=128,
        n_heads=8,
        n_layers=2,
        dropout=0.3
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def predict_channels(
    model: torch.nn.Module,
    bands_data: List[torch.Tensor],
    threshold: float = 0.5,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    预测异常通道
    
    Args:
        model: 训练好的模型
        bands_data: 多频段数据列表
        threshold: 分类阈值
        device: 设备
    
    Returns:
        logits: 原始logits
        probabilities: 概率值
        predictions: 二分类预测
    """
    if device is None:
        device = next(model.parameters()).device
    
    with torch.no_grad():
        # 将数据移到设备上
        bands_list = [band.to(device) for band in bands_data]
        
        # 预测
        logits = model(bands_list)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()
        
        return logits, probabilities, predictions


def analyze_prediction_results(
    true_labels: torch.Tensor,
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    channel_names: List[str],
    threshold: float = 0.5
) -> Dict:
    """
    分析预测结果
    
    Args:
        true_labels: 真实标签 (n_channels,)
        predictions: 预测标签 (n_channels,)
        probabilities: 预测概率 (n_channels,)
        channel_names: 通道名称列表
        threshold: 分类阈值
    
    Returns:
        分析结果字典
    """
    # 转换为numpy
    true_labels = true_labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    # 计算指标
    metrics = compute_multilabel_metrics(
        torch.from_numpy(predictions).unsqueeze(0),
        torch.from_numpy(true_labels).unsqueeze(0),
        threshold
    )
    
    # 找出真实和预测的异常通道
    true_abnormal = [channel_names[i] for i, label in enumerate(true_labels) if label == 1]
    pred_abnormal = [channel_names[i] for i, label in enumerate(predictions) if label == 1]
    
    # 计算每个通道的详细信息
    channel_details = []
    for i, ch_name in enumerate(channel_names):
        channel_details.append({
            'channel': ch_name,
            'true_label': int(true_labels[i]),
            'pred_label': int(predictions[i]),
            'probability': float(probabilities[i]),
            'correct': true_labels[i] == predictions[i]
        })
    
    # 按概率排序
    channel_details.sort(key=lambda x: x['probability'], reverse=True)
    
    return {
        'metrics': metrics,
        'true_abnormal_channels': true_abnormal,
        'pred_abnormal_channels': pred_abnormal,
        'channel_details': channel_details,
        'threshold': threshold,
        'n_true_abnormal': len(true_abnormal),
        'n_pred_abnormal': len(pred_abnormal)
    }


def quick_inference(
    checkpoint_path: str,
    data_root: str,
    sample_idx: int = None,
    threshold: float = 0.5,
    window_size: float = 6.0,
    window_stride: float = 3.0
) -> Dict:
    """
    快速推理函数
    
    Args:
        checkpoint_path: 检查点路径
        data_root: 数据根目录
        sample_idx: 样本索引（None表示随机选择）
        threshold: 分类阈值
        window_size: 窗口大小
        window_stride: 窗口步长
    
    Returns:
        推理结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载配置
    config_path = Path(checkpoint_path).parent / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {'d_model': 128, 'n_heads': 8, 'n_layers': 2, 'dropout': 0.3}
    
    # 2. 准备数据
    print("准备数据...")
    train_loader, val_loader, test_loader, channel_names = create_dataloaders(
        data_root=data_root,
        batch_size=1,
        window_size=window_size,
        window_stride=window_stride,
        val_split=0.15,
        test_split=0.15,
        num_workers=0,
        seed=42
    )
    
    # 3. 获取模型参数
    sample_batch = next(iter(val_loader))
    n_channels = sample_batch['bands'][0].shape[1]
    n_samples = sample_batch['bands'][0].shape[2]
    n_bands = len(sample_batch['bands'])
    
    # 4. 加载模型
    print(f"加载模型: {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path, n_channels, n_samples, n_bands, device)
    
    # 5. 选择样本
    if sample_idx is None:
        sample_idx = random.randint(0, len(val_loader.dataset) - 1)
    
    print(f"选择样本: {sample_idx}")
    sample = val_loader.dataset[sample_idx]
    
    # 6. 预测
    print("进行预测...")
    bands = sample['bands']
    labels = sample['labels']
    
    # 添加batch维度
    bands_tensor = [band.unsqueeze(0) for band in bands]
    labels = labels.unsqueeze(0)
    
    # 预测
    logits, probabilities, predictions = predict_channels(
        model, bands_tensor, threshold, device
    )
    
    # 7. 分析结果
    results = analyze_prediction_results(
        labels.squeeze(0),
        predictions.squeeze(0),
        probabilities.squeeze(0),
        channel_names,
        threshold
    )
    
    # 8. 打印结果
    print(f"\n{'='*60}")
    print("预测结果")
    print(f"{'='*60}")
    print(f"阈值: {threshold}")
    print(f"样本索引: {sample_idx}")
    print(f"文件: {sample.get('file', 'Unknown')}")
    print()
    
    print("通道预测详情:")
    for detail in results['channel_details']:
        status = "✅正确" if detail['correct'] else "❌错误"
        print(f"  {detail['channel']}: 真实={detail['true_label']}, "
              f"预测={detail['pred_label']}, 概率={detail['probability']:.3f} {status}")
    
    print(f"\n统计信息:")
    print(f"  真实异常通道数: {results['n_true_abnormal']}")
    print(f"  预测异常通道数: {results['n_pred_abnormal']}")
    print(f"  真实异常通道: {results['true_abnormal_channels']}")
    print(f"  预测异常通道: {results['pred_abnormal_channels']}")
    
    print(f"\n性能指标:")
    metrics = results['metrics']
    print(f"  Macro F1: {metrics['macro_f1']:.2f}%")
    print(f"  Macro Precision: {metrics['macro_precision']:.2f}%")
    print(f"  Macro Recall: {metrics['macro_recall']:.2f}%")
    print(f"  Micro F1: {metrics['micro_f1']:.2f}%")
    
    return results


def batch_inference(
    checkpoint_path: str,
    data_root: str,
    threshold: float = 0.5,
    window_size: float = 6.0,
    window_stride: float = 3.0,
    max_samples: int = 100
) -> Dict:
    """
    批量推理
    
    Args:
        checkpoint_path: 检查点路径
        data_root: 数据根目录
        threshold: 分类阈值
        window_size: 窗口大小
        window_stride: 窗口步长
        max_samples: 最大样本数
    
    Returns:
        批量推理结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    print("准备数据...")
    train_loader, val_loader, test_loader, channel_names = create_dataloaders(
        data_root=data_root,
        batch_size=8,
        window_size=window_size,
        window_stride=window_stride,
        val_split=0.15,
        test_split=0.15,
        num_workers=0,
        seed=42
    )
    
    # 获取模型参数
    sample_batch = next(iter(val_loader))
    n_channels = sample_batch['bands'][0].shape[1]
    n_samples = sample_batch['bands'][0].shape[2]
    n_bands = len(sample_batch['bands'])
    
    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path, n_channels, n_samples, n_bands, device)
    
    # 批量预测
    print("进行批量预测...")
    all_metrics = []
    sample_results = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_samples:
                break
                
            bands = batch['bands']
            labels = batch['labels'].to(device)
            
            # 将多频段数据转换为列表格式
            bands_list = [band.to(device) for band in bands]
            
            # 预测
            logits = model(bands_list)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()
            
            # 计算指标
            metrics = compute_multilabel_metrics(logits, labels, threshold)
            all_metrics.append(metrics)
            
            # 保存样本结果
            for j in range(labels.size(0)):
                sample_results.append({
                    'sample_idx': i * 8 + j,
                    'true_abnormal_count': labels[j].sum().item(),
                    'pred_abnormal_count': predictions[j].sum().item(),
                    'macro_f1': metrics['macro_f1']
                })
    
    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    print(f"\n批量推理结果 (共{len(all_metrics)}个样本):")
    print(f"  平均Macro F1: {avg_metrics['macro_f1']:.2f}%")
    print(f"  平均Macro Precision: {avg_metrics['macro_precision']:.2f}%")
    print(f"  平均Macro Recall: {avg_metrics['macro_recall']:.2f}%")
    print(f"  平均Micro F1: {avg_metrics['micro_f1']:.2f}%")
    
    return {
        'avg_metrics': avg_metrics,
        'sample_results': sample_results,
        'n_samples': len(all_metrics)
    }


if __name__ == "__main__":
    # 使用示例
    checkpoint_path = "checkpoints_basic_eeg/basic_eeg_20250101_120000/best_model.pth"
    data_root = r"E:\DataSet\EEG\EEG dataset_SUAT_processed_selected"
    
    # 检查文件是否存在
    if not Path(checkpoint_path).exists():
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        print("请修改 checkpoint_path 为正确的路径")
        sys.exit(1)
    
    if not Path(data_root).exists():
        print(f"错误: 数据目录不存在: {data_root}")
        print("请修改 data_root 为正确的路径")
        sys.exit(1)
    
    # 运行推理
    try:
        print("=== 单样本推理 ===")
        results = quick_inference(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            sample_idx=None,  # 随机选择
            threshold=0.5
        )
        
        print(f"\n=== 批量推理 ===")
        batch_results = batch_inference(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            threshold=0.5,
            max_samples=50
        )
        
        print(f"\n推理完成！")
        
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
