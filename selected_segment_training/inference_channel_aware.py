#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_channel_aware.py

通道感知模型推理脚本
加载训练好的模型，从验证集随机抽取样本进行预测分析
"""

import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
import argparse
import json
import sys
from datetime import datetime

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent / 'raw_data_training'))

from model_channel_aware_multilabel import ChannelAwareMultilabelNet, create_channel_aware_multilabel_model
from dataset_selected import create_dataloaders


def load_model_and_config(checkpoint_path, device):
    """加载模型和配置"""
    print(f"加载模型: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取配置信息
    config_path = Path(checkpoint_path).parent / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"模型配置: {config}")
    else:
        print("警告: 未找到配置文件，使用默认配置")
        config = {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'dropout': 0.3
        }
    
    return checkpoint, config


def create_model_from_checkpoint(checkpoint, config, n_channels, n_samples, n_bands, device):
    """从检查点创建模型"""
    print(f"创建模型: {n_channels}通道, {n_samples}时间点, {n_bands}频段")
    
    # 创建模型
    model = create_channel_aware_multilabel_model(
        n_channels=n_channels,
        n_samples=n_samples,
        n_bands=n_bands,
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 2),
        dropout=config.get('dropout', 0.3)
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def predict_sample(model, sample, device, threshold=0.5):
    """对单个样本进行预测"""
    with torch.no_grad():
        # 准备输入数据
        bands = sample['bands']  # List of tensors
        labels = sample['labels']  # (n_channels,)
        
        # 添加batch维度
        bands_tensor = torch.stack(bands, dim=0).unsqueeze(0).to(device)  # (1, n_bands, n_channels, n_samples)
        labels = labels.unsqueeze(0).to(device)  # (1, n_channels)
        
        # 预测
        logits = model(bands_tensor, labels)  # (1, n_channels)
        probs = torch.sigmoid(logits)  # (1, n_channels)
        pred_binary = (probs > threshold).float()  # (1, n_channels)
        
        return {
            'logits': logits.squeeze(0).cpu(),  # (n_channels,)
            'probs': probs.squeeze(0).cpu(),     # (n_channels,)
            'pred_binary': pred_binary.squeeze(0).cpu(),  # (n_channels,)
            'true_labels': labels.squeeze(0).cpu()  # (n_channels,)
        }


def format_channel_results(channel_names, true_labels, pred_binary, probs, threshold=0.5):
    """格式化通道结果"""
    results = []
    
    for i, (channel_name, true_label, pred_label, prob) in enumerate(zip(
        channel_names, true_labels, pred_binary, probs
    )):
        # 确定状态
        if true_label == 1 and pred_label == 1:
            status = "✅ 正确预测为正"
        elif true_label == 0 and pred_label == 0:
            status = "✅ 正确预测为负"
        elif true_label == 1 and pred_label == 0:
            status = "❌ 漏检 (假阴性)"
        else:  # true_label == 0 and pred_label == 1
            status = "❌ 误报 (假阳性)"
        
        results.append({
            'channel': channel_name,
            'true_label': int(true_label),
            'pred_label': int(pred_label),
            'probability': float(prob),
            'status': status
        })
    
    return results


def print_detailed_results(results, threshold=0.5):
    """打印详细结果"""
    print(f"\n{'='*80}")
    print("详细预测结果")
    print(f"{'='*80}")
    print(f"阈值: {threshold}")
    print(f"{'通道':<8} {'真实':<4} {'预测':<4} {'概率':<8} {'状态':<20}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['channel']:<8} {result['true_label']:<4} {result['pred_label']:<4} "
              f"{result['probability']:<8.3f} {result['status']:<20}")
    
    # 统计信息
    total_channels = len(results)
    correct_predictions = sum(1 for r in results if r['status'].startswith("✅"))
    false_positives = sum(1 for r in results if "误报" in r['status'])
    false_negatives = sum(1 for r in results if "漏检" in r['status'])
    
    print(f"\n统计信息:")
    print(f"  总通道数: {total_channels}")
    print(f"  正确预测: {correct_predictions} ({correct_predictions/total_channels*100:.1f}%)")
    print(f"  误报: {false_positives}")
    print(f"  漏检: {false_negatives}")
    
    # 活跃通道分析
    true_active = [r for r in results if r['true_label'] == 1]
    pred_active = [r for r in results if r['pred_label'] == 1]
    
    print(f"\n活跃通道分析:")
    print(f"  真实活跃通道: {len(true_active)}个")
    if true_active:
        active_channels = [r['channel'] for r in true_active]
        print(f"    通道: {', '.join(active_channels)}")
    
    print(f"  预测活跃通道: {len(pred_active)}个")
    if pred_active:
        pred_channels = [r['channel'] for r in pred_active]
        print(f"    通道: {', '.join(pred_channels)}")


def print_probability_ranking(results):
    """打印概率排序"""
    print(f"\n{'='*80}")
    print("通道概率排序 (从高到低)")
    print(f"{'='*80}")
    
    # 按概率排序
    sorted_results = sorted(results, key=lambda x: x['probability'], reverse=True)
    
    print(f"{'排名':<4} {'通道':<8} {'概率':<8} {'真实':<4} {'预测':<4} {'状态':<20}")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:<4} {result['channel']:<8} {result['probability']:<8.3f} "
              f"{result['true_label']:<4} {result['pred_label']:<4} {result['status']:<20}")


def main():
    parser = argparse.ArgumentParser(description='通道感知模型推理')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录')
    
    # 可选参数
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='指定样本索引，None表示随机选择')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='预测阈值')
    parser.add_argument('--window_size', type=float, default=6.0,
                        help='窗口大小')
    parser.add_argument('--window_stride', type=float, default=3.0,
                        help='窗口步长')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型和配置
    checkpoint, config = load_model_and_config(args.checkpoint, device)
    
    # 创建数据加载器
    print(f"\n准备数据...")
    print(f"  数据路径: {args.data_root}")
    
    # 从配置中获取窗口参数
    window_size = config.get('window_size', args.window_size)
    window_stride = config.get('window_stride', args.window_stride)
    
    print(f"  使用窗口大小: {window_size}秒 (来自模型配置)")
    print(f"  使用窗口步长: {window_stride}秒 (来自模型配置)")
    
    try:
        train_loader, val_loader, test_loader, channel_names = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            window_size=window_size,
            window_stride=window_stride,
            val_split=0.15,
            test_split=0.15,
            num_workers=0,
            seed=args.seed
        )
        
        print(f"  通道数: {len(channel_names)}")
        print(f"  验证集大小: {len(val_loader.dataset)}")
        
    except Exception as e:
        print(f"错误：加载数据失败: {e}")
        return
    
    # 获取样本信息
    sample_batch = next(iter(val_loader))
    n_channels = sample_batch['bands'][0].shape[1]
    n_samples = sample_batch['bands'][0].shape[2]
    n_bands = len(sample_batch['bands'])
    
    # 创建模型
    model = create_model_from_checkpoint(
        checkpoint, config, n_channels, n_samples, n_bands, device
    )
    
    # 选择样本
    if args.sample_idx is not None:
        if args.sample_idx >= len(val_loader.dataset):
            print(f"错误：样本索引 {args.sample_idx} 超出范围 (0-{len(val_loader.dataset)-1})")
            return
        sample_idx = args.sample_idx
        print(f"\n使用指定样本: {sample_idx}")
    else:
        sample_idx = random.randint(0, len(val_loader.dataset) - 1)
        print(f"\n随机选择样本: {sample_idx}")
    
    # 获取样本
    sample = val_loader.dataset[sample_idx]
    print(f"样本文件：{sample['file']}")
    # 预测
    print(f"\n进行预测...")
    results = predict_sample(model, sample, device, args.threshold)
    
    # 格式化结果
    formatted_results = format_channel_results(
        channel_names, 
        results['true_labels'], 
        results['pred_binary'], 
        results['probs'],
        args.threshold
    )
    
    # 打印结果
    print_detailed_results(formatted_results, args.threshold)
    print_probability_ranking(formatted_results)
    
    # 保存结果
    output_dir = Path(args.checkpoint).parent / 'inference_results'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'inference_sample_{sample_idx}_{timestamp}.json'
    
    # 准备保存数据
    save_data = {
        'sample_idx': sample_idx,
        'threshold': args.threshold,
        'channel_names': channel_names,
        'results': formatted_results,
        'timestamp': timestamp
    }
    
    with open(output_file, 'w',encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    # 默认参数示例
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--checkpoint', 'checkpoints_channel_aware/channel_aware_20251024_130336/best_model.pth',
            '--data_root', r'E:\DataSet\EEG\EEG dataset_SUAT_processed_selected',
            '--threshold', '0.5'
        ])
    
    main()
