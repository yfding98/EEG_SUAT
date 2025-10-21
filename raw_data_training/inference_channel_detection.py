"""
活跃通道检测推理脚本
输入：.set文件的EEG片段
输出：哪些通道是活跃的（发作源）
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from data_loader import EEGWindowExtractor
from model_channel_detection import create_channel_detector


class ChannelDetectionInference:
    """活跃通道检测推理器"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        config_path = self.checkpoint_path.parent / 'config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.window_size = self.config.get('window_size', 6.0)
        self.normalization = self.config.get('normalization', 'window_robust')
        
        self.model = None
        self.extractor = EEGWindowExtractor(window_size=self.window_size)
        self.channel_names = None
        
        print(f"推理器已初始化")
        print(f"  检查点: {checkpoint_path}")
        print(f"  设备: {self.device}")
        print(f"  窗口大小: {self.window_size}秒")
        print(f"  归一化: {self.normalization}")
    
    def _init_model(self, n_channels, n_samples):
        """初始化模型"""
        if self.model is not None:
            return
        
        self.model = create_channel_detector(
            n_channels=n_channels,
            n_samples=n_samples,
            d_model=self.config.get('d_model', 256),
            n_heads=self.config.get('n_heads', 8),
            n_layers=self.config.get('n_layers', 4),
            dropout=self.config.get('dropout', 0.3)
        )
        
        # 加载权重
        checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"模型已加载: {sum(p.numel() for p in self.model.parameters()):,} 参数")
    
    def _normalize_window(self, window_data):
        """归一化"""
        if self.normalization == 'window_robust':
            median = np.median(window_data)
            mad = np.median(np.abs(window_data - median))
            return (window_data - median) / (mad * 1.4826 + 1e-8)
        elif self.normalization == 'window_zscore':
            mean = window_data.mean()
            std = window_data.std()
            return (window_data - mean) / (std + 1e-8)
        elif self.normalization == 'channel_robust':
            normalized = np.zeros_like(window_data)
            for ch in range(window_data.shape[0]):
                median = np.median(window_data[ch])
                mad = np.median(np.abs(window_data[ch] - median))
                normalized[ch] = (window_data[ch] - median) / (mad * 1.4826 + 1e-8)
            return normalized
        return window_data
    
    def _get_channel_names(self, set_file_path):
        """获取通道名"""
        import mne
        raw = mne.io.read_raw_eeglab(set_file_path, preload=False, verbose=False)
        return raw.ch_names
    
    def predict_window(self, set_file_path, window_idx=0):
        """
        预测单个窗口的活跃通道
        
        Args:
            set_file_path: .set文件路径
            window_idx: 窗口索引
        
        Returns:
            结果字典
        """
        print(f"\n{'='*70}")
        print(f"活跃通道检测")
        print(f"{'='*70}")
        print(f"\n文件: {set_file_path}")
        
        # 获取通道名
        self.channel_names = self._get_channel_names(set_file_path)
        print(f"通道数: {len(self.channel_names)}")
        print(f"通道名: {self.channel_names}")
        
        # 提取窗口
        print(f"\n提取{self.window_size}秒窗口...")
        windows, info = self.extractor.extract_windows(set_file_path)
        
        print(f"提取到 {len(windows)} 个窗口")
        print(f"采样率: {info['sfreq']} Hz")
        
        if len(windows) == 0:
            print("未提取到有效窗口！")
            return None
        
        if window_idx >= len(windows):
            print(f"窗口索引{window_idx}超出范围，使用窗口0")
            window_idx = 0
        
        # 初始化模型
        n_channels = windows.shape[1]
        n_samples = windows.shape[2]
        self._init_model(n_channels, n_samples)
        
        # 获取窗口数据
        window_data = windows[window_idx]
        
        # 归一化
        window_normalized = self._normalize_window(window_data)
        
        # 转为tensor
        window_tensor = torch.FloatTensor(window_normalized).unsqueeze(0)
        window_tensor = window_tensor.to(self.device)
        
        # 推理
        print(f"\n推理窗口 #{window_idx}...")
        with torch.no_grad():
            channel_logits = self.model(window_tensor)  # (1, n_channels)
            channel_probs = torch.sigmoid(channel_logits)[0].cpu().numpy()  # (n_channels,)
        
        # 识别活跃通道（概率>0.5）
        active_indices = np.where(channel_probs > 0.5)[0]
        active_channels = [self.channel_names[i] for i in active_indices]
        active_probs = channel_probs[active_indices]
        
        # 结果
        result = {
            'window_idx': window_idx,
            'n_total_channels': n_channels,
            'n_active_channels': len(active_channels),
            'active_channel_indices': active_indices.tolist(),
            'active_channel_names': active_channels,
            'active_channel_probs': active_probs.tolist(),
            'all_channel_probs': channel_probs.tolist(),
            'channel_names': self.channel_names
        }
        
        # 打印结果
        self._print_result(result)
        
        return result
    
    def _print_result(self, result):
        """打印预测结果"""
        print(f"\n{'='*70}")
        print(f"预测结果 - 窗口 #{result['window_idx']}")
        print(f"{'='*70}")
        
        print(f"\n活跃通道 (共{result['n_active_channels']}个):")
        if result['n_active_channels'] > 0:
            for name, prob in zip(result['active_channel_names'], 
                                 result['active_channel_probs']):
                print(f"  ✓ {name:10s} {prob*100:6.2f}%")
        else:
            print(f"  （未检测到活跃通道，可能阈值过高）")
        
        # Top 10通道（按概率排序）
        print(f"\nTop 10 通道（按激活概率）:")
        sorted_indices = np.argsort(result['all_channel_probs'])[::-1][:10]
        
        for rank, idx in enumerate(sorted_indices, 1):
            ch_name = result['channel_names'][idx]
            prob = result['all_channel_probs'][idx]
            is_active = '✓ 活跃' if prob > 0.5 else ''
            print(f"  {rank:2d}. {ch_name:10s} {prob*100:6.2f}% {is_active}")
    
    def visualize_result(self, result, save_path='channel_detection_result.png'):
        """可视化结果"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 通道激活概率横条图
        ax = axes[0]
        channel_names = result['channel_names']
        probs = np.array(result['all_channel_probs']) * 100
        colors = ['red' if p > 50 else 'lightblue' for p in probs]
        
        y_pos = np.arange(len(channel_names))
        ax.barh(y_pos, probs, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(channel_names)
        ax.set_xlabel('激活概率 (%)', fontsize=12)
        ax.set_title(f'通道激活概率 (窗口 #{result["window_idx"]})', 
                     fontsize=14, fontweight='bold')
        ax.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='阈值(50%)')
        ax.set_xlim([0, 100])
        ax.legend()
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # 2. 活跃通道摘要
        ax = axes[1]
        ax.axis('off')
        
        summary_text = f"活跃通道检测结果\n"
        summary_text += f"{'='*40}\n\n"
        summary_text += f"窗口: #{result['window_idx']}\n"
        summary_text += f"总通道数: {result['n_total_channels']}\n"
        summary_text += f"活跃通道数: {result['n_active_channels']}\n\n"
        
        summary_text += f"检测到的活跃通道:\n"
        summary_text += f"{'-'*40}\n"
        
        if result['n_active_channels'] > 0:
            for name, prob in zip(result['active_channel_names'], 
                                 result['active_channel_probs']):
                summary_text += f"✓ {name:10s} {prob*100:6.2f}%\n"
        else:
            summary_text += "（未检测到活跃通道）\n"
        
        summary_text += f"\n{'-'*40}\n"
        summary_text += f"\nTop 5 高激活通道:\n"
        
        sorted_indices = np.argsort(result['all_channel_probs'])[::-1][:5]
        for rank, idx in enumerate(sorted_indices, 1):
            ch_name = channel_names[idx]
            prob = result['all_channel_probs'][idx]
            summary_text += f"{rank}. {ch_name:10s} {prob*100:5.2f}%\n"
        
        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
               family='monospace')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n可视化已保存: {save_path}")
        
        return save_path
    
    def predict_and_save(self, set_file_path, window_idx=0, save_dir='detection_results'):
        """预测并保存结果"""
        # 执行预测
        result = self.predict_window(set_file_path, window_idx)
        
        if result is None:
            return None
        
        # 创建保存目录
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = save_dir / f'detection_{timestamp}.json'
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nJSON结果已保存: {json_path}")
        
        # 可视化
        viz_path = save_dir / f'detection_{timestamp}.png'
        self.visualize_result(result, viz_path)
        
        return result


def main():
    parser = argparse.ArgumentParser(description='活跃通道检测推理')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--set_file', type=str, required=True,
                        help='.set文件路径')
    parser.add_argument('--window_idx', type=int, default=0,
                        help='窗口索引（默认0）')
    parser.add_argument('--save_dir', type=str, default='detection_results',
                        help='结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # 检查文件
    if not Path(args.checkpoint).exists():
        print(f"错误: 检查点不存在: {args.checkpoint}")
        return
    
    if not Path(args.set_file).exists():
        print(f"错误: .set文件不存在: {args.set_file}")
        return
    
    # 创建推理器
    inference = ChannelDetectionInference(args.checkpoint, device=args.device)
    
    # 执行推理
    result = inference.predict_and_save(
        set_file_path=args.set_file,
        window_idx=args.window_idx,
        save_dir=args.save_dir
    )
    
    if result:
        print(f"\n{'='*70}")
        print("推理完成！")
        print(f"{'='*70}")
        print(f"\n检测到 {result['n_active_channels']} 个活跃通道:")
        for ch_name in result['active_channel_names']:
            print(f"  • {ch_name}")
        
        print(f"\n结果已保存到: {args.save_dir}/")


if __name__ == "__main__":
    main()

