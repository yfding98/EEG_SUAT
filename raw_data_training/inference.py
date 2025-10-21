"""
推理脚本 - 单个.set文件的癫痫发作预测和活跃通道识别
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
from model_multitask import create_multitask_model
from utils import load_checkpoint


class EEGInference:
    """EEG推理器"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: 模型检查点路径
            device: 'cuda' or 'cpu'
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        config_path = self.checkpoint_path.parent / 'config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 创建模型
        print(f"加载模型从: {checkpoint_path}")
        print(f"使用设备: {self.device}")
        
        # 需要先推断n_channels和n_samples（从一个样本文件）
        self.window_size = self.config.get('window_size', 6.0)
        self.normalization = self.config.get('normalization', 'window_robust')
        
        self.model = None  # 延迟初始化
        self.extractor = EEGWindowExtractor(window_size=self.window_size)
        
        # 通道名映射
        self.channel_names = None
    
    def _init_model(self, n_channels, n_samples):
        """初始化模型"""
        if self.model is not None:
            return
        
        self.model = create_multitask_model(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=self.config.get('n_classes', 5),
            d_model=self.config.get('d_model', 256),
            n_heads=self.config.get('n_heads', 8),
            n_layers=self.config.get('n_layers', 4),
            dropout=self.config.get('dropout', 0.3)
        )
        
        # 加载权重
        load_checkpoint(self.checkpoint_path, self.model)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"模型已加载: {sum(p.numel() for p in self.model.parameters()):,} 参数")
    
    def _normalize_window(self, window_data):
        """归一化单个窗口"""
        if self.normalization == 'none':
            return window_data
        
        elif self.normalization == 'window_zscore':
            mean = window_data.mean()
            std = window_data.std()
            return (window_data - mean) / (std + 1e-8)
        
        elif self.normalization == 'window_robust':
            median = np.median(window_data)
            mad = np.median(np.abs(window_data - median))
            return (window_data - median) / (mad * 1.4826 + 1e-8)
        
        elif self.normalization == 'channel_zscore':
            normalized = np.zeros_like(window_data)
            for ch in range(window_data.shape[0]):
                mean = window_data[ch].mean()
                std = window_data[ch].std()
                normalized[ch] = (window_data[ch] - mean) / (std + 1e-8)
            return normalized
        
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
    
    def predict_file(self, set_file_path, window_idx=None, return_all_windows=False):
        """
        预测单个.set文件
        
        Args:
            set_file_path: .set文件路径
            window_idx: 指定窗口索引（None表示所有窗口）
            return_all_windows: 是否返回所有窗口的结果
        
        Returns:
            结果字典
        """
        print(f"\n处理文件: {set_file_path}")
        
        # 获取通道名
        self.channel_names = self._get_channel_names(set_file_path)
        print(f"通道数: {len(self.channel_names)}")
        print(f"通道名: {self.channel_names}")
        
        # 提取窗口
        print(f"\n提取{self.window_size}秒窗口...")
        windows, info = self.extractor.extract_windows(set_file_path)
        
        print(f"提取到 {len(windows)} 个窗口")
        print(f"采样率: {info['sfreq']} Hz")
        print(f"窗口形状: {windows.shape}")
        
        if len(windows) == 0:
            print("未提取到有效窗口！")
            return None
        
        # 初始化模型
        n_channels = windows.shape[1]
        n_samples = windows.shape[2]
        self._init_model(n_channels, n_samples)
        
        # 选择窗口
        if window_idx is not None:
            if window_idx >= len(windows):
                print(f"警告: 窗口索引{window_idx}超出范围，使用最后一个窗口")
                window_idx = len(windows) - 1
            selected_windows = [window_idx]
        else:
            selected_windows = range(len(windows)) if return_all_windows else [0]
        
        # 推理
        results = []
        
        print(f"\n开始推理...")
        with torch.no_grad():
            for idx in selected_windows:
                window_data = windows[idx]  # (n_channels, n_samples)
                
                # 归一化
                window_normalized = self._normalize_window(window_data)
                
                # 转为tensor
                window_tensor = torch.FloatTensor(window_normalized).unsqueeze(0)  # (1, n_ch, n_samp)
                window_tensor = window_tensor.to(self.device)
                
                # 推理
                seizure_logits, channel_logits, relation_logits = self.model(window_tensor)
                
                # 解析结果
                seizure_probs = torch.softmax(seizure_logits, dim=1)[0]  # (n_classes,)
                seizure_pred = seizure_logits.argmax(dim=1).item()
                seizure_conf = seizure_probs[seizure_pred].item()
                
                channel_probs = torch.sigmoid(channel_logits)[0]  # (n_channels,)
                channel_preds = (channel_probs > 0.5).cpu().numpy()
                channel_probs = channel_probs.cpu().numpy()
                
                relation_matrix = torch.sigmoid(relation_logits)[0].cpu().numpy()  # (n_ch, n_ch)
                
                # 获取活跃通道
                active_channel_indices = np.where(channel_preds)[0]
                active_channel_names = [self.channel_names[i] for i in active_channel_indices]
                active_channel_probs = channel_probs[active_channel_indices]
                
                result = {
                    'window_idx': idx,
                    'seizure_type': seizure_pred,
                    'seizure_confidence': seizure_conf,
                    'seizure_probs': seizure_probs.cpu().numpy(),
                    'active_channel_indices': active_channel_indices,
                    'active_channel_names': active_channel_names,
                    'active_channel_probs': active_channel_probs,
                    'all_channel_probs': channel_probs,
                    'relation_matrix': relation_matrix
                }
                
                results.append(result)
        
        return {
            'file_path': set_file_path,
            'n_windows': len(windows),
            'sfreq': info['sfreq'],
            'channel_names': self.channel_names,
            'results': results
        }
    
    def print_results(self, prediction_results):
        """打印预测结果"""
        if prediction_results is None:
            return
        
        print(f"\n{'='*70}")
        print(f"预测结果")
        print(f"{'='*70}")
        
        print(f"\n文件: {prediction_results['file_path']}")
        print(f"总窗口数: {prediction_results['n_windows']}")
        print(f"采样率: {prediction_results['sfreq']} Hz")
        
        for result in prediction_results['results']:
            print(f"\n{'-'*70}")
            print(f"窗口 #{result['window_idx']}")
            print(f"{'-'*70}")
            
            # 发作类型
            print(f"\n发作类型预测:")
            print(f"  预测类别: SZ{result['seizure_type']}")
            print(f"  置信度: {result['seizure_confidence']*100:.2f}%")
            
            print(f"\n所有类别概率:")
            for i, prob in enumerate(result['seizure_probs']):
                print(f"  SZ{i}: {prob*100:.2f}%")
            
            # 活跃通道
            print(f"\n活跃通道预测 (共{len(result['active_channel_names'])}个):")
            if len(result['active_channel_names']) > 0:
                for name, prob in zip(result['active_channel_names'], 
                                     result['active_channel_probs']):
                    print(f"  {name}: {prob*100:.2f}%")
            else:
                print(f"  （未检测到活跃通道）")
            
            # Top 5通道（按概率排序）
            print(f"\nTop 5 通道（按激活概率）:")
            top5_indices = np.argsort(result['all_channel_probs'])[::-1][:5]
            for rank, idx in enumerate(top5_indices, 1):
                ch_name = prediction_results['channel_names'][idx]
                prob = result['all_channel_probs'][idx]
                is_active = '✓' if result['all_channel_probs'][idx] > 0.5 else ' '
                print(f"  {rank}. {ch_name:10s} {prob*100:5.2f}% {is_active}")
    
    def visualize_results(self, prediction_results, save_dir='inference_results'):
        """可视化预测结果"""
        if prediction_results is None:
            return
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for result in prediction_results['results']:
            window_idx = result['window_idx']
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 发作类型概率
            ax = axes[0, 0]
            classes = [f"SZ{i}" for i in range(len(result['seizure_probs']))]
            colors = ['green' if i == result['seizure_type'] else 'gray' 
                     for i in range(len(classes))]
            ax.bar(classes, result['seizure_probs'] * 100, color=colors)
            ax.set_title(f'发作类型预测 (窗口 #{window_idx})', fontsize=12, fontweight='bold')
            ax.set_ylabel('概率 (%)')
            ax.set_ylim([0, 100])
            ax.axhline(y=50, color='r', linestyle='--', alpha=0.3)
            
            # 2. 通道激活概率
            ax = axes[0, 1]
            channel_names = prediction_results['channel_names']
            probs = result['all_channel_probs'] * 100
            colors = ['red' if p > 50 else 'blue' for p in probs]
            
            ax.barh(range(len(channel_names)), probs, color=colors)
            ax.set_yticks(range(len(channel_names)))
            ax.set_yticklabels(channel_names, fontsize=8)
            ax.set_xlabel('激活概率 (%)')
            ax.set_title('通道激活概率', fontsize=12, fontweight='bold')
            ax.axvline(x=50, color='r', linestyle='--', alpha=0.3, label='阈值(50%)')
            ax.set_xlim([0, 100])
            ax.legend()
            ax.invert_yaxis()
            
            # 3. 通道关系热图
            ax = axes[1, 0]
            relation = result['relation_matrix']
            im = ax.imshow(relation, cmap='hot', vmin=0, vmax=1)
            ax.set_xticks(range(len(channel_names)))
            ax.set_yticks(range(len(channel_names)))
            ax.set_xticklabels(channel_names, rotation=90, fontsize=7)
            ax.set_yticklabels(channel_names, fontsize=7)
            ax.set_title('通道关系矩阵', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax, label='关联强度')
            
            # 4. 活跃通道列表
            ax = axes[1, 1]
            ax.axis('off')
            
            # 总结信息
            summary_text = f"预测摘要 (窗口 #{window_idx})\n\n"
            summary_text += f"发作类型: SZ{result['seizure_type']}\n"
            summary_text += f"置信度: {result['seizure_confidence']*100:.2f}%\n\n"
            summary_text += f"活跃通道 ({len(result['active_channel_names'])}个):\n"
            
            if len(result['active_channel_names']) > 0:
                for name, prob in zip(result['active_channel_names'], 
                                     result['active_channel_probs']):
                    summary_text += f"  • {name}: {prob*100:.2f}%\n"
            else:
                summary_text += "  （未检测到活跃通道）\n"
            
            summary_text += f"\nTop 5通道:\n"
            top5_indices = np.argsort(result['all_channel_probs'])[::-1][:5]
            for rank, idx in enumerate(top5_indices, 1):
                ch_name = channel_names[idx]
                prob = result['all_channel_probs'][idx]
                summary_text += f"  {rank}. {ch_name}: {prob*100:.2f}%\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   family='monospace')
            
            plt.tight_layout()
            
            # 保存
            save_path = save_dir / f'window_{window_idx}_prediction.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"可视化已保存: {save_path}")
    
    def predict(self, set_file_path, window_idx=None, visualize=True, 
                save_results=True, save_dir='inference_results'):
        """
        预测并显示结果
        
        Args:
            set_file_path: .set文件路径
            window_idx: 指定窗口（None=第一个窗口）
            visualize: 是否可视化
            save_results: 是否保存结果
            save_dir: 保存目录
        
        Returns:
            预测结果字典
        """
        # 执行预测
        results = self.predict_file(
            set_file_path, 
            window_idx=window_idx, 
            return_all_windows=(window_idx is None)
        )
        
        if results is None:
            return None
        
        # 打印结果
        self.print_results(results)
        
        # 可视化
        if visualize:
            self.visualize_results(results, save_dir)
        
        # 保存JSON结果
        if save_results:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 转换numpy为list以便JSON序列化
            results_serializable = {
                'file_path': str(results['file_path']),
                'n_windows': results['n_windows'],
                'sfreq': results['sfreq'],
                'channel_names': results['channel_names'],
                'results': []
            }
            
            for r in results['results']:
                results_serializable['results'].append({
                    'window_idx': r['window_idx'],
                    'seizure_type': int(r['seizure_type']),
                    'seizure_confidence': float(r['seizure_confidence']),
                    'seizure_probs': r['seizure_probs'].tolist(),
                    'active_channel_names': r['active_channel_names'],
                    'active_channel_probs': r['active_channel_probs'].tolist(),
                    'all_channel_probs': r['all_channel_probs'].tolist()
                })
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_path = save_dir / f'prediction_{timestamp}.json'
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_serializable, f, indent=2, ensure_ascii=False)
            
            print(f"\n结果已保存到: {json_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='EEG癫痫发作推理 - 活跃通道识别')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径 (best_model.pth)')
    parser.add_argument('--set_file', type=str, required=True,
                        help='.set文件路径')
    parser.add_argument('--window_idx', type=int, default=None,
                        help='指定窗口索引（默认第一个窗口）')
    parser.add_argument('--all_windows', action='store_true',
                        help='处理所有窗口')
    parser.add_argument('--save_dir', type=str, default='inference_results',
                        help='结果保存目录')
    parser.add_argument('--no_visualize', action='store_true',
                        help='不生成可视化图片')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.checkpoint).exists():
        print(f"错误: 检查点文件不存在: {args.checkpoint}")
        return
    
    if not Path(args.set_file).exists():
        print(f"错误: .set文件不存在: {args.set_file}")
        return
    
    # 创建推理器
    inference = EEGInference(args.checkpoint, device=args.device)
    
    # 执行推理
    results = inference.predict(
        set_file_path=args.set_file,
        window_idx=args.window_idx,
        visualize=not args.no_visualize,
        save_results=True,
        save_dir=args.save_dir
    )
    
    print(f"\n{'='*70}")
    print("推理完成！")
    print(f"{'='*70}")
    
    if results:
        print(f"\n查看结果:")
        print(f"  JSON: {args.save_dir}/prediction_*.json")
        if not args.no_visualize:
            print(f"  图片: {args.save_dir}/window_*_prediction.png")


if __name__ == "__main__":
    main()

