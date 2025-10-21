"""
数据质量分析和badcase检测
分析每个窗口的质量，标记问题窗口
"""

import numpy as np
import torch
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import warnings

from data_loader import EEGWindowExtractor
from dataset import RawEEGDataset

warnings.filterwarnings('ignore')


class WindowQualityAnalyzer:
    """窗口质量分析器"""
    
    def __init__(self):
        self.issues = defaultdict(list)
        
    def check_nan_inf(self, window_data, file_path, window_idx):
        """检查NaN和Inf"""
        has_nan = np.isnan(window_data).any()
        has_inf = np.isinf(window_data).any()
        
        if has_nan or has_inf:
            self.issues['nan_inf'].append({
                'file': file_path,
                'window_idx': window_idx,
                'has_nan': bool(has_nan),
                'has_inf': bool(has_inf)
            })
            return False
        return True
    
    def check_extreme_values(self, window_data, file_path, window_idx, 
                           threshold_low=-500, threshold_high=500):
        """检查极端值"""
        min_val = window_data.min()
        max_val = window_data.max()
        
        if min_val < threshold_low or max_val > threshold_high:
            self.issues['extreme_values'].append({
                'file': file_path,
                'window_idx': window_idx,
                'min': float(min_val),
                'max': float(max_val)
            })
            return False
        return True
    
    def check_zero_variance(self, window_data, file_path, window_idx, min_std=0.1):
        """检查方差过小（可能是平坦信号）"""
        channel_stds = window_data.std(axis=1)
        
        if (channel_stds < min_std).any():
            self.issues['low_variance'].append({
                'file': file_path,
                'window_idx': window_idx,
                'channel_stds': channel_stds.tolist()
            })
            return False
        return True
    
    def check_high_variance(self, window_data, file_path, window_idx, max_std=200):
        """检查方差过大（可能是噪声）"""
        channel_stds = window_data.std(axis=1)
        
        if (channel_stds > max_std).any():
            self.issues['high_variance'].append({
                'file': file_path,
                'window_idx': window_idx,
                'channel_stds': channel_stds.tolist()
            })
            return False
        return True
    
    def check_saturation(self, window_data, file_path, window_idx, threshold=0.3):
        """检查信号饱和（某个值重复出现过多）"""
        for ch_idx, channel_data in enumerate(window_data):
            unique_vals, counts = np.unique(channel_data, return_counts=True)
            max_count_ratio = counts.max() / len(channel_data)
            
            if max_count_ratio > threshold:
                self.issues['saturation'].append({
                    'file': file_path,
                    'window_idx': window_idx,
                    'channel': ch_idx,
                    'max_ratio': float(max_count_ratio)
                })
                return False
        return True
    
    def check_trend(self, window_data, file_path, window_idx, max_slope=0.5):
        """检查是否有明显趋势（线性漂移）"""
        for ch_idx, channel_data in enumerate(window_data):
            # 简单线性回归
            x = np.arange(len(channel_data))
            slope = np.polyfit(x, channel_data, 1)[0]
            
            if abs(slope) > max_slope:
                self.issues['trend'].append({
                    'file': file_path,
                    'window_idx': window_idx,
                    'channel': ch_idx,
                    'slope': float(slope)
                })
                return False
        return True
    
    def analyze_window(self, window_data, file_path, window_idx):
        """综合分析一个窗口"""
        checks = [
            self.check_nan_inf(window_data, file_path, window_idx),
            self.check_extreme_values(window_data, file_path, window_idx),
            self.check_zero_variance(window_data, file_path, window_idx),
            self.check_high_variance(window_data, file_path, window_idx),
            self.check_saturation(window_data, file_path, window_idx),
            self.check_trend(window_data, file_path, window_idx)
        ]
        
        return all(checks)  # 所有检查都通过才返回True


def analyze_dataset_quality(data_root, labels_csv, window_size=6.0, cache_dir='cache'):
    """分析整个数据集的质量"""
    
    print("="*60)
    print("数据质量分析")
    print("="*60)
    
    # 创建分析器
    analyzer = WindowQualityAnalyzer()
    
    # 加载数据集（只是为了获取文件列表）
    labels_df = pd.read_csv(labels_csv)
    extractor = EEGWindowExtractor(window_size=window_size)
    data_root = Path(data_root)
    
    good_windows = []
    bad_windows = []
    
    print(f"\n分析 {len(labels_df)} 个文件...")
    
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        file_path = row['data_file_path']
        full_path = data_root / file_path
        
        if not full_path.exists():
            print(f"文件不存在: {full_path}")
            continue
        
        try:
            # 提取窗口
            windows, info = extractor.extract_windows(str(full_path))
            
            if len(windows) == 0:
                continue
            
            # 分析每个窗口
            for window_idx in range(len(windows)):
                window_data = windows[window_idx]
                
                is_good = analyzer.analyze_window(window_data, file_path, window_idx)
                
                if is_good:
                    good_windows.append({
                        'file_path': file_path,
                        'window_idx': window_idx
                    })
                else:
                    bad_windows.append({
                        'file_path': file_path,
                        'window_idx': window_idx
                    })
        
        except Exception as e:
            print(f"\n处理文件出错 {file_path}: {e}")
            continue
    
    # 统计结果
    total_windows = len(good_windows) + len(bad_windows)
    
    print(f"\n{'='*60}")
    print(f"分析结果")
    print(f"{'='*60}")
    print(f"总窗口数: {total_windows}")
    print(f"好窗口数: {len(good_windows)} ({len(good_windows)/total_windows*100:.1f}%)")
    print(f"坏窗口数: {len(bad_windows)} ({len(bad_windows)/total_windows*100:.1f}%)")
    
    print(f"\n问题类型统计:")
    for issue_type, issues in analyzer.issues.items():
        print(f"  {issue_type}: {len(issues)} 个窗口")
    
    return good_windows, bad_windows, analyzer.issues


def analyze_prediction_difficulty(model, dataloader, device, save_path='difficult_samples.json'):
    """分析预测难度，找出难以学习的样本"""
    
    print("\n分析预测难度...")
    
    model.eval()
    
    window_losses = []
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = batch['data'].to(device)
            target = batch['label'].to(device)
            file_paths = batch['file_path']
            window_indices = batch['window_idx'].cpu().numpy()
            
            output = model(data)
            losses = criterion(output, target)
            
            for i, loss in enumerate(losses):
                window_losses.append({
                    'file_path': file_paths[i],
                    'window_idx': int(window_indices[i]),
                    'loss': float(loss.cpu()),
                    'predicted': int(output[i].argmax().cpu()),
                    'actual': int(target[i].cpu())
                })
    
    # 按loss排序
    window_losses.sort(key=lambda x: x['loss'], reverse=True)
    
    # 找出难样本（loss > 某个阈值）
    loss_values = [x['loss'] for x in window_losses]
    loss_mean = np.mean(loss_values)
    loss_std = np.std(loss_values)
    threshold = loss_mean + 2 * loss_std
    
    difficult_samples = [x for x in window_losses if x['loss'] > threshold]
    
    print(f"\n难样本统计:")
    print(f"  Loss均值: {loss_mean:.4f}")
    print(f"  Loss标准差: {loss_std:.4f}")
    print(f"  阈值 (μ + 2σ): {threshold:.4f}")
    print(f"  难样本数: {len(difficult_samples)}")
    
    # 保存
    with open(save_path, 'w') as f:
        json.dump({
            'all_samples': window_losses,
            'difficult_samples': difficult_samples,
            'statistics': {
                'loss_mean': loss_mean,
                'loss_std': loss_std,
                'threshold': threshold
            }
        }, f, indent=2)
    
    return difficult_samples


def create_filtered_dataset_info(good_windows, bad_windows_quality, 
                                 difficult_samples=None, save_path='filtered_dataset.json'):
    """创建过滤后的数据集信息"""
    
    # 转换为集合以便快速查找
    bad_set = set((w['file_path'], w['window_idx']) for w in bad_windows_quality)
    
    if difficult_samples:
        for sample in difficult_samples:
            bad_set.add((sample['file_path'], sample['window_idx']))
    
    # 过滤好窗口
    filtered_windows = [
        w for w in good_windows 
        if (w['file_path'], w['window_idx']) not in bad_set
    ]
    
    print(f"\n过滤后数据集:")
    print(f"  原始窗口数: {len(good_windows)}")
    print(f"  质量问题窗口: {len(bad_windows_quality)}")
    if difficult_samples:
        print(f"  难样本窗口: {len(difficult_samples)}")
    print(f"  过滤后窗口数: {len(filtered_windows)}")
    
    # 按文件统计
    file_stats = defaultdict(lambda: {'good': 0, 'bad': 0})
    
    for w in good_windows:
        if (w['file_path'], w['window_idx']) in bad_set:
            file_stats[w['file_path']]['bad'] += 1
        else:
            file_stats[w['file_path']]['good'] += 1
    
    # 保存
    result = {
        'filtered_windows': filtered_windows,
        'bad_windows': list(bad_set),
        'statistics': {
            'total_good': len(filtered_windows),
            'total_bad': len(bad_set),
            'files': {
                file: stats for file, stats in file_stats.items()
            }
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n过滤信息已保存到: {save_path}")
    
    return filtered_windows, bad_set


def main():
    """主函数"""
    
    data_root = r'E:\DataSet\EEG\EEG dataset_SUAT_processed'
    labels_csv = r'E:\output\connectivity_features\labels.csv'
    
    # 第一步：质量分析
    print("步骤1: 数据质量分析")
    good_windows, bad_windows_quality, issues = analyze_dataset_quality(
        data_root, labels_csv
    )
    
    # 保存质量分析结果
    with open('quality_issues.json', 'w') as f:
        json.dump({
            'good_windows': good_windows,
            'bad_windows': bad_windows_quality,
            'issues': dict(issues)
        }, f, indent=2)
    
    print("\n质量分析结果已保存到: quality_issues.json")
    
    # 第二步：训练初步模型分析难样本（可选，如果有已训练的模型）
    checkpoint_path = Path('checkpoints')
    latest_checkpoint = None
    
    # 查找最新的checkpoint
    if checkpoint_path.exists():
        checkpoints = list(checkpoint_path.glob('*/best_model.pth'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    difficult_samples = None
    if latest_checkpoint:
        print(f"\n步骤2: 使用模型分析难样本")
        print(f"加载模型: {latest_checkpoint}")
        
        try:
            from dataset import RawEEGDataset
            from model import create_model
            from torch.utils.data import DataLoader
            
            # 创建数据集
            dataset = RawEEGDataset(
                data_root=data_root,
                labels_csv=labels_csv,
                window_size=6.0,
                use_cache=True
            )
            
            dataloader = DataLoader(
                dataset, batch_size=32, shuffle=False, num_workers=0
            )
            
            # 获取数据形状
            sample = dataset[0]
            n_channels = sample['data'].shape[0]
            n_samples = sample['data'].shape[1]
            
            # 加载模型
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = create_model(
                model_type='lightweight',
                n_channels=n_channels,
                n_samples=n_samples,
                n_classes=5
            )
            
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            # 分析难样本
            difficult_samples = analyze_prediction_difficulty(
                model, dataloader, device
            )
            
        except Exception as e:
            print(f"分析难样本时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n步骤2: 跳过（未找到已训练模型）")
    
    # 第三步：创建过滤后的数据集
    print(f"\n步骤3: 创建过滤后的数据集")
    filtered_windows, bad_set = create_filtered_dataset_info(
        good_windows, bad_windows_quality, difficult_samples
    )
    
    # 生成详细报告
    print(f"\n{'='*60}")
    print("详细报告")
    print(f"{'='*60}")
    
    # 按文件统计问题
    file_issues = defaultdict(lambda: defaultdict(int))
    
    for issue_type, issue_list in issues.items():
        for issue in issue_list:
            file_issues[issue['file']][issue_type] += 1
    
    print("\n每个文件的问题统计:")
    for file_path, file_issue_counts in sorted(file_issues.items()):
        print(f"\n{file_path}:")
        for issue_type, count in file_issue_counts.items():
            print(f"  {issue_type}: {count}")
    
    print(f"\n{'='*60}")
    print("完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

