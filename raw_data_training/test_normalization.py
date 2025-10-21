"""
测试归一化效果
"""

import numpy as np
import torch
from dataset import RawEEGDataset


def test_normalization_methods():
    """测试不同的归一化方法"""
    
    data_root = r'E:\DataSet\EEG\EEG dataset_SUAT_processed'
    labels_csv = r'E:\output\connectivity_features\labels.csv'
    
    methods = ['none', 'window_zscore', 'window_robust', 'channel_zscore', 'channel_robust']
    
    print("="*70)
    print("归一化方法对比测试")
    print("="*70)
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"方法: {method}")
        print(f"{'='*70}")
        
        # 创建数据集
        dataset = RawEEGDataset(
            data_root=data_root,
            labels_csv=labels_csv,
            window_size=6.0,
            use_cache=True,
            normalization=method
        )
        
        if len(dataset) == 0:
            print("数据集为空！")
            continue
        
        # 采样100个窗口
        n_samples = min(100, len(dataset))
        
        all_means = []
        all_stds = []
        all_mins = []
        all_maxs = []
        
        for i in range(n_samples):
            sample = dataset[i]
            data = sample['data'].numpy()
            
            all_means.append(data.mean())
            all_stds.append(data.std())
            all_mins.append(data.min())
            all_maxs.append(data.max())
        
        all_means = np.array(all_means)
        all_stds = np.array(all_stds)
        all_mins = np.array(all_mins)
        all_maxs = np.array(all_maxs)
        
        # 打印统计
        print(f"\n统计 (基于{n_samples}个窗口):")
        print(f"  均值的均值: {all_means.mean():.6f}")
        print(f"  均值的标准差: {all_means.std():.6f}")
        print(f"  标准差的均值: {all_stds.mean():.6f}")
        print(f"  标准差的标准差: {all_stds.std():.6f}")
        print(f"  全局最小值: {all_mins.min():.6f}")
        print(f"  全局最大值: {all_maxs.max():.6f}")
        
        # 评估
        print(f"\n评估:")
        
        # 理想情况：均值接近0，标准差接近1
        if method != 'none':
            mean_centered = abs(all_means.mean()) < 0.1
            std_normalized = abs(all_stds.mean() - 1.0) < 0.3
            
            print(f"  均值居中: {'✓' if mean_centered else '✗'} "
                  f"(理想: 接近0, 实际: {all_means.mean():.4f})")
            print(f"  方差归一化: {'✓' if std_normalized else '✗'} "
                  f"(理想: 接近1, 实际: {all_stds.mean():.4f})")
            
            # 检查稳定性
            mean_stability = all_means.std() < 0.5
            std_stability = all_stds.std() < 0.5
            
            print(f"  均值稳定性: {'✓' if mean_stability else '✗'} "
                  f"(标准差: {all_means.std():.4f})")
            print(f"  方差稳定性: {'✓' if std_stability else '✗'} "
                  f"(标准差: {all_stds.std():.4f})")
            
            # 综合评分
            score = sum([mean_centered, std_normalized, mean_stability, std_stability])
            print(f"\n  综合评分: {score}/4 {'★' * score}")
        else:
            print(f"  原始数据（未归一化）")
    
    print(f"\n{'='*70}")
    print("推荐:")
    print("  - 对于被试者差异大: window_robust 或 channel_robust")
    print("  - 对于数据干净: window_zscore")
    print("  - 通道差异大: channel_zscore 或 channel_robust")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_normalization_methods()

