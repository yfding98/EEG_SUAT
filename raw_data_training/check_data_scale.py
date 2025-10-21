"""
检查数据尺度问题
验证loss从706→288→1.3的剧烈波动可能是数据尺度不一致
"""

import numpy as np
from tqdm import tqdm
from dataset import RawEEGDataset


def check_data_scale():
    """检查数据尺度"""
    
    print("="*60)
    print("数据尺度检查")
    print("="*60)
    
    data_root = r'E:\DataSet\EEG\EEG dataset_SUAT_processed'
    labels_csv = r'E:\output\connectivity_features\labels.csv'
    
    print("\n加载数据集...")
    dataset = RawEEGDataset(
        data_root=data_root,
        labels_csv=labels_csv,
        window_size=6.0,
        use_cache=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    if len(dataset) == 0:
        print("数据集为空！")
        return
    
    # 收集统计信息
    mins = []
    maxs = []
    means = []
    stds = []
    
    print("\n分析每个窗口...")
    for i in tqdm(range(min(len(dataset), 1000))):  # 采样前1000个
        sample = dataset[i]
        data = sample['data'].numpy()
        
        mins.append(data.min())
        maxs.append(data.max())
        means.append(data.mean())
        stds.append(data.std())
    
    mins = np.array(mins)
    maxs = np.array(maxs)
    means = np.array(means)
    stds = np.array(stds)
    
    # 打印统计
    print(f"\n{'='*60}")
    print("统计结果")
    print(f"{'='*60}")
    
    print(f"\n最小值:")
    print(f"  全局最小: {mins.min():.6f}")
    print(f"  平均最小: {mins.mean():.6f}")
    print(f"  中位数: {np.median(mins):.6f}")
    print(f"  标准差: {mins.std():.6f}")
    
    print(f"\n最大值:")
    print(f"  全局最大: {maxs.max():.6f}")
    print(f"  平均最大: {maxs.mean():.6f}")
    print(f"  中位数: {np.median(maxs):.6f}")
    print(f"  标准差: {maxs.std():.6f}")
    
    print(f"\n均值:")
    print(f"  全局均值范围: [{means.min():.6f}, {means.max():.6f}]")
    print(f"  平均: {means.mean():.6f}")
    print(f"  中位数: {np.median(means):.6f}")
    print(f"  标准差: {means.std():.6f}")
    
    print(f"\n标准差:")
    print(f"  标准差范围: [{stds.min():.6f}, {stds.max():.6f}]")
    print(f"  平均: {stds.mean():.6f}")
    print(f"  中位数: {np.median(stds):.6f}")
    
    # 检测异常
    print(f"\n{'='*60}")
    print("异常检测")
    print(f"{'='*60}")
    
    # 极端值
    extreme_samples = []
    for i, (min_val, max_val) in enumerate(zip(mins, maxs)):
        if abs(min_val) > 500 or abs(max_val) > 500:
            extreme_samples.append(i)
    
    print(f"\n极端值样本 (|值| > 500):")
    print(f"  数量: {len(extreme_samples)} ({len(extreme_samples)/len(mins)*100:.1f}%)")
    
    # 尺度差异大
    scale_range = maxs - mins
    scale_mean = scale_range.mean()
    scale_std = scale_range.std()
    
    large_scale = []
    small_scale = []
    for i, scale in enumerate(scale_range):
        if scale > scale_mean + 3 * scale_std:
            large_scale.append(i)
        elif scale < scale_mean - 3 * scale_std:
            small_scale.append(i)
    
    print(f"\n尺度异常样本:")
    print(f"  过大尺度: {len(large_scale)} ({len(large_scale)/len(mins)*100:.1f}%)")
    print(f"  过小尺度: {len(small_scale)} ({len(small_scale)/len(mins)*100:.1f}%)")
    
    # 方差异常
    std_mean = stds.mean()
    std_std = stds.std()
    
    low_var = []
    high_var = []
    for i, std in enumerate(stds):
        if std < std_mean - 3 * std_std:
            low_var.append(i)
        elif std > std_mean + 3 * std_std:
            high_var.append(i)
    
    print(f"\n方差异常样本:")
    print(f"  低方差: {len(low_var)} ({len(low_var)/len(stds)*100:.1f}%)")
    print(f"  高方差: {len(high_var)} ({len(high_var)/len(stds)*100:.1f}%)")
    
    # 诊断
    print(f"\n{'='*60}")
    print("诊断和建议")
    print(f"{'='*60}")
    
    need_normalization = False
    
    # 检查是否需要归一化
    if stds.std() / stds.mean() > 0.5:
        print("\n⚠️  标准差变化大 → 强烈建议归一化")
        need_normalization = True
    
    if maxs.max() - mins.min() > 1000:
        print("⚠️  数据范围过大 → 建议归一化")
        need_normalization = True
    
    if len(extreme_samples) > len(mins) * 0.05:
        print(f"⚠️  极端值样本过多 ({len(extreme_samples)}) → 建议过滤或裁剪")
    
    if len(low_var) > len(stds) * 0.05:
        print(f"⚠️  低方差样本过多 ({len(low_var)}) → 可能是平坦信号，建议过滤")
    
    if len(high_var) > len(stds) * 0.05:
        print(f"⚠️  高方差样本过多 ({len(high_var)}) → 可能是噪声，建议过滤")
    
    if need_normalization:
        print(f"\n{'='*60}")
        print("归一化方案")
        print(f"{'='*60}")
        
        print("\n方案1: Z-score标准化（推荐）")
        print("  data = (data - mean) / (std + 1e-8)")
        print("  适用于大多数情况")
        
        print("\n方案2: Robust归一化")
        print("  data = (data - median) / (mad + 1e-8)")
        print("  对异常值更鲁棒")
        
        print("\n方案3: Min-Max归一化")
        print("  data = (data - min) / (max - min + 1e-8)")
        print("  适用于范围一致的情况")
        
        print("\n实现位置: dataset.py 的 __getitem__ 方法")
    
    print(f"\n{'='*60}")
    
    return {
        'need_normalization': need_normalization,
        'extreme_samples': len(extreme_samples),
        'low_var_samples': len(low_var),
        'high_var_samples': len(high_var)
    }


if __name__ == "__main__":
    result = check_data_scale()
    
    print("\n完成!")
    
    if result and result['need_normalization']:
        print("\n⚠️  强烈建议修改 dataset.py 添加数据归一化！")
        print("这可能是验证loss波动大的主要原因。")

