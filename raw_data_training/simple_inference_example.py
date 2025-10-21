"""
简单的推理示例
演示如何使用训练好的模型预测活跃通道
"""

from pathlib import Path
from inference import EEGInference


def example_1_basic():
    """示例1: 基础推理"""
    
    print("="*70)
    print("示例1: 基础推理 - 预测单个窗口")
    print("="*70)
    
    # 1. 设置路径
    checkpoint_path = 'checkpoints_multitask/multitask_20241018_143000/best_model.pth'
    set_file = r'E:\DataSet\EEG\EEG dataset_SUAT_processed\头皮数据-6例\曾静君\SZ1_preICA_reject_1_postICA_merged_F7_Fp1_Sph_L.set'
    
    # 检查文件是否存在
    if not Path(checkpoint_path).exists():
        print(f"模型不存在，请先训练: python train_multitask.py")
        # 查找最新的
        checkpoints = list(Path('checkpoints_multitask').glob('*/best_model.pth'))
        if checkpoints:
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"使用最新模型: {checkpoint_path}")
        else:
            return
    
    if not Path(set_file).exists():
        print(f"\n文件不存在: {set_file}")
        print("请修改脚本中的set_file路径")
        return
    
    # 2. 创建推理器
    print(f"\n创建推理器...")
    inference = EEGInference(checkpoint_path, device='cuda')
    
    # 3. 执行推理
    print(f"\n执行推理...")
    results = inference.predict(
        set_file_path=set_file,
        window_idx=0,  # 第一个窗口
        visualize=True,
        save_results=True,
        save_dir='example_results'
    )
    
    # 4. 访问结果
    if results and len(results['results']) > 0:
        r = results['results'][0]
        
        print(f"\n{'='*70}")
        print("提取关键信息")
        print(f"{'='*70}")
        
        print(f"\n发作类型: SZ{r['seizure_type']}")
        print(f"置信度: {r['seizure_confidence']*100:.2f}%")
        
        print(f"\n预测的活跃通道:")
        for ch_name in r['active_channel_names']:
            print(f"  ✓ {ch_name}")
        
        print(f"\n真实的活跃通道（从文件名）:")
        print(f"  [F7, Fp1, Sph_L]")
        
        # 对比
        true_channels = ['F7', 'Fp1', 'Sph_L']
        pred_channels = r['active_channel_names']
        
        matched = set(true_channels) & set(pred_channels)
        missed = set(true_channels) - set(pred_channels)
        extra = set(pred_channels) - set(true_channels)
        
        print(f"\n匹配分析:")
        print(f"  匹配: {list(matched)}")
        print(f"  遗漏: {list(missed) if missed else '无'}")
        print(f"  多余: {list(extra) if extra else '无'}")
        
        if len(matched) == len(true_channels) and len(extra) == 0:
            print(f"\n  ✓✓✓ 完美匹配！")
        elif len(matched) >= len(true_channels) * 0.8:
            print(f"\n  ✓ 基本正确")
        else:
            print(f"\n  ✗ 需要改进")


def example_2_multiple_windows():
    """示例2: 预测多个窗口"""
    
    print("\n" + "="*70)
    print("示例2: 多窗口推理 - 分析一致性")
    print("="*70)
    
    # 查找模型和文件（同示例1）
    checkpoints = list(Path('checkpoints_multitask').glob('*/best_model.pth'))
    if not checkpoints:
        print("未找到模型")
        return
    
    checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    set_file = r'E:\DataSet\EEG\EEG dataset_SUAT_processed\头皮数据-6例\曾静君\SZ1_preICA_reject_1_postICA_merged_F7_Fp1_Sph_L.set'
    
    if not Path(set_file).exists():
        print(f"文件不存在")
        return
    
    # 创建推理器
    inference = EEGInference(checkpoint_path)
    
    # 预测前5个窗口
    print(f"\n预测前5个窗口...")
    
    all_active_channels = []
    all_seizure_types = []
    
    for window_idx in range(5):
        results = inference.predict(
            set_file_path=set_file,
            window_idx=window_idx,
            visualize=False,
            save_results=False
        )
        
        if results and len(results['results']) > 0:
            r = results['results'][0]
            all_active_channels.append(r['active_channel_names'])
            all_seizure_types.append(r['seizure_type'])
            
            print(f"\n窗口{window_idx}:")
            print(f"  发作类型: SZ{r['seizure_type']} ({r['seizure_confidence']*100:.1f}%)")
            print(f"  活跃通道: {r['active_channel_names']}")
    
    # 分析一致性
    print(f"\n{'='*70}")
    print("一致性分析")
    print(f"{'='*70}")
    
    # 发作类型一致性
    from collections import Counter
    type_counter = Counter(all_seizure_types)
    print(f"\n发作类型分布:")
    for sz_type, count in type_counter.most_common():
        print(f"  SZ{sz_type}: {count}/5 窗口")
    
    # 通道一致性
    all_channels_flat = [ch for channels in all_active_channels for ch in channels]
    channel_counter = Counter(all_channels_flat)
    
    print(f"\n活跃通道频率:")
    for ch, count in channel_counter.most_common():
        print(f"  {ch}: {count}/5 窗口 ({count/5*100:.0f}%)")


def example_3_channel_analysis():
    """示例3: 详细通道分析"""
    
    print("\n" + "="*70)
    print("示例3: 详细通道分析")
    print("="*70)
    
    checkpoints = list(Path('checkpoints_multitask').glob('*/best_model.pth'))
    if not checkpoints:
        print("未找到模型")
        return
    
    checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    set_file = r'E:\DataSet\EEG\EEG dataset_SUAT_processed\头皮数据-6例\曾静君\SZ1_preICA_reject_1_postICA_merged_F7_Fp1_Sph_L.set'
    
    if not Path(set_file).exists():
        return
    
    inference = EEGInference(checkpoint_path)
    
    results = inference.predict(
        set_file_path=set_file,
        window_idx=0,
        visualize=False,
        save_results=False
    )
    
    if results and len(results['results']) > 0:
        r = results['results'][0]
        channel_names = results['channel_names']
        
        print(f"\n所有21个通道的激活概率:")
        print(f"{'通道':10s} {'概率':>8s} {'状态':>6s}")
        print("-"*30)
        
        for ch_name, prob in zip(channel_names, r['all_channel_probs']):
            status = '活跃' if prob > 0.5 else ''
            print(f"{ch_name:10s} {prob*100:7.2f}% {status:>6s}")
        
        # 分析通道关系
        print(f"\n活跃通道间的关系强度:")
        active_indices = [i for i, p in enumerate(r['all_channel_probs']) if p > 0.5]
        
        if len(active_indices) >= 2:
            relation = r['relation_matrix']
            for i in range(len(active_indices)):
                for j in range(i+1, len(active_indices)):
                    idx_i = active_indices[i]
                    idx_j = active_indices[j]
                    ch_i = channel_names[idx_i]
                    ch_j = channel_names[idx_j]
                    strength = relation[idx_i, idx_j]
                    print(f"  {ch_i} ↔ {ch_j}: {strength*100:.2f}%")


def main():
    """运行所有示例"""
    
    print("\n" + "="*70)
    print("EEG推理示例脚本")
    print("="*70)
    
    print("\n提示: 请先确保已训练多任务模型")
    print("      python train_multitask.py")
    
    # 示例1
    try:
        example_1_basic()
    except Exception as e:
        print(f"\n示例1出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 示例2
    try:
        example_2_multiple_windows()
    except Exception as e:
        print(f"\n示例2出错: {e}")
    
    # 示例3
    try:
        example_3_channel_analysis()
    except Exception as e:
        print(f"\n示例3出错: {e}")
    
    print(f"\n{'='*70}")
    print("所有示例完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

