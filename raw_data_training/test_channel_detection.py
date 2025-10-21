"""
快速测试活跃通道检测
"""

from pathlib import Path
from inference_channel_detection import ChannelDetectionInference


def find_latest_checkpoint(checkpoints_dir='checkpoints_channel_detection'):
    """查找最新的检查点"""
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"检查点目录不存在: {checkpoints_dir}")
        return None
    
    checkpoints = list(checkpoints_path.glob('*/best_model.pth'))
    
    if not checkpoints:
        print(f"未找到检查点")
        print(f"请先训练模型: python train_channel_detection.py")
        return None
    
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"找到最新检查点: {latest}")
    
    return latest


def test_detection():
    """测试活跃通道检测"""
    
    print("="*70)
    print("活跃通道检测测试")
    print("="*70)
    
    # 查找模型
    checkpoint = find_latest_checkpoint()
    if checkpoint is None:
        return
    
    # 测试文件
    test_file = r'E:\DataSet\EEG\EEG dataset_SUAT_processed\头皮数据-6例\曾静君\SZ1_preICA_reject_1_postICA_merged_F7_Fp1_Sph_L.set'
    
    if not Path(test_file).exists():
        print(f"\n测试文件不存在: {test_file}")
        print("请修改test_channel_detection.py中的test_file路径")
        return
    
    print(f"\n测试文件: {test_file}")
    print(f"根据文件名，真实活跃通道应该是: [F7, Fp1, Sph_L]")
    
    # 创建推理器
    inference = ChannelDetectionInference(checkpoint)
    
    # 预测
    result = inference.predict_and_save(
        set_file_path=test_file,
        window_idx=0,
        save_dir='test_detection_results'
    )
    
    if result:
        # 验证结果
        print(f"\n{'='*70}")
        print("验证预测结果")
        print(f"{'='*70}")
        
        true_active = ['F7', 'Fp1', 'Sph_L']
        pred_active = result['active_channel_names']
        
        print(f"\n真实活跃通道: {true_active}")
        print(f"预测活跃通道: {pred_active}")
        
        # 计算匹配度
        matched = set(true_active) & set(pred_active)
        missed = set(true_active) - set(pred_active)
        extra = set(pred_active) - set(true_active)
        
        print(f"\n匹配分析:")
        print(f"  ✓ 正确识别: {list(matched)}")
        print(f"  ✗ 遗漏: {list(missed) if missed else '无'}")
        print(f"  ! 多余: {list(extra) if extra else '无'}")
        
        # 计算指标
        tp = len(matched)
        fp = len(extra)
        fn = len(missed)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n性能指标:")
        print(f"  精确率: {precision*100:.2f}%")
        print(f"  召回率: {recall*100:.2f}%")
        print(f"  F1分数: {f1*100:.2f}%")
        
        if f1 >= 0.8:
            print(f"\n  ✓✓✓ 检测效果很好！")
        elif f1 >= 0.6:
            print(f"\n  ✓✓ 检测效果良好")
        elif f1 >= 0.4:
            print(f"\n  ✓ 检测效果一般，可能需要调参")
        else:
            print(f"\n  ✗ 检测效果较差，需要检查模型训练")
        
        print(f"\n结果已保存到: test_detection_results/")


def test_multiple_windows():
    """测试多个窗口的一致性"""
    
    print("\n" + "="*70)
    print("多窗口一致性测试")
    print("="*70)
    
    checkpoint = find_latest_checkpoint()
    if checkpoint is None:
        return
    
    test_file = r'E:\DataSet\EEG\EEG dataset_SUAT_processed\头皮数据-6例\曾静君\SZ1_preICA_reject_1_postICA_merged_F7_Fp1_Sph_L.set'
    
    if not Path(test_file).exists():
        return
    
    inference = ChannelDetectionInference(checkpoint)
    
    # 预测前5个窗口
    print(f"\n预测前5个窗口...")
    
    all_predictions = []
    
    for window_idx in range(5):
        result = inference.predict_window(test_file, window_idx)
        
        if result:
            all_predictions.append(result['active_channel_names'])
            
            print(f"\n窗口{window_idx}: {result['active_channel_names']}")
    
    # 统计一致性
    from collections import Counter
    
    all_channels = [ch for pred in all_predictions for ch in pred]
    channel_freq = Counter(all_channels)
    
    print(f"\n{'='*70}")
    print("通道出现频率（5个窗口中）")
    print(f"{'='*70}")
    
    for ch, count in channel_freq.most_common():
        print(f"  {ch:10s} {count}/5 窗口 ({count/5*100:.0f}%)")
    
    print(f"\n一致性分析:")
    consistent_channels = [ch for ch, count in channel_freq.items() if count >= 4]
    print(f"  高度一致(≥4/5): {consistent_channels}")
    
    true_active = ['F7', 'Fp1', 'Sph_L']
    matched = set(true_active) & set(consistent_channels)
    
    if len(matched) >= 2:
        print(f"  ✓ 预测一致且与真实标注匹配")
    else:
        print(f"  ! 预测可能不稳定或标注有问题")


if __name__ == "__main__":
    import sys
    
    print("\n选择测试模式:")
    print("  1. 单窗口检测（推荐）")
    print("  2. 多窗口一致性测试")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\n请选择 (1/2): ").strip() or '1'
    
    if choice == '1':
        test_detection()
    elif choice == '2':
        test_multiple_windows()
    else:
        print("无效选择")

