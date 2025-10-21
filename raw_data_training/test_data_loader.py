"""
测试数据加载器和数据集
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import EEGWindowExtractor
from dataset import RawEEGDataset


def test_window_extractor():
    """测试窗口提取器"""
    print("="*60)
    print("测试窗口提取器")
    print("="*60)
    
    # 测试文件路径
    test_file = r"E:\DataSet\EEG\EEG dataset_SUAT_processed\头皮数据-6例\曾静君\SZ1_preICA_reject_1_postICA_merged_F7_Fp1_Sph_L.set"
    
    print(f"\n测试文件: {test_file}")
    
    # 创建提取器
    extractor = EEGWindowExtractor(window_size=6.0, overlap=0.0)
    
    try:
        # 提取窗口
        print("\n提取窗口中...")
        windows, info = extractor.extract_windows(test_file)
        
        print(f"\n提取结果:")
        print(f"  采样率: {info['sfreq']} Hz")
        print(f"  通道数: {info['n_channels']}")
        print(f"  窗口数量: {info['n_windows']}")
        print(f"  窗口形状: {windows.shape}")
        print(f"  窗口大小: {info['window_size']} 秒 ({info['window_samples']} 采样点)")
        
        if len(windows) > 0:
            print(f"\n第一个窗口统计:")
            print(f"  均值: {windows[0].mean():.4f}")
            print(f"  标准差: {windows[0].std():.4f}")
            print(f"  最小值: {windows[0].min():.4f}")
            print(f"  最大值: {windows[0].max():.4f}")
            
        return True
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """测试数据集"""
    print("\n" + "="*60)
    print("测试数据集")
    print("="*60)
    
    data_root = r"E:\DataSet\EEG\EEG dataset_SUAT_processed"
    labels_csv = r"E:\output\connectivity_features\labels.csv"
    
    try:
        print("\n创建数据集...")
        dataset = RawEEGDataset(
            data_root=data_root,
            labels_csv=labels_csv,
            window_size=6.0,
            use_cache=True
        )
        
        print(f"\n数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            print("\n测试数据加载...")
            sample = dataset[0]
            
            print(f"\n第一个样本:")
            print(f"  数据形状: {sample['data'].shape}")
            print(f"  标签: {sample['label']}")
            print(f"  文件: {sample['file_path']}")
            print(f"  窗口索引: {sample['window_idx']}")
            
            print("\n数据统计:")
            print(f"  均值: {sample['data'].mean():.4f}")
            print(f"  标准差: {sample['data'].std():.4f}")
            print(f"  最小值: {sample['data'].min():.4f}")
            print(f"  最大值: {sample['data'].max():.4f}")
            
            return True
        else:
            print("警告: 数据集为空!")
            return False
            
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n开始测试...\n")
    
    # 测试窗口提取器
    success1 = test_window_extractor()
    
    # 测试数据集
    success2 = test_dataset()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"窗口提取器测试: {'通过' if success1 else '失败'}")
    print(f"数据集测试: {'通过' if success2 else '失败'}")
    
    if success1 and success2:
        print("\n所有测试通过! ✓")
    else:
        print("\n部分测试失败! ✗")


if __name__ == "__main__":
    main()

