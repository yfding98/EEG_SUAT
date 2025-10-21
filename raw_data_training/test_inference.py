"""
推理测试脚本 - 快速测试单个文件
"""

import sys
from pathlib import Path

# 示例文件路径
EXAMPLE_SET_FILE = r"E:\DataSet\EEG\EEG dataset_SUAT_processed\头皮数据-6例\曾静君\SZ1_preICA_reject_1_postICA_merged_F7_Fp1_Sph_L.set"


def find_latest_checkpoint(checkpoints_dir='checkpoints_multitask'):
    """查找最新的检查点"""
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"检查点目录不存在: {checkpoints_dir}")
        return None
    
    # 查找所有best_model.pth
    checkpoints = list(checkpoints_path.glob('*/best_model.pth'))
    
    if not checkpoints:
        print(f"未找到检查点文件")
        return None
    
    # 返回最新的
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"找到最新检查点: {latest}")
    
    return latest


def test_single_file():
    """测试单个文件"""
    
    print("="*70)
    print("EEG推理测试")
    print("="*70)
    
    # 查找检查点
    checkpoint = find_latest_checkpoint()
    
    if checkpoint is None:
        print("\n请先训练模型:")
        print("  python train_multitask.py")
        return
    
    # 检查测试文件
    if not Path(EXAMPLE_SET_FILE).exists():
        print(f"\n示例文件不存在: {EXAMPLE_SET_FILE}")
        print("\n请修改 test_inference.py 中的 EXAMPLE_SET_FILE 为实际文件路径")
        return
    
    print(f"\n测试文件: {EXAMPLE_SET_FILE}")
    
    # 导入推理器
    from inference import EEGInference
    
    # 创建推理器
    print("\n创建推理器...")
    inference = EEGInference(checkpoint, device='cuda')
    
    # 执行推理
    print("\n执行推理...")
    results = inference.predict(
        set_file_path=EXAMPLE_SET_FILE,
        window_idx=0,  # 只预测第一个窗口
        visualize=True,
        save_results=True,
        save_dir='test_inference_results'
    )
    
    if results:
        print(f"\n{'='*70}")
        print("测试成功！")
        print(f"{'='*70}")
        print(f"\n结果已保存到: test_inference_results/")
        print(f"  - prediction_*.json: 预测结果JSON")
        print(f"  - window_0_prediction.png: 可视化图片")
    else:
        print("\n推理失败！")


def test_multiple_windows():
    """测试多个窗口"""
    
    print("="*70)
    print("多窗口推理测试")
    print("="*70)
    
    checkpoint = find_latest_checkpoint()
    if checkpoint is None:
        return
    
    if not Path(EXAMPLE_SET_FILE).exists():
        print(f"\n示例文件不存在: {EXAMPLE_SET_FILE}")
        return
    
    from inference import EEGInference
    
    inference = EEGInference(checkpoint, device='cuda')
    
    # 预测前3个窗口
    print("\n预测前3个窗口...")
    for window_idx in range(3):
        print(f"\n{'='*70}")
        print(f"窗口 #{window_idx}")
        print(f"{'='*70}")
        
        results = inference.predict(
            set_file_path=EXAMPLE_SET_FILE,
            window_idx=window_idx,
            visualize=True,
            save_results=False,
            save_dir=f'test_inference_results/window_{window_idx}'
        )
        
        if results and len(results['results']) > 0:
            r = results['results'][0]
            print(f"\n简要结果:")
            print(f"  发作类型: SZ{r['seizure_type']} (置信度: {r['seizure_confidence']*100:.1f}%)")
            print(f"  活跃通道: {r['active_channel_names']}")


if __name__ == "__main__":
    print("\n选择测试模式:")
    print("  1. 测试单个窗口（推荐）")
    print("  2. 测试多个窗口")
    print("  3. 退出")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == '1':
        test_single_file()
    elif choice == '2':
        test_multiple_windows()
    else:
        print("退出")

