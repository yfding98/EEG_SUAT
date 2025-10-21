"""
快速分析当前模型的问题样本
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from dataset import RawEEGDataset, create_dataloaders
from model import create_model
from utils import load_checkpoint


def analyze_current_model(checkpoint_path, data_root, labels_csv):
    """分析当前模型，找出问题样本"""
    
    print(f"分析模型: {checkpoint_path}")
    
    # 加载配置
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 创建数据集
    print("加载数据...")
    dataset = RawEEGDataset(
        data_root=data_root,
        labels_csv=labels_csv,
        window_size=config.get('window_size', 6.0),
        use_cache=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0
    )
    
    # 获取数据形状
    sample = dataset[0]
    n_channels = sample['data'].shape[0]
    n_samples = sample['data'].shape[1]
    n_classes = config.get('n_classes', 5)
    
    # 加载模型
    print("加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_model(
        model_type=config.get('model_type', 'lightweight'),
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 分析每个样本
    print("分析样本...")
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = batch['data'].to(device)
            target = batch['label'].to(device)
            file_paths = batch['file_path']
            window_indices = batch['window_idx'].cpu().numpy()
            
            output = model(data)
            losses = criterion(output, target)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            for i in range(len(data)):
                results.append({
                    'file_path': file_paths[i],
                    'window_idx': int(window_indices[i]),
                    'loss': float(losses[i].cpu()),
                    'predicted': int(preds[i].cpu()),
                    'actual': int(target[i].cpu()),
                    'confidence': float(probs[i].max().cpu()),
                    'correct': int(preds[i].cpu()) == int(target[i].cpu())
                })
    
    # 分析结果
    print("\n分析结果:")
    
    losses = [r['loss'] for r in results]
    loss_mean = np.mean(losses)
    loss_std = np.std(losses)
    loss_threshold = loss_mean + 2 * loss_std
    
    print(f"  总样本数: {len(results)}")
    print(f"  Loss均值: {loss_mean:.4f}")
    print(f"  Loss标准差: {loss_std:.4f}")
    print(f"  Loss阈值 (μ+2σ): {loss_threshold:.4f}")
    
    # 找出高loss样本
    high_loss_samples = [r for r in results if r['loss'] > loss_threshold]
    print(f"  高loss样本: {len(high_loss_samples)} ({len(high_loss_samples)/len(results)*100:.1f}%)")
    
    # 找出预测错误样本
    wrong_samples = [r for r in results if not r['correct']]
    print(f"  预测错误: {len(wrong_samples)} ({len(wrong_samples)/len(results)*100:.1f}%)")
    
    # 找出低置信度样本
    low_conf_samples = [r for r in results if r['confidence'] < 0.5]
    print(f"  低置信度(<0.5): {len(low_conf_samples)} ({len(low_conf_samples)/len(results)*100:.1f}%)")
    
    # 组合badcase: 高loss OR 预测错误 OR 低置信度
    bad_samples_set = set()
    for r in high_loss_samples + wrong_samples + low_conf_samples:
        bad_samples_set.add((r['file_path'], r['window_idx']))
    
    bad_samples = [
        {'file_path': file_path, 'window_idx': window_idx}
        for file_path, window_idx in bad_samples_set
    ]
    
    print(f"  总badcase数: {len(bad_samples)} ({len(bad_samples)/len(results)*100:.1f}%)")
    
    # 保存结果
    output_file = 'badcase_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'bad_windows': bad_samples,
            'statistics': {
                'total_samples': len(results),
                'bad_samples': len(bad_samples),
                'loss_mean': loss_mean,
                'loss_std': loss_std,
                'loss_threshold': loss_threshold,
                'high_loss_count': len(high_loss_samples),
                'wrong_predictions': len(wrong_samples),
                'low_confidence': len(low_conf_samples)
            },
            'all_results': results
        }, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 按文件统计badcase
    file_stats = {}
    for bad in bad_samples:
        file_path = bad['file_path']
        if file_path not in file_stats:
            file_stats[file_path] = 0
        file_stats[file_path] += 1
    
    print(f"\n每个文件的badcase数:")
    for file_path, count in sorted(file_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {file_path}: {count}")
    
    return bad_samples


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        # 查找最新的checkpoint
        checkpoint_path = Path('checkpoints')
        if checkpoint_path.exists():
            checkpoints = list(checkpoint_path.glob('*/best_model.pth'))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"使用最新checkpoint: {latest}")
            else:
                print("未找到checkpoint")
                sys.exit(1)
        else:
            print("未找到checkpoints目录")
            sys.exit(1)
    else:
        latest = Path(sys.argv[1])
    
    data_root = r'E:\DataSet\EEG\EEG dataset_SUAT_processed'
    labels_csv = r'E:\output\connectivity_features\labels.csv'
    
    bad_samples = analyze_current_model(latest, data_root, labels_csv)
    
    print(f"\n完成！现在可以使用 badcase_analysis.json 来过滤数据集")
    print("运行: python train_improved.py --bad_windows_file badcase_analysis.json")

