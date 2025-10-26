#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_frequency_bands.py

运行所有频段的训练脚本
自动训练所有频段并聚合结果
"""

import subprocess
import sys
import time
from pathlib import Path
import argparse
from datetime import datetime
import json
from typing import List, Dict
import threading
import queue


class FrequencyBandTrainer:
    """频段训练管理器"""
    
    def __init__(self, data_root: str, base_save_dir: str = "checkpoints_frequency_band"):
        self.data_root = data_root
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        self.results = {}
        self.start_time = None
        
    def train_single_band(self, frequency_band: str, config: Dict) -> Dict:
        """训练单个频段"""
        print(f"\n{'='*80}")
        print(f"开始训练 {frequency_band} 频段")
        print(f"{'='*80}")
        
        # 构建训练命令
        cmd = [
            'python', 'train_frequency_band.py',
            '--data_root', self.data_root,
            '--frequency_band', frequency_band,
            '--window_size', str(config.get('window_size', 6.0)),
            '--window_stride', str(config.get('window_stride', 3.0)),
            '--sampling_rate', str(config.get('sampling_rate', 250)),
            '--d_model', str(config.get('d_model', 128)),
            '--n_heads', str(config.get('n_heads', 8)),
            '--n_layers', str(config.get('n_layers', 2)),
            '--dropout', str(config.get('dropout', 0.3)),
            '--batch_size', str(config.get('batch_size', 8)),
            '--lr', str(config.get('lr', 0.001)),
            '--weight_decay', str(config.get('weight_decay', 0.01)),
            '--n_epochs', str(config.get('n_epochs', 50)),
            '--early_stopping_patience', str(config.get('early_stopping_patience', 20)),
            '--save_dir', str(self.base_save_dir)
        ]
        
        # 添加布尔参数
        if config.get('use_focal_loss', False):
            cmd.append('--use_focal_loss')
        if config.get('use_class_weights', True):
            cmd.append('--use_class_weights')
        
        # 添加其他参数
        if 'focal_alpha' in config:
            cmd.extend(['--focal_alpha', str(config['focal_alpha'])])
        if 'focal_gamma' in config:
            cmd.extend(['--focal_gamma', str(config['focal_gamma'])])
        if 'val_split' in config:
            cmd.extend(['--val_split', str(config['val_split'])])
        if 'test_split' in config:
            cmd.extend(['--test_split', str(config['test_split'])])
        if 'seed' in config:
            cmd.extend(['--seed', str(config['seed'])])
        
        print(f"执行命令: {' '.join(cmd)}")
        
        try:
            # 运行训练
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).parent
            )
            
            print(f"{frequency_band} 频段训练完成!")
            
            # 查找结果文件
            band_dirs = list(self.base_save_dir.glob(f"{frequency_band}_*"))
            if band_dirs:
                latest_dir = max(band_dirs, key=lambda x: x.stat().st_mtime)
                result_file = latest_dir / 'final_results.json'
                
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                    
                    return {
                        'frequency_band': frequency_band,
                        'status': 'success',
                        'result_data': result_data,
                        'experiment_dir': str(latest_dir)
                    }
                else:
                    return {
                        'frequency_band': frequency_band,
                        'status': 'no_results',
                        'error': '未找到结果文件'
                    }
            else:
                return {
                    'frequency_band': frequency_band,
                    'status': 'no_experiment_dir',
                    'error': '未找到实验目录'
                }
                
        except subprocess.CalledProcessError as e:
            print(f"错误: {frequency_band} 频段训练失败")
            print(f"错误信息: {e.stderr}")
            return {
                'frequency_band': frequency_band,
                'status': 'failed',
                'error': str(e)
            }
    
    def train_all_bands(self, config: Dict, parallel: bool = False, max_workers: int = 2) -> Dict:
        """训练所有频段"""
        self.start_time = datetime.now()
        
        print(f"\n{'='*80}")
        print("开始训练所有频段")
        print(f"{'='*80}")
        print(f"数据路径: {self.data_root}")
        print(f"保存目录: {self.base_save_dir}")
        print(f"频段列表: {', '.join(self.frequency_bands)}")
        print(f"并行训练: {parallel}")
        if parallel:
            print(f"最大工作进程: {max_workers}")
        
        if parallel:
            return self._train_parallel(config, max_workers)
        else:
            return self._train_sequential(config)
    
    def _train_sequential(self, config: Dict) -> Dict:
        """顺序训练所有频段"""
        results = {}
        
        for i, band in enumerate(self.frequency_bands, 1):
            print(f"\n进度: {i}/{len(self.frequency_bands)} - {band} 频段")
            
            result = self.train_single_band(band, config)
            results[band] = result
            
            # 打印当前状态
            if result['status'] == 'success':
                test_f1 = result['result_data'].get('test_metrics', {}).get('macro_f1', 0)
                print(f"  {band} 频段完成: F1={test_f1:.2f}%")
            else:
                print(f"  {band} 频段失败: {result.get('error', '未知错误')}")
        
        return results
    
    def _train_parallel(self, config: Dict, max_workers: int) -> Dict:
        """并行训练所有频段"""
        import concurrent.futures
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_band = {
                executor.submit(self.train_single_band, band, config): band 
                for band in self.frequency_bands
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_band):
                band = future_to_band[future]
                try:
                    result = future.result()
                    results[band] = result
                    
                    if result['status'] == 'success':
                        test_f1 = result['result_data'].get('test_metrics', {}).get('macro_f1', 0)
                        print(f"  {band} 频段完成: F1={test_f1:.2f}%")
                    else:
                        print(f"  {band} 频段失败: {result.get('error', '未知错误')}")
                        
                except Exception as e:
                    print(f"  {band} 频段异常: {e}")
                    results[band] = {
                        'frequency_band': band,
                        'status': 'exception',
                        'error': str(e)
                    }
        
        return results
    
    def aggregate_results(self, results: Dict) -> Dict:
        """聚合结果"""
        print(f"\n{'='*80}")
        print("聚合频段训练结果")
        print(f"{'='*80}")
        
        # 统计成功和失败的频段
        successful_bands = [band for band, result in results.items() if result['status'] == 'success']
        failed_bands = [band for band, result in results.items() if result['status'] != 'success']
        
        print(f"成功频段: {len(successful_bands)} - {', '.join(successful_bands)}")
        print(f"失败频段: {len(failed_bands)} - {', '.join(failed_bands)}")
        
        if not successful_bands:
            print("没有成功的频段训练")
            return {}
        
        # 计算各频段性能
        band_performance = {}
        for band in successful_bands:
            result = results[band]
            test_metrics = result['result_data'].get('test_metrics', {})
            band_performance[band] = {
                'test_f1': test_metrics.get('macro_f1', 0),
                'test_precision': test_metrics.get('macro_precision', 0),
                'test_recall': test_metrics.get('macro_recall', 0),
                'test_micro_f1': test_metrics.get('micro_f1', 0),
                'val_f1': result['result_data'].get('best_val_f1', 0)
            }
        
        # 找出最佳频段
        best_band = max(band_performance.keys(), key=lambda x: band_performance[x]['test_f1'])
        best_f1 = band_performance[best_band]['test_f1']
        
        print(f"\n最佳频段: {best_band} (F1: {best_f1:.2f}%)")
        
        # 打印所有频段性能
        print(f"\n各频段性能:")
        for band in sorted(band_performance.keys(), key=lambda x: band_performance[x]['test_f1'], reverse=True):
            perf = band_performance[band]
            print(f"  {band}: F1={perf['test_f1']:.2f}%, Precision={perf['test_precision']:.2f}%, Recall={perf['test_recall']:.2f}%")
        
        # 计算总训练时间
        if self.start_time:
            total_time = datetime.now() - self.start_time
            print(f"\n总训练时间: {total_time}")
        
        return {
            'band_performance': band_performance,
            'best_band': best_band,
            'best_f1': best_f1,
            'successful_bands': successful_bands,
            'failed_bands': failed_bands,
            'total_time': str(total_time) if self.start_time else None
        }
    
    def save_summary(self, results: Dict, summary: Dict):
        """保存汇总结果"""
        summary_file = self.base_save_dir / 'training_summary.json'
        
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'data_root': self.data_root,
            'base_save_dir': str(self.base_save_dir),
            'frequency_bands': self.frequency_bands,
            'results': results,
            'summary': summary
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n汇总结果保存到: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='运行所有频段训练')
    
    # 基本参数
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据根目录')
    parser.add_argument('--save_dir', type=str, default='checkpoints_frequency_band',
                       help='保存目录')
    
    # 训练参数
    parser.add_argument('--window_size', type=float, default=6.0,
                       help='窗口大小（秒）')
    parser.add_argument('--window_stride', type=float, default=3.0,
                       help='窗口步长（秒）')
    parser.add_argument('--sampling_rate', type=int, default=250,
                       help='采样率')
    parser.add_argument('--d_model', type=int, default=128,
                       help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=2,
                       help='Transformer层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout率')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--n_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='早停耐心值')
    
    # 损失函数参数
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='使用Focal Loss')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                       help='使用类别权重')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                       help='Focal Loss alpha参数')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal Loss gamma参数')
    
    # 其他参数
    parser.add_argument('--parallel', action='store_true',
                       help='并行训练')
    parser.add_argument('--max_workers', type=int, default=2,
                       help='最大工作进程数')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='验证集比例')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = FrequencyBandTrainer(
        data_root=args.data_root,
        base_save_dir=args.save_dir
    )
    
    # 构建配置
    config = {
        'window_size': args.window_size,
        'window_stride': args.window_stride,
        'sampling_rate': args.sampling_rate,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'n_epochs': args.n_epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'use_focal_loss': args.use_focal_loss,
        'use_class_weights': args.use_class_weights,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        'val_split': args.val_split,
        'test_split': args.test_split,
        'seed': args.seed
    }
    
    # 训练所有频段
    results = trainer.train_all_bands(config, parallel=args.parallel, max_workers=args.max_workers)
    
    # 聚合结果
    summary = trainer.aggregate_results(results)
    
    # 保存汇总
    trainer.save_summary(results, summary)
    
    print(f"\n{'='*80}")
    print("所有频段训练完成!")
    print(f"{'='*80}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--data_root', r'E:\DataSet\EEG\EEG dataset_SUAT_processed_selected',
            '--window_size', '6',
            '--window_stride', '3',
            '--batch_size', '8',
            '--d_model', '128',
            '--n_heads', '8',
            '--n_layers', '2',
            '--use_focal_loss',
            '--use_class_weights',
            '--n_epochs', '30'
        ])
    main()

