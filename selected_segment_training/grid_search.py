#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_search.py

网格搜索脚本，用于为 train_channel_aware_kfold_optimized.py 寻找最佳超参数
"""

import subprocess
import itertools
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd
import os


class GridSearchRunner:
    """网格搜索运行器"""
    
    def __init__(self, data_root, base_save_dir, train_script_path):
        self.data_root = data_root
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        self.train_script_path = train_script_path
        
        # 记录所有实验结果
        self.results = []
        
        # 结果保存路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = self.base_save_dir / f"grid_search_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"网格搜索结果将保存到: {self.results_dir}")
    
    def define_search_space(self, search_type='quick'):
        """
        定义超参数搜索空间
        
        Args:
            search_type: 'quick' (快速搜索), 'medium' (中等), 'full' (完整搜索)
        """
        if search_type == 'quick':
            # 快速搜索：较少的组合，适合初步探索
            search_space = {
                'd_model': [64, 128],
                'n_heads': [4],
                'n_layers': [2],
                'dropout': [0.3],
                'batch_size': [16],
                'lr': [0.0001, 0.001],
                'weight_decay': [0.01],
                'window_size': [2.0],
                'window_stride': [1.0],
                'use_iou_loss': [False],
                'iou_weight': [1.5],
                'iou_type': ['basic']
            }
        elif search_type == 'medium':
            # 中等搜索：平衡搜索空间和时间
            search_space = {
                'd_model': [64, 128, 256],
                'n_heads': [4, 8],
                'n_layers': [2, 3],
                'dropout': [0.2, 0.3, 0.4],
                'batch_size': [16, 32],
                'lr': [0.00001, 0.0001, 0.001],
                'weight_decay': [0.001, 0.01, 0.1],
                'window_size': [2.0],
                'window_stride': [1.0],
                'use_iou_loss': [False, True],
                'iou_weight': [1.5],
                'iou_type': ['basic']
            }
        elif search_type == 'full':
            # 完整搜索：全面的超参数空间
            search_space = {
                'd_model': [64, 128, 256, 512],
                'n_heads': [4, 8, 16],
                'n_layers': [2, 3, 4],
                'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                'batch_size': [8, 16, 32],
                'lr': [0.00001, 0.00005, 0.0001, 0.0005, 0.001],
                'weight_decay': [0.0, 0.001, 0.01, 0.1],
                'window_size': [1.0, 2.0, 3.0],
                'window_stride': [0.5, 1.0, 1.5],
                'use_iou_loss': [False, True],
                'iou_weight': [1.0, 1.5, 2.0],
                'iou_type': ['basic', 'focal', 'weighted']
            }
        else:
            raise ValueError(f"Unknown search_type: {search_type}")
        
        return search_space
    
    def generate_combinations(self, search_space):
        """生成所有超参数组合"""
        keys = list(search_space.keys())
        values = list(search_space.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            
            # 验证约束条件
            if not self.validate_params(param_dict):
                continue
                
            combinations.append(param_dict)
        
        return combinations
    
    def validate_params(self, params):
        """验证参数组合是否合理"""
        # d_model 必须能被 n_heads 整除
        if params['d_model'] % params['n_heads'] != 0:
            return False
        
        # 如果不使用 IoU loss，IoU 相关参数无意义
        # 但为了简化，我们还是保留这些参数
        
        return True
    
    def run_training(self, params, experiment_id, n_folds=5, n_epochs=30):
        """
        运行单次训练实验
        
        Args:
            params: 超参数字典
            experiment_id: 实验编号
            n_folds: K折数量
            n_epochs: 训练轮数
            
        Returns:
            dict: 实验结果
        """
        print(f"\n{'='*80}")
        print(f"实验 {experiment_id}")
        print(f"{'='*80}")
        print(f"参数配置:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # 构建命令行参数
        cmd = [
            'python', self.train_script_path,
            '--data_root', self.data_root,
            '--d_model', str(params['d_model']),
            '--n_heads', str(params['n_heads']),
            '--n_layers', str(params['n_layers']),
            '--dropout', str(params['dropout']),
            '--batch_size', str(params['batch_size']),
            '--lr', str(params['lr']),
            '--weight_decay', str(params['weight_decay']),
            '--window_size', str(params['window_size']),
            '--window_stride', str(params['window_stride']),
            '--n_folds', str(n_folds),
            '--n_epochs', str(n_epochs),
            '--iou_weight', str(params['iou_weight']),
            '--iou_type', params['iou_type'],
            '--save_dir', str(self.results_dir / f"exp_{experiment_id}")
        ]
        
        if params['use_iou_loss']:
            cmd.append('--use_iou_loss')
        
        # 运行训练
        try:
            print(f"\n执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"训练完成!")
            
            # 读取训练结果
            result_path = self.results_dir / f"exp_{experiment_id}" / "channel_aware_kfold_optimized_fixed_*" / "kfold_results.json"
            result_files = list(self.results_dir.glob(f"exp_{experiment_id}/channel_aware_kfold_optimized_fixed_*/kfold_results.json"))
            
            if result_files:
                with open(result_files[0], 'r') as f:
                    training_results = json.load(f)
                
                mean_f1 = training_results['mean_f1']
                std_f1 = training_results['std_f1']
                max_f1 = training_results['max_f1']
                min_f1 = training_results['min_f1']
                
                print(f"结果: Mean F1={mean_f1:.2f}% ± {std_f1:.2f}%")
                
                return {
                    'experiment_id': experiment_id,
                    'params': params,
                    'mean_f1': mean_f1,
                    'std_f1': std_f1,
                    'max_f1': max_f1,
                    'min_f1': min_f1,
                    'status': 'success'
                }
            else:
                print("警告: 未找到结果文件")
                return {
                    'experiment_id': experiment_id,
                    'params': params,
                    'mean_f1': 0.0,
                    'std_f1': 0.0,
                    'max_f1': 0.0,
                    'min_f1': 0.0,
                    'status': 'no_results'
                }
                
        except subprocess.CalledProcessError as e:
            print(f"错误: 训练失败")
            print(f"错误信息: {e.stderr}")
            return {
                'experiment_id': experiment_id,
                'params': params,
                'mean_f1': 0.0,
                'std_f1': 0.0,
                'max_f1': 0.0,
                'min_f1': 0.0,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_grid_search(self, search_space, n_folds=5, n_epochs=30, resume_from=0):
        """
        执行网格搜索
        
        Args:
            search_space: 超参数搜索空间
            n_folds: K折数量
            n_epochs: 每个实验的训练轮数
            resume_from: 从第几个实验开始（用于恢复中断的搜索）
        """
        combinations = self.generate_combinations(search_space)
        total_experiments = len(combinations)
        
        print(f"\n{'='*80}")
        print(f"网格搜索配置")
        print(f"{'='*80}")
        print(f"总实验数: {total_experiments}")
        print(f"K折数: {n_folds}")
        print(f"每个实验训练轮数: {n_epochs}")
        print(f"从实验 {resume_from} 开始")
        
        # 执行所有实验
        for idx, params in enumerate(combinations[resume_from:], start=resume_from):
            result = self.run_training(params, idx, n_folds, n_epochs)
            self.results.append(result)
            
            # 保存中间结果
            self.save_results()
            
            # 打印当前最佳结果
            self.print_best_results()
        
        print(f"\n{'='*80}")
        print(f"网格搜索完成!")
        print(f"{'='*80}")
        
        return self.results
    
    def save_results(self):
        """保存搜索结果"""
        # 保存详细结果（JSON）
        results_json_path = self.results_dir / "grid_search_results.json"
        with open(results_json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 保存结果表格（CSV）
        results_data = []
        for result in self.results:
            row = {
                'experiment_id': result['experiment_id'],
                'mean_f1': result['mean_f1'],
                'std_f1': result['std_f1'],
                'max_f1': result['max_f1'],
                'min_f1': result['min_f1'],
                'status': result['status']
            }
            row.update(result['params'])
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        df = df.sort_values('mean_f1', ascending=False)
        
        results_csv_path = self.results_dir / "grid_search_results.csv"
        df.to_csv(results_csv_path, index=False)
        
        print(f"\n结果已保存到:")
        print(f"  JSON: {results_json_path}")
        print(f"  CSV: {results_csv_path}")
    
    def print_best_results(self, top_k=5):
        """打印当前最佳结果"""
        if not self.results:
            return
        
        # 按 mean_f1 排序
        sorted_results = sorted(
            [r for r in self.results if r['status'] == 'success'],
            key=lambda x: x['mean_f1'],
            reverse=True
        )
        
        if not sorted_results:
            print("\n暂无成功的实验结果")
            return
        
        print(f"\n{'='*80}")
        print(f"当前 Top {min(top_k, len(sorted_results))} 结果")
        print(f"{'='*80}")
        
        for i, result in enumerate(sorted_results[:top_k], 1):
            print(f"\n第 {i} 名: 实验 {result['experiment_id']}")
            print(f"  Mean F1: {result['mean_f1']:.2f}% ± {result['std_f1']:.2f}%")
            print(f"  参数:")
            for key, value in result['params'].items():
                print(f"    {key}: {value}")
    
    def generate_report(self):
        """生成详细的搜索报告"""
        if not self.results:
            print("没有结果可生成报告")
            return
        
        report_path = self.results_dir / "grid_search_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("网格搜索报告\n")
            f.write("="*80 + "\n\n")
            
            # 统计信息
            successful = [r for r in self.results if r['status'] == 'success']
            failed = [r for r in self.results if r['status'] == 'failed']
            
            f.write(f"总实验数: {len(self.results)}\n")
            f.write(f"成功: {len(successful)}\n")
            f.write(f"失败: {len(failed)}\n\n")
            
            if successful:
                # 最佳结果
                best_result = max(successful, key=lambda x: x['mean_f1'])
                f.write("="*80 + "\n")
                f.write("最佳结果\n")
                f.write("="*80 + "\n")
                f.write(f"实验ID: {best_result['experiment_id']}\n")
                f.write(f"Mean F1: {best_result['mean_f1']:.2f}% ± {best_result['std_f1']:.2f}%\n")
                f.write(f"Max F1: {best_result['max_f1']:.2f}%\n")
                f.write(f"Min F1: {best_result['min_f1']:.2f}%\n\n")
                f.write("最佳参数:\n")
                for key, value in best_result['params'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                # Top 10 结果
                sorted_results = sorted(successful, key=lambda x: x['mean_f1'], reverse=True)
                f.write("="*80 + "\n")
                f.write("Top 10 结果\n")
                f.write("="*80 + "\n\n")
                
                for i, result in enumerate(sorted_results[:10], 1):
                    f.write(f"{i}. 实验 {result['experiment_id']}\n")
                    f.write(f"   Mean F1: {result['mean_f1']:.2f}% ± {result['std_f1']:.2f}%\n")
                    f.write(f"   关键参数: ")
                    key_params = ['d_model', 'n_heads', 'n_layers', 'lr', 'dropout']
                    param_str = ", ".join([f"{k}={result['params'][k]}" for k in key_params if k in result['params']])
                    f.write(param_str + "\n\n")
                
                # 参数分析
                f.write("="*80 + "\n")
                f.write("参数影响分析\n")
                f.write("="*80 + "\n\n")
                
                # 分析每个参数的影响
                for param_name in successful[0]['params'].keys():
                    f.write(f"\n{param_name}:\n")
                    param_groups = {}
                    for result in successful:
                        param_value = result['params'][param_name]
                        if param_value not in param_groups:
                            param_groups[param_value] = []
                        param_groups[param_value].append(result['mean_f1'])
                    
                    for value, f1_scores in sorted(param_groups.items(), key=lambda x: np.mean(x[1]), reverse=True):
                        mean_f1 = np.mean(f1_scores)
                        std_f1 = np.std(f1_scores)
                        f.write(f"  {value}: {mean_f1:.2f}% ± {std_f1:.2f}% (n={len(f1_scores)})\n")
        
        print(f"\n详细报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='EEG模型超参数网格搜索')
    
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录')
    parser.add_argument('--train_script', type=str, 
                        default='train_channel_aware_kfold_optimized.py',
                        help='训练脚本路径')
    parser.add_argument('--save_dir', type=str, 
                        default='grid_search_results',
                        help='结果保存目录')
    parser.add_argument('--search_type', type=str, 
                        default='quick',
                        choices=['quick', 'medium', 'full'],
                        help='搜索类型: quick(快速), medium(中等), full(完整)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='K折交叉验证折数')
    parser.add_argument('--n_epochs', type=int, default=30,
                        help='每个实验的训练轮数')
    parser.add_argument('--resume_from', type=int, default=0,
                        help='从第几个实验开始（用于恢复中断的搜索）')
    
    args = parser.parse_args()
    
    # 创建网格搜索运行器
    runner = GridSearchRunner(
        data_root=args.data_root,
        base_save_dir=args.save_dir,
        train_script_path=args.train_script
    )
    
    # 定义搜索空间
    search_space = runner.define_search_space(args.search_type)
    
    print(f"\n搜索空间:")
    for key, values in search_space.items():
        print(f"  {key}: {values}")
    
    # 运行网格搜索
    results = runner.run_grid_search(
        search_space=search_space,
        n_folds=args.n_folds,
        n_epochs=args.n_epochs,
        resume_from=args.resume_from
    )
    
    # 生成报告
    runner.generate_report()
    
    print(f"\n网格搜索完成! 共完成 {len(results)} 个实验")


if __name__ == "__main__":
    main()

