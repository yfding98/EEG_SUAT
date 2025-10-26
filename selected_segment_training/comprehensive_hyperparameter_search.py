#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comprehensive_hyperparameter_search.py

基于train_basic_eeg_classifier.py的完整超参数搜索方案
支持网格搜索和贝叶斯搜索，覆盖所有超参数
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import sys
import os
import itertools
from typing import Dict, List, Any, Optional
import pandas as pd

try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
        plot_contour
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("警告: Optuna未安装。请运行: pip install optuna plotly kaleido")


class ComprehensiveHyperparameterSearch:
    """基于train_basic_eeg_classifier.py的完整超参数搜索"""
    
    def __init__(self, data_root: str, base_save_dir: str):
        self.data_root = data_root
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果保存路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = self.base_save_dir / f"comprehensive_search_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"超参数搜索结果将保存到: {self.results_dir}")
        
        # 试验计数器
        self.trial_counter = 0
        self.results = []
    
    def define_search_space(self, search_type: str = 'quick') -> Dict[str, List]:
        """定义完整的搜索空间"""
        if search_type == 'quick':
            return {
                # 数据参数
                'window_size': [3.0, 6.0],
                'window_stride': [1.0, 3.0],
                
                # 模型参数
                'd_model': [64, 128],
                'n_heads': [4, 8],
                'n_layers': [2],
                'dropout': [0.3],
                
                # 训练参数
                'batch_size': [8, 16],
                'lr': [0.0001, 0.001],
                'weight_decay': [0.01],
                'early_stopping_patience': [20],
                
                # 损失函数参数
                'use_focal_loss': [False, True],
                'use_class_weights': [True],
                'focal_alpha': [1.0],
                'focal_gamma': [2.0],
                
                # 数据分割参数
                'val_split': [0.15],
                'test_split': [0.15]
            }
        elif search_type == 'medium':
            return {
                # 数据参数
                'window_size': [2.0, 3.0, 6.0],
                'window_stride': [1.0, 2.0, 3.0],
                
                # 模型参数
                'd_model': [64, 128, 256],
                'n_heads': [4, 8],
                'n_layers': [2, 3],
                'dropout': [0.2, 0.3, 0.4],
                
                # 训练参数
                'batch_size': [8, 16, 32],
                'lr': [0.00001, 0.0001, 0.001],
                'weight_decay': [0.001, 0.01, 0.1],
                'early_stopping_patience': [15, 20, 25],
                
                # 损失函数参数
                'use_focal_loss': [False, True],
                'use_class_weights': [True],
                'focal_alpha': [0.5, 1.0, 2.0],
                'focal_gamma': [1.0, 2.0, 3.0],
                
                # 数据分割参数
                'val_split': [0.1, 0.15, 0.2],
                'test_split': [0.1, 0.15, 0.2]
            }
        elif search_type == 'full':
            return {
                # 数据参数
                'window_size': [1.0, 2.0, 3.0, 6.0, 9.0],
                'window_stride': [0.5, 1.0, 2.0, 3.0, 4.5],
                
                # 模型参数
                'd_model': [32, 64, 128, 256, 512],
                'n_heads': [2, 4, 8, 16],
                'n_layers': [1, 2, 3, 4, 6],
                'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                
                # 训练参数
                'batch_size': [4, 8, 16, 32, 64],
                'lr': [0.000001, 0.00001, 0.0001, 0.001, 0.01],
                'weight_decay': [0.0, 0.0001, 0.001, 0.01, 0.1],
                'early_stopping_patience': [10, 15, 20, 25, 30],
                
                # 损失函数参数
                'use_focal_loss': [False, True],
                'use_class_weights': [False, True],
                'focal_alpha': [0.25, 0.5, 1.0, 2.0, 4.0],
                'focal_gamma': [0.5, 1.0, 2.0, 3.0, 4.0],
                
                # 数据分割参数
                'val_split': [0.05, 0.1, 0.15, 0.2, 0.25],
                'test_split': [0.05, 0.1, 0.15, 0.2, 0.25]
            }
        else:
            raise ValueError(f"Unknown search_type: {search_type}")
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """验证参数组合的合理性"""
        # d_model 必须能被 n_heads 整除
        if params['d_model'] % params['n_heads'] != 0:
            return False
        
        # 窗口步长必须小于窗口大小
        if params['window_stride'] >= params['window_size']:
            return False
        
        # 验证集和测试集比例不能太大
        if params['val_split'] + params['test_split'] >= 0.5:
            return False
        
        # 学习率必须在合理范围内
        if params['lr'] < 1e-6 or params['lr'] > 0.1:
            return False
        
        # 批大小必须为正数
        if params['batch_size'] <= 0:
            return False
        
        return True
    
    def build_command(self, params: Dict[str, Any], experiment_id: int, 
                     n_epochs: int = 30) -> List[str]:
        """构建训练命令"""
        save_dir = str(self.results_dir / f"exp_{experiment_id}")
        
        cmd = [
            'python', 'train_basic_eeg_classifier.py',
            '--data_root', self.data_root,
            '--window_size', str(params['window_size']),
            '--window_stride', str(params['window_stride']),
            '--d_model', str(params['d_model']),
            '--n_heads', str(params['n_heads']),
            '--n_layers', str(params['n_layers']),
            '--dropout', str(params['dropout']),
            '--batch_size', str(params['batch_size']),
            '--lr', str(params['lr']),
            '--weight_decay', str(params['weight_decay']),
            '--early_stopping_patience', str(params['early_stopping_patience']),
            '--focal_alpha', str(params['focal_alpha']),
            '--focal_gamma', str(params['focal_gamma']),
            '--val_split', str(params['val_split']),
            '--test_split', str(params['test_split']),
            '--n_epochs', str(n_epochs),
            '--save_dir', save_dir
        ]
        
        # 添加布尔参数
        if params['use_focal_loss']:
            cmd.append('--use_focal_loss')
        if params['use_class_weights']:
            cmd.append('--use_class_weights')
        
        return cmd
    
    def run_training(self, params: Dict[str, Any], experiment_id: int, 
                    n_epochs: int = 30) -> Dict[str, Any]:
        """运行单次训练实验"""
        print(f"\n{'='*80}")
        print(f"实验 {experiment_id}")
        print(f"{'='*80}")
        print(f"参数配置:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # 构建命令
        cmd = self.build_command(params, experiment_id, n_epochs)
        
        try:
            print(f"\n执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"训练完成!")
            
            # 查找结果文件
            save_dir = self.results_dir / f"exp_{experiment_id}"
            result_files = list(save_dir.glob("basic_eeg_*/final_results.json"))
            
            if result_files:
                with open(result_files[0], 'r') as f:
                    training_results = json.load(f)
                
                # 解析结果
                test_metrics = training_results.get('test_metrics', {})
                best_val_f1 = training_results.get('best_val_f1', 0.0)
                
                mean_f1 = test_metrics.get('macro_f1', 0.0)
                std_f1 = 0.0  # 单次训练没有标准差
                max_f1 = test_metrics.get('macro_f1', 0.0)
                min_f1 = test_metrics.get('macro_f1', 0.0)
                
                print(f"结果: Test F1={mean_f1:.2f}%, Val F1={best_val_f1:.2f}%")
                
                return {
                    'experiment_id': experiment_id,
                    'params': params,
                    'metrics': {
                        'mean_f1': mean_f1,
                        'std_f1': std_f1,
                        'max_f1': max_f1,
                        'min_f1': min_f1,
                        'best_val_f1': best_val_f1
                    },
                    'status': 'success',
                    'result_file': str(result_files[0])
                }
            else:
                print("警告: 未找到结果文件")
                return {
                    'experiment_id': experiment_id,
                    'params': params,
                    'metrics': {'mean_f1': 0.0, 'std_f1': 0.0, 'max_f1': 0.0, 'min_f1': 0.0, 'best_val_f1': 0.0},
                    'status': 'no_results'
                }
                
        except subprocess.CalledProcessError as e:
            print(f"错误: 训练失败")
            print(f"错误信息: {e.stderr}")
            return {
                'experiment_id': experiment_id,
                'params': params,
                'metrics': {'mean_f1': 0.0, 'std_f1': 0.0, 'max_f1': 0.0, 'min_f1': 0.0, 'best_val_f1': 0.0},
                'status': 'failed',
                'error': str(e)
            }
    
    def run_grid_search(self, search_space: Dict[str, List], n_epochs: int = 30, 
                       resume_from: int = 0) -> List[Dict[str, Any]]:
        """运行网格搜索"""
        # 生成所有参数组合
        keys = list(search_space.keys())
        values = list(search_space.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            if self.validate_params(param_dict):
                combinations.append(param_dict)
        
        total_experiments = len(combinations)
        
        print(f"\n{'='*80}")
        print(f"网格搜索配置")
        print(f"{'='*80}")
        print(f"总实验数: {total_experiments}")
        print(f"每个实验训练轮数: {n_epochs}")
        print(f"从实验 {resume_from} 开始")
        
        # 执行所有实验
        for idx, params in enumerate(combinations[resume_from:], start=resume_from):
            result = self.run_training(params, idx, n_epochs)
            self.results.append(result)
            
            # 保存中间结果
            self.save_results()
            
            # 打印当前最佳结果
            self.print_best_results()
        
        print(f"\n{'='*80}")
        print(f"网格搜索完成!")
        print(f"{'='*80}")
        
        return self.results
    
    def run_bayesian_search(self, n_trials: int = 50, n_epochs: int = 30, 
                           timeout: Optional[int] = None) -> Any:
        """运行贝叶斯搜索"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("需要安装Optuna: pip install optuna")
        
        def objective(trial):
            # 定义超参数搜索空间
            params = {}
            
            # 数据参数
            params['window_size'] = trial.suggest_categorical('window_size', [1.0, 2.0, 3.0, 6.0, 9.0])
            params['window_stride'] = trial.suggest_categorical('window_stride', [0.5, 1.0, 2.0, 3.0, 4.5])
            
            # 模型参数
            d_model_choices = [32, 64, 128, 256, 512]
            d_model = trial.suggest_categorical('d_model', d_model_choices)
            
            # n_heads必须是d_model的因数
            valid_n_heads = [h for h in [2, 4, 8, 16] if d_model % h == 0]
            n_heads = trial.suggest_categorical('n_heads', valid_n_heads)
            
            params.update({
                'd_model': d_model,
                'n_heads': n_heads,
                'n_layers': trial.suggest_int('n_layers', 1, 6),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
                
                # 训练参数
                'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64]),
                'lr': trial.suggest_float('lr', 1e-6, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
                'early_stopping_patience': trial.suggest_int('early_stopping_patience', 10, 30),
                
                # 损失函数参数
                'use_focal_loss': trial.suggest_categorical('use_focal_loss', [False, True]),
                'use_class_weights': trial.suggest_categorical('use_class_weights', [False, True]),
                'focal_alpha': trial.suggest_float('focal_alpha', 0.25, 4.0, log=True),
                'focal_gamma': trial.suggest_float('focal_gamma', 0.5, 4.0, log=True),
                
                # 数据分割参数
                'val_split': trial.suggest_float('val_split', 0.05, 0.25, step=0.05),
                'test_split': trial.suggest_float('test_split', 0.05, 0.25, step=0.05)
            })
            
            # 验证参数
            if not self.validate_params(params):
                return 0.0
            
            # 运行训练
            result = self.run_training(params, self.trial_counter, n_epochs)
            self.trial_counter += 1
            
            return result['metrics']['mean_f1']
        
        # 创建研究
        study = optuna.create_study(
            direction='maximize',
            study_name='comprehensive_eeg_optimization',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        
        # 运行优化
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # 保存研究结果
        self._save_study(study)
        
        return study
    
    def save_results(self):
        """保存搜索结果"""
        # 保存详细结果（JSON）
        results_json_path = self.results_dir / "search_results.json"
        with open(results_json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 保存结果表格（CSV）
        if self.results:
            results_data = []
            for result in self.results:
                row = {
                    'experiment_id': result['experiment_id'],
                    'mean_f1': result['metrics']['mean_f1'],
                    'std_f1': result['metrics']['std_f1'],
                    'max_f1': result['metrics']['max_f1'],
                    'min_f1': result['metrics']['min_f1'],
                    'best_val_f1': result['metrics']['best_val_f1'],
                    'status': result['status']
                }
                row.update(result['params'])
                results_data.append(row)
            
            df = pd.DataFrame(results_data)
            df = df.sort_values('mean_f1', ascending=False)
            
            results_csv_path = self.results_dir / "search_results.csv"
            df.to_csv(results_csv_path, index=False)
            
            print(f"\n结果已保存到:")
            print(f"  JSON: {results_json_path}")
            print(f"  CSV: {results_csv_path}")
    
    def _save_study(self, study):
        """保存Optuna研究结果"""
        # 保存最佳参数
        best_params_path = self.results_dir / "best_params.json"
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        # 保存所有试验结果
        trials_data = []
        for trial in study.trials:
            trial_dict = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime_start': str(trial.datetime_start),
                'datetime_complete': str(trial.datetime_complete),
                'duration': str(trial.duration) if trial.duration else None
            }
            trials_data.append(trial_dict)
        
        trials_path = self.results_dir / "all_trials.json"
        with open(trials_path, 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        # 保存摘要
        summary = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'datetime_start': str(study.trials[0].datetime_start) if study.trials else None
        }
        
        summary_path = self.results_dir / "optimization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def print_best_results(self, top_k: int = 5):
        """打印当前最佳结果"""
        if not self.results:
            return
        
        # 按 mean_f1 排序
        sorted_results = sorted(
            [r for r in self.results if r['status'] == 'success'],
            key=lambda x: x['metrics']['mean_f1'],
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
            print(f"  Test F1: {result['metrics']['mean_f1']:.2f}%")
            print(f"  Val F1: {result['metrics']['best_val_f1']:.2f}%")
            print(f"  关键参数:")
            key_params = ['d_model', 'n_heads', 'n_layers', 'lr', 'dropout', 'batch_size']
            for key in key_params:
                if key in result['params']:
                    print(f"    {key}: {result['params'][key]}")
    
    def generate_report(self):
        """生成详细的搜索报告"""
        if not self.results:
            print("没有结果可生成报告")
            return
        
        report_path = self.results_dir / "search_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EEG通道分类器超参数搜索报告\n")
            f.write("="*80 + "\n\n")
            
            # 统计信息
            successful = [r for r in self.results if r['status'] == 'success']
            failed = [r for r in self.results if r['status'] == 'failed']
            
            f.write(f"总实验数: {len(self.results)}\n")
            f.write(f"成功: {len(successful)}\n")
            f.write(f"失败: {len(failed)}\n\n")
            
            if successful:
                # 最佳结果
                best_result = max(successful, key=lambda x: x['metrics']['mean_f1'])
                f.write("="*80 + "\n")
                f.write("最佳结果\n")
                f.write("="*80 + "\n")
                f.write(f"实验ID: {best_result['experiment_id']}\n")
                f.write(f"Test F1: {best_result['metrics']['mean_f1']:.2f}%\n")
                f.write(f"Val F1: {best_result['metrics']['best_val_f1']:.2f}%\n\n")
                f.write("最佳参数:\n")
                for key, value in best_result['params'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                # Top 10 结果
                sorted_results = sorted(successful, key=lambda x: x['metrics']['mean_f1'], reverse=True)
                f.write("="*80 + "\n")
                f.write("Top 10 结果\n")
                f.write("="*80 + "\n\n")
                
                for i, result in enumerate(sorted_results[:10], 1):
                    f.write(f"{i}. 实验 {result['experiment_id']}\n")
                    f.write(f"   Test F1: {result['metrics']['mean_f1']:.2f}%\n")
                    f.write(f"   Val F1: {result['metrics']['best_val_f1']:.2f}%\n")
                    f.write(f"   关键参数: ")
                    key_params = ['d_model', 'n_heads', 'n_layers', 'lr', 'dropout', 'batch_size']
                    param_str = ", ".join([f"{k}={result['params'][k]}" for k in key_params if k in result['params']])
                    f.write(param_str + "\n\n")
        
        print(f"\n详细报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='EEG通道分类器完整超参数搜索')
    
    # 基本参数
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录')
    parser.add_argument('--save_dir', type=str, default='comprehensive_search_results',
                       help='结果保存目录')
    
    # 搜索参数
    parser.add_argument('--search_type', type=str, default='grid',
                       choices=['grid', 'bayesian'],
                       help='搜索类型')
    parser.add_argument('--search_space', type=str, default='quick',
                       choices=['quick', 'medium', 'full'],
                       help='搜索空间大小')
    
    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--n_trials', type=int, default=50, help='贝叶斯搜索试验次数')
    parser.add_argument('--timeout', type=int, default=None, help='超时时间（秒）')
    parser.add_argument('--resume_from', type=int, default=0, help='从第几个实验开始')
    
    args = parser.parse_args()
    
    # 创建搜索器
    searcher = ComprehensiveHyperparameterSearch(
        data_root=args.data_root,
        base_save_dir=args.save_dir
    )
    
    if args.search_type == 'grid':
        # 网格搜索
        search_space = searcher.define_search_space(args.search_space)
        
        print(f"\n搜索空间:")
        for key, values in search_space.items():
            print(f"  {key}: {values}")
        
        results = searcher.run_grid_search(
            search_space=search_space,
            n_epochs=args.n_epochs,
            resume_from=args.resume_from
        )
        
        print(f"\n网格搜索完成! 共完成 {len(results)} 个实验")
        
    elif args.search_type == 'bayesian':
        # 贝叶斯搜索
        if not OPTUNA_AVAILABLE:
            print("\n错误: 需要安装Optuna库")
            print("请运行: pip install optuna plotly kaleido")
            sys.exit(1)
        
        study = searcher.run_bayesian_search(
            n_trials=args.n_trials,
            n_epochs=args.n_epochs,
            timeout=args.timeout
        )
        
        print(f"\n贝叶斯搜索完成!")
        print(f"最佳F1分数: {study.best_value:.2f}%")
        print(f"最佳参数: {study.best_params}")
    
    # 生成报告
    searcher.generate_report()


if __name__ == "__main__":
    main()
