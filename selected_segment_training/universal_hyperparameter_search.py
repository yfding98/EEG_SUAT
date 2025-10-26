#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
universal_hyperparameter_search.py

通用超参数搜索框架
支持多种训练脚本和搜索策略
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import sys
import os
from typing import Dict, List, Any, Optional
import itertools

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
    print("警告: Optuna未安装。请运行: pip install optuna")


class TrainingScriptConfig:
    """训练脚本配置类"""
    
    def __init__(self, script_name: str, script_path: str, result_patterns: List[str], 
                 required_params: List[str], optional_params: List[str]):
        self.script_name = script_name
        self.script_path = script_path
        self.result_patterns = result_patterns  # 结果文件匹配模式
        self.required_params = required_params  # 必需参数
        self.optional_params = optional_params  # 可选参数
    
    def build_command(self, params: Dict[str, Any], data_root: str, save_dir: str, 
                     n_folds: int = 5, n_epochs: int = 30) -> List[str]:
        """构建训练命令"""
        cmd = ['python', self.script_path, '--data_root', data_root]
        
        # 添加必需参数
        for param in self.required_params:
            if param in params:
                cmd.extend([f'--{param}', str(params[param])])
        
        # 添加可选参数
        for param in self.optional_params:
            if param in params:
                if isinstance(params[param], bool):
                    if params[param]:
                        cmd.append(f'--{param}')
                else:
                    cmd.extend([f'--{param}', str(params[param])])
        
        # 添加训练特定参数
        if 'n_folds' in self.required_params or 'n_folds' in self.optional_params:
            cmd.extend(['--n_folds', str(n_folds)])
        if 'n_epochs' in self.required_params or 'n_epochs' in self.optional_params:
            cmd.extend(['--n_epochs', str(n_epochs)])
        
        # 添加保存目录
        cmd.extend(['--save_dir', save_dir])
        
        return cmd
    
    def find_result_file(self, save_dir: str, experiment_id: int) -> Optional[str]:
        """查找结果文件"""
        save_path = Path(save_dir)
        
        for pattern in self.result_patterns:
            # 替换占位符
            pattern = pattern.replace('{experiment_id}', str(experiment_id))
            pattern = pattern.replace('{save_dir}', str(save_path))
            
            # 查找匹配的文件
            result_files = list(save_path.glob(pattern))
            if result_files:
                return str(result_files[0])
        
        return None
    
    def parse_result(self, result_file: str) -> Dict[str, float]:
        """解析结果文件"""
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # 根据不同的结果格式解析
            if 'test_metrics' in data:
                # 基础EEG分类器格式
                return {
                    'mean_f1': data['test_metrics']['macro_f1'],
                    'std_f1': 0.0,
                    'max_f1': data['test_metrics']['macro_f1'],
                    'min_f1': data['test_metrics']['macro_f1']
                }
            elif 'mean_f1' in data:
                # K折交叉验证格式
                return {
                    'mean_f1': data['mean_f1'],
                    'std_f1': data.get('std_f1', 0.0),
                    'max_f1': data.get('max_f1', data['mean_f1']),
                    'min_f1': data.get('min_f1', data['mean_f1'])
                }
            elif 'best_val_f1' in data:
                # 简单训练格式
                return {
                    'mean_f1': data['best_val_f1'],
                    'std_f1': 0.0,
                    'max_f1': data['best_val_f1'],
                    'min_f1': data['best_val_f1']
                }
            else:
                print(f"警告: 未知的结果格式: {result_file}")
                return {'mean_f1': 0.0, 'std_f1': 0.0, 'max_f1': 0.0, 'min_f1': 0.0}
                
        except Exception as e:
            print(f"错误: 解析结果文件失败: {e}")
            return {'mean_f1': 0.0, 'std_f1': 0.0, 'max_f1': 0.0, 'min_f1': 0.0}


# 预定义的训练脚本配置
TRAINING_SCRIPTS = {
    'basic_eeg': TrainingScriptConfig(
        script_name='基础EEG分类器',
        script_path='train_basic_eeg_classifier.py',
        result_patterns=[
            '{save_dir}/basic_eeg_*/final_results.json'
        ],
        required_params=['d_model', 'n_heads', 'n_layers', 'dropout', 'batch_size', 'lr', 'weight_decay'],
        optional_params=['use_focal_loss', 'use_class_weights', 'focal_alpha', 'focal_gamma', 'early_stopping_patience']
    ),
    
    'channel_aware_kfold': TrainingScriptConfig(
        script_name='通道感知K折分类器',
        script_path='train_channel_aware_kfold_optimized.py',
        result_patterns=[
            '{save_dir}/channel_aware_kfold_*/kfold_results.json'
        ],
        required_params=['d_model', 'n_heads', 'n_layers', 'dropout', 'batch_size', 'lr', 'weight_decay', 'window_size', 'window_stride'],
        optional_params=['use_iou_loss', 'iou_weight', 'iou_type', 'n_folds', 'n_epochs']
    ),
    
    'channel_aware_simple': TrainingScriptConfig(
        script_name='通道感知简单分类器',
        script_path='train_channel_aware.py',
        result_patterns=[
            '{save_dir}/channel_aware_*/final_results.json'
        ],
        required_params=['d_model', 'n_heads', 'n_layers', 'dropout', 'batch_size', 'lr', 'weight_decay'],
        optional_params=['use_class_weights', 'early_stopping_patience']
    )
}


class UniversalHyperparameterSearch:
    """通用超参数搜索类"""
    
    def __init__(self, data_root: str, script_config: TrainingScriptConfig, base_save_dir: str):
        self.data_root = data_root
        self.script_config = script_config
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果保存路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = self.base_save_dir / f"{script_config.script_name}_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"超参数搜索结果将保存到: {self.results_dir}")
        
        # 试验计数器
        self.trial_counter = 0
        self.results = []
    
    def define_search_space(self, search_type: str = 'quick') -> Dict[str, List]:
        """定义搜索空间"""
        if search_type == 'quick':
            return {
                'd_model': [64, 128],
                'n_heads': [4, 8],
                'n_layers': [2],
                'dropout': [0.3],
                'batch_size': [8, 16],
                'lr': [0.0001, 0.001],
                'weight_decay': [0.01],
                'window_size': [3.0, 6.0],
                'window_stride': [1.0, 3.0]
            }
        elif search_type == 'medium':
            return {
                'd_model': [64, 128, 256],
                'n_heads': [4, 8],
                'n_layers': [2, 3],
                'dropout': [0.2, 0.3, 0.4],
                'batch_size': [8, 16, 32],
                'lr': [0.00001, 0.0001, 0.001],
                'weight_decay': [0.001, 0.01, 0.1],
                'window_size': [2.0, 3.0, 6.0],
                'window_stride': [1.0, 2.0, 3.0]
            }
        elif search_type == 'full':
            return {
                'd_model': [64, 128, 256, 512],
                'n_heads': [4, 8, 16],
                'n_layers': [2, 3, 4],
                'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                'batch_size': [4, 8, 16, 32],
                'lr': [0.00001, 0.00005, 0.0001, 0.0005, 0.001],
                'weight_decay': [0.0, 0.001, 0.01, 0.1],
                'window_size': [1.0, 2.0, 3.0, 6.0],
                'window_stride': [0.5, 1.0, 2.0, 3.0]
            }
        else:
            raise ValueError(f"Unknown search_type: {search_type}")
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """验证参数组合"""
        # d_model 必须能被 n_heads 整除
        if 'd_model' in params and 'n_heads' in params:
            if params['d_model'] % params['n_heads'] != 0:
                return False
        
        return True
    
    def run_training(self, params: Dict[str, Any], experiment_id: int, 
                    n_folds: int = 5, n_epochs: int = 30) -> Dict[str, Any]:
        """运行单次训练"""
        print(f"\n{'='*80}")
        print(f"实验 {experiment_id} - {self.script_config.script_name}")
        print(f"{'='*80}")
        print(f"参数配置:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # 构建命令
        save_dir = str(self.results_dir / f"exp_{experiment_id}")
        cmd = self.script_config.build_command(params, self.data_root, save_dir, n_folds, n_epochs)
        
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
            result_file = self.script_config.find_result_file(save_dir, experiment_id)
            
            if result_file:
                print(f"找到结果文件: {result_file}")
                metrics = self.script_config.parse_result(result_file)
                
                print(f"结果: Mean F1={metrics['mean_f1']:.2f}% ± {metrics['std_f1']:.2f}%")
                
                return {
                    'experiment_id': experiment_id,
                    'params': params,
                    'metrics': metrics,
                    'status': 'success',
                    'result_file': result_file
                }
            else:
                print("警告: 未找到结果文件")
                return {
                    'experiment_id': experiment_id,
                    'params': params,
                    'metrics': {'mean_f1': 0.0, 'std_f1': 0.0, 'max_f1': 0.0, 'min_f1': 0.0},
                    'status': 'no_results'
                }
                
        except subprocess.CalledProcessError as e:
            print(f"错误: 训练失败")
            print(f"错误信息: {e.stderr}")
            return {
                'experiment_id': experiment_id,
                'params': params,
                'metrics': {'mean_f1': 0.0, 'std_f1': 0.0, 'max_f1': 0.0, 'min_f1': 0.0},
                'status': 'failed',
                'error': str(e)
            }
    
    def run_grid_search(self, search_space: Dict[str, List], n_folds: int = 5, 
                       n_epochs: int = 30, resume_from: int = 0) -> List[Dict[str, Any]]:
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
        print(f"训练脚本: {self.script_config.script_name}")
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
    
    def run_bayesian_search(self, n_trials: int = 50, n_folds: int = 5, 
                           n_epochs: int = 30, timeout: Optional[int] = None) -> Any:
        """运行贝叶斯搜索"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("需要安装Optuna: pip install optuna")
        
        def objective(trial):
            # 定义超参数搜索空间
            params = {}
            
            # 基础参数
            d_model_choices = [64, 128, 256, 512]
            d_model = trial.suggest_categorical('d_model', d_model_choices)
            
            # n_heads必须是d_model的因数
            valid_n_heads = [h for h in [4, 8, 16] if d_model % h == 0]
            n_heads = trial.suggest_categorical('n_heads', valid_n_heads)
            
            params.update({
                'd_model': d_model,
                'n_heads': n_heads,
                'n_layers': trial.suggest_int('n_layers', 2, 4),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
                'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
                'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
                'window_size': trial.suggest_categorical('window_size', [1.0, 2.0, 3.0, 6.0]),
                'window_stride': trial.suggest_categorical('window_stride', [0.5, 1.0, 2.0, 3.0])
            })
            
            # 运行训练
            result = self.run_training(params, self.trial_counter, n_folds, n_epochs)
            self.trial_counter += 1
            
            return result['metrics']['mean_f1']
        
        # 创建研究
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{self.script_config.script_name}_optimization',
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
        
        print(f"\n结果已保存到: {results_json_path}")
    
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
            print(f"  Mean F1: {result['metrics']['mean_f1']:.2f}% ± {result['metrics']['std_f1']:.2f}%")
            print(f"  参数:")
            for key, value in result['params'].items():
                print(f"    {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='通用超参数搜索')
    
    # 基本参数
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录')
    parser.add_argument('--script_type', type=str, required=True, 
                       choices=list(TRAINING_SCRIPTS.keys()),
                       help='训练脚本类型')
    parser.add_argument('--save_dir', type=str, default='hyperparameter_search_results',
                       help='结果保存目录')
    
    # 搜索参数
    parser.add_argument('--search_type', type=str, default='grid',
                       choices=['grid', 'bayesian'],
                       help='搜索类型')
    parser.add_argument('--search_space', type=str, default='quick',
                       choices=['quick', 'medium', 'full'],
                       help='搜索空间大小')
    
    # 训练参数
    parser.add_argument('--n_folds', type=int, default=5, help='K折数')
    parser.add_argument('--n_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--n_trials', type=int, default=50, help='贝叶斯搜索试验次数')
    parser.add_argument('--timeout', type=int, default=None, help='超时时间（秒）')
    parser.add_argument('--resume_from', type=int, default=0, help='从第几个实验开始')
    
    args = parser.parse_args()
    
    # 获取训练脚本配置
    script_config = TRAINING_SCRIPTS[args.script_type]
    
    print(f"使用训练脚本: {script_config.script_name}")
    print(f"脚本路径: {script_config.script_path}")
    
    # 创建搜索器
    searcher = UniversalHyperparameterSearch(
        data_root=args.data_root,
        script_config=script_config,
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
            n_folds=args.n_folds,
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
            n_folds=args.n_folds,
            n_epochs=args.n_epochs,
            timeout=args.timeout
        )
        
        print(f"\n贝叶斯搜索完成!")
        print(f"最佳F1分数: {study.best_value:.2f}%")
        print(f"最佳参数: {study.best_params}")


if __name__ == "__main__":
    main()
