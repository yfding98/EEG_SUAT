#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bayesian_search.py

使用Optuna进行贝叶斯超参数优化
比网格搜索更高效，适合大规模超参数空间
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import sys

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


class BayesianSearchRunner:
    """贝叶斯优化搜索运行器"""
    
    def __init__(self, data_root, base_save_dir, train_script_path, script_type='auto'):
        if not OPTUNA_AVAILABLE:
            raise ImportError("需要安装Optuna: pip install optuna")
        
        self.data_root = data_root
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        self.train_script_path = train_script_path
        self.script_type = script_type
        
        # 结果保存路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = self.base_save_dir / f"bayesian_search_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"贝叶斯优化结果将保存到: {self.results_dir}")
        
        # 试验计数器
        self.trial_counter = 0
    
    def objective(self, trial, n_folds=5, n_epochs=30):
        """
        Optuna优化目标函数
        
        Args:
            trial: Optuna trial对象
            n_folds: K折数
            n_epochs: 训练轮数
            
        Returns:
            float: 优化目标值（mean_f1）
        """
        # 定义超参数搜索空间
        d_model_choices = [64, 128, 256, 512]
        d_model = trial.suggest_categorical('d_model', d_model_choices)
        
        # n_heads必须是d_model的因数
        valid_n_heads = [h for h in [4, 8, 16] if d_model % h == 0]
        n_heads = trial.suggest_categorical('n_heads', valid_n_heads)
        
        n_layers = trial.suggest_int('n_layers', 2, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
        
        window_size = trial.suggest_categorical('window_size', [1.0, 2.0, 3.0])
        window_stride = trial.suggest_categorical('window_stride', [0.5, 1.0, 1.5])
        
        use_iou_loss = trial.suggest_categorical('use_iou_loss', [False, True])
        
        if use_iou_loss:
            iou_weight = trial.suggest_float('iou_weight', 1.0, 2.5, step=0.5)
            iou_type = trial.suggest_categorical('iou_type', ['basic', 'focal', 'weighted'])
        else:
            iou_weight = 1.5
            iou_type = 'basic'
        
        # 构建参数字典
        params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'dropout': dropout,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'window_size': window_size,
            'window_stride': window_stride,
            'use_iou_loss': use_iou_loss,
            'iou_weight': iou_weight,
            'iou_type': iou_type
        }
        
        print(f"\n{'='*80}")
        print(f"Trial {self.trial_counter} (Optuna Trial {trial.number})")
        print(f"{'='*80}")
        print(f"参数配置:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # 运行训练
        mean_f1 = self._run_training(params, self.trial_counter, n_folds, n_epochs)
        
        self.trial_counter += 1
        
        return mean_f1
    
    def _run_training(self, params, experiment_id, n_folds, n_epochs):
        """运行单次训练实验"""
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
            '--save_dir', str(self.results_dir / f"trial_{experiment_id}")
        ]
        
        if params['use_iou_loss']:
            cmd.append('--use_iou_loss')
        
        try:
            print(f"\n执行训练...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"训练完成!")
            
            # 读取训练结果 - 支持多种训练脚本的结果格式
            result_files = []
            
            # 尝试查找基础EEG分类器的结果
            basic_eeg_results = list(self.results_dir.glob(
                f"trial_{experiment_id}/basic_eeg_*/final_results.json"
            ))
            if basic_eeg_results:
                result_files = basic_eeg_results
            
            # 尝试查找K折交叉验证的结果
            if not result_files:
                kfold_results = list(self.results_dir.glob(
                    f"trial_{experiment_id}/channel_aware_kfold_*/kfold_results.json"
                ))
                if kfold_results:
                    result_files = kfold_results
            
            if result_files:
                with open(result_files[0], 'r') as f:
                    training_results = json.load(f)
                
                # 处理不同的结果格式
                if 'test_metrics' in training_results:
                    # 基础EEG分类器格式
                    mean_f1 = training_results['test_metrics']['macro_f1']
                    print(f"结果: Test F1={mean_f1:.2f}%")
                elif 'mean_f1' in training_results:
                    # K折交叉验证格式
                    mean_f1 = training_results['mean_f1']
                    std_f1 = training_results.get('std_f1', 0)
                    print(f"结果: Mean F1={mean_f1:.2f}% ± {std_f1:.2f}%")
                else:
                    print("警告: 未知的结果格式")
                    return 0.0
                
                return mean_f1
            else:
                print("警告: 未找到结果文件")
                print(f"搜索路径: {self.results_dir}/trial_{experiment_id}/")
                return 0.0
                
        except subprocess.CalledProcessError as e:
            print(f"错误: 训练失败")
            print(f"错误信息: {e.stderr}")
            return 0.0
    
    def run_optimization(
        self,
        n_trials=50,
        n_folds=5,
        n_epochs=30,
        timeout=None,
        n_jobs=1
    ):
        """
        运行贝叶斯优化
        
        Args:
            n_trials: 试验次数
            n_folds: K折数
            n_epochs: 每个试验的训练轮数
            timeout: 超时时间（秒），None表示无限制
            n_jobs: 并行作业数（需要多个GPU）
            
        Returns:
            optuna.Study: 优化研究对象
        """
        print(f"\n{'='*80}")
        print(f"开始贝叶斯优化")
        print(f"{'='*80}")
        print(f"试验次数: {n_trials}")
        print(f"K折数: {n_folds}")
        print(f"每个试验训练轮数: {n_epochs}")
        print(f"并行作业数: {n_jobs}")
        if timeout:
            print(f"超时时间: {timeout}秒")
        
        # 创建研究
        study = optuna.create_study(
            direction='maximize',  # 最大化F1分数
            study_name='eeg_hyperparameter_optimization',
            sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
            pruner=optuna.pruners.MedianPruner(  # 提前终止表现不佳的试验
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        
        # 运行优化
        study.optimize(
            lambda trial: self.objective(trial, n_folds, n_epochs),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        # 保存研究结果
        self._save_study(study)
        
        # 生成可视化
        self._generate_visualizations(study)
        
        # 打印结果
        self._print_results(study)
        
        return study
    
    def _save_study(self, study):
        """保存研究结果"""
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
        
        print(f"\n结果已保存到:")
        print(f"  最佳参数: {best_params_path}")
        print(f"  所有试验: {trials_path}")
        print(f"  优化摘要: {summary_path}")
    
    def _generate_visualizations(self, study):
        """生成可视化图表"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            
            # 优化历史
            fig1 = plot_optimization_history(study)
            fig1.write_html(str(self.results_dir / "optimization_history.html"))
            
            # 参数重要性
            fig2 = plot_param_importances(study)
            fig2.write_html(str(self.results_dir / "param_importances.html"))
            
            # 参数切片图
            fig3 = plot_slice(study)
            fig3.write_html(str(self.results_dir / "param_slice.html"))
            
            # 参数等高线图（选择重要参数）
            if len(study.trials) > 10:
                try:
                    fig4 = plot_contour(study, params=['lr', 'd_model'])
                    fig4.write_html(str(self.results_dir / "param_contour.html"))
                except Exception as e:
                    print(f"生成等高线图失败: {e}")
            
            print(f"\n可视化图表已保存到:")
            print(f"  {self.results_dir}/optimization_history.html")
            print(f"  {self.results_dir}/param_importances.html")
            print(f"  {self.results_dir}/param_slice.html")
            
        except Exception as e:
            print(f"生成可视化时出错: {e}")
    
    def _print_results(self, study):
        """打印优化结果"""
        print(f"\n{'='*80}")
        print("优化完成!")
        print(f"{'='*80}")
        
        print(f"\n最佳试验: Trial {study.best_trial.number}")
        print(f"最佳F1分数: {study.best_value:.2f}%")
        
        print(f"\n最佳参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        print(f"\n优化统计:")
        print(f"  总试验数: {len(study.trials)}")
        print(f"  完成的试验: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"  修剪的试验: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"  失败的试验: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        
        # 打印Top 5试验
        print(f"\nTop 5 试验:")
        top_trials = sorted(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
            key=lambda t: t.value,
            reverse=True
        )[:5]
        
        for i, trial in enumerate(top_trials, 1):
            print(f"\n{i}. Trial {trial.number}")
            print(f"   F1: {trial.value:.2f}%")
            print(f"   关键参数: lr={trial.params['lr']:.6f}, "
                  f"d_model={trial.params['d_model']}, "
                  f"dropout={trial.params['dropout']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='EEG模型贝叶斯超参数优化')
    
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录')
    parser.add_argument('--train_script', type=str,
                        default='train_channel_aware_kfold_optimized.py',
                        help='训练脚本路径')
    parser.add_argument('--save_dir', type=str,
                        default='bayesian_search_results',
                        help='结果保存目录')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='试验次数')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='K折交叉验证折数')
    parser.add_argument('--n_epochs', type=int, default=30,
                        help='每个试验的训练轮数')
    parser.add_argument('--timeout', type=int, default=None,
                        help='超时时间（秒）')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='并行作业数（需要多个GPU）')
    
    args = parser.parse_args()
    
    if not OPTUNA_AVAILABLE:
        print("\n错误: 需要安装Optuna库")
        print("请运行: pip install optuna plotly kaleido")
        sys.exit(1)
    
    # 创建贝叶斯搜索运行器
    runner = BayesianSearchRunner(
        data_root=args.data_root,
        base_save_dir=args.save_dir,
        train_script_path=args.train_script
    )
    
    # 运行优化
    study = runner.run_optimization(
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        n_epochs=args.n_epochs,
        timeout=args.timeout,
        n_jobs=args.n_jobs
    )
    
    print(f"\n优化完成! 结果已保存到: {runner.results_dir}")


if __name__ == "__main__":
    main()

