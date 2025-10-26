#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_frequency_band_results.py

频段结果聚合脚本
聚合所有频段的训练结果，生成综合报告
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class FrequencyBandResultAggregator:
    """频段结果聚合器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        self.results = {}
        self.summary = {}
        
    def load_results(self):
        """加载所有频段的结果"""
        print("加载频段训练结果...")
        
        for band in self.frequency_bands:
            band_results = self._load_band_results(band)
            if band_results:
                self.results[band] = band_results
                print(f"  {band} 频段: {len(band_results)} 个结果")
            else:
                print(f"  警告: {band} 频段未找到结果")
        
        print(f"总共加载了 {len(self.results)} 个频段的结果")
    
    def _load_band_results(self, frequency_band: str) -> List[Dict]:
        """加载特定频段的结果"""
        band_results = []
        
        # 查找该频段的所有结果目录
        band_dirs = list(self.results_dir.glob(f"{frequency_band}_*"))
        
        for band_dir in band_dirs:
            if not band_dir.is_dir():
                continue
            
            # 查找结果文件
            result_file = band_dir / 'final_results.json'
            config_file = band_dir / 'config.json'
            
            if result_file.exists() and config_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                    
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    # 合并结果和配置
                    combined_result = {
                        'frequency_band': frequency_band,
                        'experiment_dir': str(band_dir),
                        'timestamp': band_dir.name.split('_')[-1],
                        **result_data,
                        'config': config_data
                    }
                    
                    band_results.append(combined_result)
                    
                except Exception as e:
                    print(f"  警告: 加载 {band_dir} 失败: {e}")
                    continue
        
        # 按测试F1分数排序
        band_results.sort(key=lambda x: x.get('test_metrics', {}).get('macro_f1', 0), reverse=True)
        
        return band_results
    
    def generate_summary(self):
        """生成汇总报告"""
        print("\n生成汇总报告...")
        
        summary_data = []
        
        for band, results in self.results.items():
            if not results:
                continue
            
            # 获取最佳结果
            best_result = results[0]
            test_metrics = best_result.get('test_metrics', {})
            config = best_result.get('config', {})
            
            summary_data.append({
                'frequency_band': band,
                'best_test_f1': test_metrics.get('macro_f1', 0),
                'best_test_precision': test_metrics.get('macro_precision', 0),
                'best_test_recall': test_metrics.get('macro_recall', 0),
                'best_test_micro_f1': test_metrics.get('micro_f1', 0),
                'best_val_f1': best_result.get('best_val_f1', 0),
                'd_model': config.get('d_model', 0),
                'n_heads': config.get('n_heads', 0),
                'n_layers': config.get('n_layers', 0),
                'dropout': config.get('dropout', 0),
                'batch_size': config.get('batch_size', 0),
                'lr': config.get('lr', 0),
                'n_epochs': config.get('n_epochs', 0),
                'experiment_dir': best_result.get('experiment_dir', ''),
                'timestamp': best_result.get('timestamp', '')
            })
        
        self.summary = pd.DataFrame(summary_data)
        
        # 按测试F1分数排序
        self.summary = self.summary.sort_values('best_test_f1', ascending=False)
        
        print("汇总报告生成完成")
        return self.summary
    
    def save_results(self, output_dir: str = None):
        """保存结果"""
        if output_dir is None:
            output_dir = self.results_dir / 'aggregated_results'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存汇总表格
        summary_path = output_dir / 'frequency_band_summary.csv'
        self.summary.to_csv(summary_path, index=False)
        print(f"汇总表格保存到: {summary_path}")
        
        # 保存详细结果
        detailed_results = {}
        for band, results in self.results.items():
            detailed_results[band] = results
        
        detailed_path = output_dir / 'detailed_results.json'
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"详细结果保存到: {detailed_path}")
        
        # 生成报告
        self._generate_report(output_dir)
        
        return output_dir
    
    def _generate_report(self, output_dir: Path):
        """生成详细报告"""
        report_path = output_dir / 'frequency_band_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("频段分离训练结果报告\n")
            f.write("="*80 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"结果目录: {self.results_dir}\n\n")
            
            # 总体统计
            f.write("="*80 + "\n")
            f.write("总体统计\n")
            f.write("="*80 + "\n")
            f.write(f"成功训练的频段数: {len(self.results)}\n")
            f.write(f"总实验数: {sum(len(results) for results in self.results.values())}\n\n")
            
            # 各频段最佳结果
            f.write("="*80 + "\n")
            f.write("各频段最佳结果\n")
            f.write("="*80 + "\n\n")
            
            for _, row in self.summary.iterrows():
                f.write(f"频段: {row['frequency_band']}\n")
                f.write(f"  测试F1: {row['best_test_f1']:.2f}%\n")
                f.write(f"  测试Precision: {row['best_test_precision']:.2f}%\n")
                f.write(f"  测试Recall: {row['best_test_recall']:.2f}%\n")
                f.write(f"  测试Micro F1: {row['best_test_micro_f1']:.2f}%\n")
                f.write(f"  验证F1: {row['best_val_f1']:.2f}%\n")
                f.write(f"  模型参数: d_model={row['d_model']}, n_heads={row['n_heads']}, n_layers={row['n_layers']}\n")
                f.write(f"  训练参数: batch_size={row['batch_size']}, lr={row['lr']}, epochs={row['n_epochs']}\n")
                f.write(f"  实验目录: {row['experiment_dir']}\n\n")
            
            # 频段性能对比
            f.write("="*80 + "\n")
            f.write("频段性能对比\n")
            f.write("="*80 + "\n")
            
            if len(self.summary) > 0:
                best_band = self.summary.iloc[0]
                worst_band = self.summary.iloc[-1]
                
                f.write(f"最佳频段: {best_band['frequency_band']} (F1: {best_band['best_test_f1']:.2f}%)\n")
                f.write(f"最差频段: {worst_band['frequency_band']} (F1: {worst_band['best_test_f1']:.2f}%)\n")
                f.write(f"性能差异: {best_band['best_test_f1'] - worst_band['best_test_f1']:.2f}%\n\n")
                
                # 频段排名
                f.write("频段排名 (按测试F1分数):\n")
                for i, (_, row) in enumerate(self.summary.iterrows(), 1):
                    f.write(f"  {i}. {row['frequency_band']}: {row['best_test_f1']:.2f}%\n")
        
        print(f"详细报告保存到: {report_path}")
    
    def create_visualizations(self, output_dir: Path):
        """创建可视化图表"""
        if self.summary.empty:
            print("没有数据可以可视化")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('频段分离训练结果分析', fontsize=16)
        
        # 1. 各频段F1分数对比
        ax1 = axes[0, 0]
        bands = self.summary['frequency_band']
        f1_scores = self.summary['best_test_f1']
        bars = ax1.bar(bands, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_title('各频段测试F1分数')
        ax1.set_ylabel('F1分数 (%)')
        ax1.set_ylim(0, 100)
        
        # 添加数值标签
        for bar, score in zip(bars, f1_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{score:.1f}%', ha='center', va='bottom')
        
        # 2. 频段性能分布
        ax2 = axes[0, 1]
        ax2.scatter(self.summary['best_test_precision'], self.summary['best_test_recall'], 
                   c=range(len(self.summary)), s=100, alpha=0.7)
        ax2.set_xlabel('Precision (%)')
        ax2.set_ylabel('Recall (%)')
        ax2.set_title('Precision vs Recall')
        
        # 添加频段标签
        for i, band in enumerate(bands):
            ax2.annotate(band, (self.summary.iloc[i]['best_test_precision'], 
                               self.summary.iloc[i]['best_test_recall']))
        
        # 3. 模型参数对比
        ax3 = axes[1, 0]
        model_sizes = self.summary['d_model'] * self.summary['n_heads'] * self.summary['n_layers']
        ax3.scatter(model_sizes, f1_scores, s=100, alpha=0.7)
        ax3.set_xlabel('模型复杂度 (d_model × n_heads × n_layers)')
        ax3.set_ylabel('F1分数 (%)')
        ax3.set_title('模型复杂度 vs 性能')
        
        # 4. 训练参数热力图
        ax4 = axes[1, 1]
        heatmap_data = self.summary[['d_model', 'n_heads', 'n_layers', 'dropout', 'batch_size']].T
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='viridis', ax=ax4)
        ax4.set_title('训练参数热力图')
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = output_dir / 'frequency_band_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表保存到: {plot_path}")
    
    def get_best_models(self, top_k: int = 3) -> List[Dict]:
        """获取最佳模型"""
        if self.summary.empty:
            return []
        
        best_models = []
        for i, (_, row) in enumerate(self.summary.head(top_k).iterrows()):
            best_models.append({
                'rank': i + 1,
                'frequency_band': row['frequency_band'],
                'test_f1': row['best_test_f1'],
                'experiment_dir': row['experiment_dir'],
                'config': {
                    'd_model': row['d_model'],
                    'n_heads': row['n_heads'],
                    'n_layers': row['n_layers'],
                    'dropout': row['dropout'],
                    'batch_size': row['batch_size'],
                    'lr': row['lr']
                }
            })
        
        return best_models


def main():
    parser = argparse.ArgumentParser(description='频段结果聚合')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='结果目录')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--create_plots', action='store_true',
                       help='创建可视化图表')
    
    args = parser.parse_args()
    
    # 创建聚合器
    aggregator = FrequencyBandResultAggregator(args.results_dir)
    
    # 加载结果
    aggregator.load_results()
    
    # 生成汇总
    summary = aggregator.generate_summary()
    
    if summary.empty:
        print("没有找到任何结果")
        return
    
    # 打印汇总
    print("\n" + "="*80)
    print("频段训练结果汇总")
    print("="*80)
    print(summary.to_string(index=False))
    
    # 保存结果
    output_dir = aggregator.save_results(args.output_dir)
    
    # 创建可视化
    if args.create_plots:
        try:
            aggregator.create_visualizations(output_dir)
        except Exception as e:
            print(f"创建可视化失败: {e}")
    
    # 获取最佳模型
    best_models = aggregator.get_best_models(3)
    print("\n" + "="*80)
    print("最佳模型 (Top 3)")
    print("="*80)
    for model in best_models:
        print(f"{model['rank']}. {model['frequency_band']} 频段: F1={model['test_f1']:.2f}%")
        print(f"   配置: d_model={model['config']['d_model']}, n_heads={model['config']['n_heads']}")
        print(f"   目录: {model['experiment_dir']}")
        print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--results_dir', 'checkpoints_frequency_band',
            '--create_plots'
        ])
    main()

