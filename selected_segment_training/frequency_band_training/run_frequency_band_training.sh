#!/bin/bash
# 频段分离训练运行脚本

# ============================================
# 配置变量
# ============================================
DATA_ROOT="/mnt/hd1/dyf/dataset/EEG dataset_SUAT_processed_selected"
SAVE_DIR="checkpoints_frequency_band"
CONFIG_FILE="frequency_band_config.json"

echo "开始频段分离训练..."

# ============================================
# 1. 训练所有频段
# ============================================
echo "=== 训练所有频段 ==="
python run_all_frequency_bands.py \
    --data_root "$DATA_ROOT" \
    --save_dir "$SAVE_DIR" \
    --window_size 6.0 \
    --window_stride 3.0 \
    --batch_size 8 \
    --d_model 128 \
    --n_heads 8 \
    --n_layers 2 \
    --use_focal_loss \
    --use_class_weights \
    --n_epochs 30

# ============================================
# 2. 聚合结果
# ============================================
echo "=== 聚合频段结果 ==="
python aggregate_frequency_band_results.py \
    --results_dir "$SAVE_DIR" \
    --create_plots

# ============================================
# 3. 生成报告
# ============================================
echo "=== 生成训练报告 ==="
python -c "
import json
from pathlib import Path
from datetime import datetime

# 读取汇总结果
summary_file = Path('$SAVE_DIR') / 'training_summary.json'
if summary_file.exists():
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print('='*80)
    print('频段分离训练完成报告')
    print('='*80)
    print(f'训练时间: {summary.get(\"timestamp\", \"未知\")}')
    print(f'数据路径: {summary.get(\"data_root\", \"未知\")}')
    print(f'保存目录: {summary.get(\"base_save_dir\", \"未知\")}')
    
    if 'summary' in summary:
        s = summary['summary']
        print(f'成功频段: {len(s.get(\"successful_bands\", []))}')
        print(f'失败频段: {len(s.get(\"failed_bands\", []))}')
        print(f'最佳频段: {s.get(\"best_band\", \"未知\")} (F1: {s.get(\"best_f1\", 0):.2f}%)')
        print(f'总训练时间: {s.get(\"total_time\", \"未知\")}')
    
    print('='*80)
else:
    print('未找到汇总结果文件')
"

echo "频段分离训练完成！"
echo "结果保存在: $SAVE_DIR"
echo "查看详细报告: $SAVE_DIR/aggregated_results/frequency_band_report.txt"

