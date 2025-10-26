#!/bin/bash
# 基础EEG分类器超参数搜索脚本

# ============================================
# 配置变量
# ============================================
DATA_ROOT="/mnt/hd1/dyf/workspace/EEG_SUAT/EEG dataset_SUAT_processed_selected"
SAVE_DIR="basic_eeg_search_results"

echo "开始基础EEG分类器超参数搜索..."

# ============================================
# 快速网格搜索（推荐首次使用）
# ============================================
echo "=== 快速网格搜索 ==="
python universal_hyperparameter_search.py \
    --data_root "$DATA_ROOT" \
    --script_type "basic_eeg" \
    --search_type "grid" \
    --search_space "quick" \
    --n_folds 3 \
    --n_epochs 15 \
    --save_dir "$SAVE_DIR"

# ============================================
# 贝叶斯搜索（推荐用于正式实验）
# ============================================
echo "=== 贝叶斯搜索 ==="
python universal_hyperparameter_search.py \
    --data_root "$DATA_ROOT" \
    --script_type "basic_eeg" \
    --search_type "bayesian" \
    --n_trials 30 \
    --n_folds 3 \
    --n_epochs 20 \
    --save_dir "$SAVE_DIR"

# ============================================
# 中等网格搜索（如果需要更全面的搜索）
# ============================================
# echo "=== 中等网格搜索 ==="
# python universal_hyperparameter_search.py \
#     --data_root "$DATA_ROOT" \
#     --script_type "basic_eeg" \
#     --search_type "grid" \
#     --search_space "medium" \
#     --n_folds 3 \
#     --n_epochs 20 \
#     --save_dir "$SAVE_DIR"

echo "基础EEG分类器搜索完成！"
echo "结果保存在: $SAVE_DIR"
