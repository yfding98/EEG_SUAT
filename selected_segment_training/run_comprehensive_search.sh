#!/bin/bash
# 完整超参数搜索运行脚本

# ============================================
# 配置变量
# ============================================
DATA_ROOT="/mnt/hd1/dyf/dataset/EEG dataset_SUAT_processed_selected"
SAVE_DIR="comprehensive_search_results"

echo "开始EEG通道分类器完整超参数搜索..."

# ============================================
# 快速网格搜索（推荐首次使用）
# ============================================
echo "=== 快速网格搜索 ==="
python comprehensive_hyperparameter_search.py \
    --data_root "$DATA_ROOT" \
    --search_type "grid" \
    --search_space "quick" \
    --n_epochs 20 \
    --save_dir "$SAVE_DIR"

# ============================================
# 贝叶斯搜索（推荐用于正式实验）
# ============================================
echo "=== 贝叶斯搜索 ==="
python comprehensive_hyperparameter_search.py \
    --data_root "$DATA_ROOT" \
    --search_type "bayesian" \
    --n_trials 30 \
    --n_epochs 25 \
    --save_dir "$SAVE_DIR"

# ============================================
# 中等网格搜索（如果需要更全面的搜索）
# ============================================
# echo "=== 中等网格搜索 ==="
# python comprehensive_hyperparameter_search.py \
#     --data_root "$DATA_ROOT" \
#     --search_type "grid" \
#     --search_space "medium" \
#     --n_epochs 30 \
#     --save_dir "$SAVE_DIR"

# ============================================
# 完整网格搜索（计算量大，需要充足时间）
# ============================================
# echo "=== 完整网格搜索 ==="
# python comprehensive_hyperparameter_search.py \
#     --data_root "$DATA_ROOT" \
#     --search_type "grid" \
#     --search_space "full" \
#     --n_epochs 30 \
#     --save_dir "$SAVE_DIR"

# ============================================
# 大规模贝叶斯搜索
# ============================================
# echo "=== 大规模贝叶斯搜索 ==="
# python comprehensive_hyperparameter_search.py \
#     --data_root "$DATA_ROOT" \
#     --search_type "bayesian" \
#     --n_trials 100 \
#     --n_epochs 30 \
#     --timeout 86400 \
#     --save_dir "$SAVE_DIR"

echo "完整超参数搜索完成！"
echo "结果保存在: $SAVE_DIR"
