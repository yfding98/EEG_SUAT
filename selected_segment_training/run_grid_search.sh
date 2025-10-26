#!/bin/bash
# 网格搜索运行脚本示例

# ============================================
# 配置变量
# ============================================
TRAIN_SCRIPT="train_channel_aware_kfold_optimized.py"
DATA_ROOT="/mnt/hd1/dyf/workspace/EEG_SUAT/EEG dataset_SUAT_processed_selected"
SAVE_DIR="grid_search_results"
N_FOLDS=3
N_EPOCHS=20

# ============================================
# 快速搜索（推荐首次使用）
# 组合数较少，适合快速验证
# ============================================
# python grid_search.py \
#     --data_root "$DATA_ROOT" \
#     --train_script "$TRAIN_SCRIPT" \
#     --save_dir "$SAVE_DIR" \
#     --search_type "quick" \
#     --n_folds 5 \
#     --n_epochs 30

# ============================================
# 中等搜索（推荐用于正式实验）
# 平衡搜索范围和时间成本
# ============================================
# python grid_search.py \
#     --data_root "$DATA_ROOT" \
#     --train_script "$TRAIN_SCRIPT" \
#     --save_dir "$SAVE_DIR" \
#     --search_type "medium" \
#     --n_folds 5 \
#     --n_epochs 30

# ============================================
# 完整搜索（计算量大，需要充足时间）
# 全面的超参数空间探索
# ============================================
python grid_search.py \
    --data_root "$DATA_ROOT" \
    --train_script "$TRAIN_SCRIPT" \
    --save_dir "$SAVE_DIR" \
    --search_type "full" \
    --n_folds "$N_FOLDS" \
    --n_epochs "$N_EPOCHS"

# ============================================
# 恢复中断的搜索
# 如果搜索中断，可以使用 --resume_from 参数继续
# ============================================
# python grid_search.py \
#     --data_root "$DATA_ROOT" \
#     --train_script "$TRAIN_SCRIPT" \
#     --save_dir "$SAVE_DIR" \
#     --search_type "medium" \
#     --n_folds 5 \
#     --n_epochs 30 \
#     --resume_from 10
