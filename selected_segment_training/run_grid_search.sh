#!/bin/bash
# 网格搜索运行脚本示例

# ============================================
# 快速搜索（推荐首次使用）
# 组合数较少，适合快速验证
# ============================================
# python grid_search.py \
#     --data_root "/mnt/hd1/dyf/workspace/EEG_SUAT/EEG dataset_SUAT_processed_selected" \
#     --train_script "train_channel_aware_kfold_optimized.py" \
#     --save_dir "grid_search_results" \
#     --search_type "quick" \
#     --n_folds 5 \
#     --n_epochs 30

# ============================================
# 中等搜索（推荐用于正式实验）
# 平衡搜索范围和时间成本
# ============================================
# python grid_search.py \
#     --data_root "/mnt/hd1/dyf/workspace/EEG_SUAT/EEG dataset_SUAT_processed_selected" \
#     --train_script "train_channel_aware_kfold_optimized.py" \
#     --save_dir "grid_search_results" \
#     --search_type "medium" \
#     --n_folds 5 \
#     --n_epochs 30

# ============================================
# 完整搜索（计算量大，需要充足时间）
# 全面的超参数空间探索
# ============================================
python grid_search.py \
    --data_root "/mnt/hd1/dyf/workspace/EEG_SUAT/EEG dataset_SUAT_processed_selected" \
    --train_script "train_channel_aware_kfold_optimized.py" \
    --save_dir "grid_search_results" \
    --search_type "full" \
    --n_folds 3 \
    --n_epochs 20

# ============================================
# 恢复中断的搜索
# 如果搜索中断，可以使用 --resume_from 参数继续
# ============================================
# python grid_search.py \
#     --data_root "/mnt/hd1/dyf/workspace/EEG_SUAT/EEG dataset_SUAT_processed_selected" \
#     --train_script "train_channel_aware_kfold_optimized.py" \
#     --save_dir "grid_search_results" \
#     --search_type "medium" \
#     --n_folds 5 \
#     --n_epochs 30 \
#     --resume_from 10

