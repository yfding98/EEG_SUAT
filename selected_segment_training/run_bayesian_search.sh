
#!/bin/bash
# 贝叶斯优化运行脚本示例

# ============================================
# 配置变量
# ============================================
TRAIN_SCRIPT="train_basic_eeg_classifier.py"
DATA_ROOT="/mnt/hd1/dyf/workspace/EEG_SUAT/EEG dataset_SUAT_processed_selected"
SAVE_DIR="bayesian_search_results"
N_TRIALS=50
N_FOLDS=5
N_EPOCHS=30

# ============================================
# 基础贝叶斯优化（推荐）
# 使用Optuna的TPE算法，比网格搜索更高效
# ============================================
python bayesian_search.py \
    --data_root "$DATA_ROOT" \
    --train_script "$TRAIN_SCRIPT" \
    --save_dir "$SAVE_DIR" \
    --n_trials "$N_TRIALS" \
    --n_folds "$N_FOLDS" \
    --n_epochs "$N_EPOCHS"

# ============================================
# 快速测试（少量试验）
# ============================================
# python bayesian_search.py \
#     --data_root "$DATA_ROOT" \
#     --train_script "$TRAIN_SCRIPT" \
#     --save_dir "$SAVE_DIR" \
#     --n_trials 10 \
#     --n_folds 3 \
#     --n_epochs 15

# ============================================
# 大规模优化（100次试验）
# ============================================
# python bayesian_search.py \
#     --data_root "$DATA_ROOT" \
#     --train_script "$TRAIN_SCRIPT" \
#     --save_dir "$SAVE_DIR" \
#     --n_trials 100 \
#     --n_folds 5 \
#     --n_epochs 30

# ============================================
# 带超时的优化（例如运行24小时）
# ============================================
# python bayesian_search.py \
#     --data_root "$DATA_ROOT" \
#     --train_script "$TRAIN_SCRIPT" \
#     --save_dir "$SAVE_DIR" \
#     --n_trials 1000 \
#     --timeout 86400 \
#     --n_folds 5 \
#     --n_epochs 30

# ============================================
# 并行优化（需要多个GPU）
# 注意：需要设置CUDA_VISIBLE_DEVICES来指定不同的GPU
# ============================================
# python bayesian_search.py \
#     --data_root "$DATA_ROOT" \
#     --train_script "$TRAIN_SCRIPT" \
#     --save_dir "$SAVE_DIR" \
#     --n_trials 100 \
#     --n_folds 5 \
#     --n_epochs 30 \
#     --n_jobs 2