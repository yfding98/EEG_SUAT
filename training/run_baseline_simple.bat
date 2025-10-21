@echo off
setlocal

REM 设置UTF-8编码
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ================================================================================
echo Baseline Training - Simple Version (No Augmentation)
echo ================================================================================
echo.
echo 目标：排除增强和复杂策略的干扰，建立干净的基线
echo.
echo 关闭的功能：
echo   - SMOTE (避免伪样本)
echo   - Mixup (避免标签噪声)
echo   - 课程学习 (避免训练不稳定)
echo   - 对比学习预训练 (简化流程)
echo.
echo 保守的配置：
echo   - 单一特征: pearson (最稳定)
echo   - 简单采样: weight_per_positive=3.0 (从20.0降低)
echo   - 基础损失: Focal Loss (gamma=2.0)
echo   - 小模型: hidden_dim=256, layers=3
echo.

REM 配置路径
set FEATURES_ROOT=E:\output\connectivity_features
set LABELS_CSV=E:\output\connectivity_features\labels.csv
set SAVE_DIR=checkpoints_baseline_simple

REM 单一特征（最安全、最稳定）
set MATRIX_KEYS=pearson

echo Configuration:
echo   Features: %FEATURES_ROOT%
echo   Labels: %LABELS_CSV%
echo   Matrix: %MATRIX_KEYS%
echo   Output: %SAVE_DIR%
echo.

REM 基础参数（保守设置）
set HIDDEN_DIM=256
set NUM_LAYERS=3
set DROPOUT=0.2
set BATCH_SIZE=8
set EPOCHS=100
set LR=0.0001
set WARMUP_EPOCHS=10

REM 不平衡处理（保守）
set MIN_CHANNEL_SAMPLES=15
set THRESHOLD=0.40
set WEIGHT_PER_POSITIVE=3.0
set BASE_GAMMA=2.0

echo Starting Baseline Training...
echo.

python training\train_ultimate.py ^
    --features_root "%FEATURES_ROOT%" ^
    --labels_csv "%LABELS_CSV%" ^
    --matrix_keys %MATRIX_KEYS% ^
    --min_channel_samples %MIN_CHANNEL_SAMPLES% ^
    --hidden_dim %HIDDEN_DIM% ^
    --num_layers %NUM_LAYERS% ^
    --dropout %DROPOUT% ^
    --batch_size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --warmup_epochs %WARMUP_EPOCHS% ^
    --sampler_type multilevel ^
    --weight_per_positive %WEIGHT_PER_POSITIVE% ^
    --loss_type focal ^
    --base_gamma %BASE_GAMMA% ^
    --threshold %THRESHOLD% ^
    --save_dir %SAVE_DIR% ^
    --device cuda

echo.
echo ================================================================================
echo Baseline Training Complete!
echo ================================================================================
echo.
echo Results saved to: %SAVE_DIR%
echo.
echo 下一步分析：
echo 1. 检查 per-channel results 是否更均衡
echo 2. 对比 Jaccard/mAP 是否提升
echo 3. 如果基线就不好，说明特征问题
echo 4. 如果基线好，逐步加回优化
echo.

pause

