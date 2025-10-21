@echo off
setlocal

REM 设置UTF-8编码
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ================================================================================
echo Ultimate Multi-Label Training (WITHOUT Curriculum Learning)
echo ================================================================================
echo.

REM 配置路径
set FEATURES_ROOT=E:\output\connectivity_features
set LABELS_CSV=E:\output\connectivity_features\labels.csv
set SAVE_DIR=checkpoints_ultimate_nocurriculum

REM 特征选择（只使用对拼接不敏感的特征）
REM 排除相位同步特征（PLV, PLI, wPLI）因为30s片段可能是拼接的
set MATRIX_KEYS=pearson spearman partial_corr

echo Configuration:
echo   Features Root: %FEATURES_ROOT%
echo   Labels CSV: %LABELS_CSV%
echo   Save Directory: %SAVE_DIR%
echo   Matrix Keys: %MATRIX_KEYS%
echo.

REM 训练参数
set HIDDEN_DIM=512
set NUM_LAYERS=6
set DROPOUT=0.15
set BATCH_SIZE=8
set EPOCHS=150
set LR=0.0002
set WARMUP_EPOCHS=15

REM 高级优化（不使用课程学习）
set MIN_CHANNEL_SAMPLES=15
set THRESHOLD=0.48
set WEIGHT_PER_POSITIVE=20.0
set BASE_GAMMA=3.0

echo Starting Training...
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
    --use_contrastive_pretrain ^
    --pretrain_epochs 50 ^
    --use_smote ^
    --smote_per_channel 50 ^
    --use_mixup ^
    --sampler_type multilevel ^
    --weight_per_positive %WEIGHT_PER_POSITIVE% ^
    --loss_type channel_adaptive ^
    --base_gamma %BASE_GAMMA% ^
    --threshold %THRESHOLD% ^
    --save_dir %SAVE_DIR% ^
    --device cuda

echo.
echo ================================================================================
echo Training Complete!
echo ================================================================================
echo Results saved to: %SAVE_DIR%
echo.

pause

