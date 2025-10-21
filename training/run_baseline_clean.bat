@echo off
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ================================================================================
echo Clean Baseline Training - 诊断性实验
echo ================================================================================
echo.
echo 实施的改进:
echo   1. 关闭 SMOTE 和 Mixup
echo   2. 保守采样权重 (3.0 代替 20.0)
echo   3. 自动阈值搜索
echo   4. 单一特征 (Pearson)
echo   5. 小模型 (256d, 3层)
echo.

set FEATURES_ROOT=E:\output\connectivity_features
set LABELS_CSV=E:\output\connectivity_features\labels.csv
set SAVE_DIR=checkpoints_baseline_clean

echo Configuration:
echo   Features: %FEATURES_ROOT%
echo   Matrix: pearson (单一最稳定特征)
echo   Model: 256d x 3 layers
echo   Weight: 3.0 (保守)
echo   Threshold: Auto-search
echo.

python training\train_baseline_clean.py ^
    --features_root "%FEATURES_ROOT%" ^
    --labels_csv "%LABELS_CSV%" ^
    --matrix_keys pearson ^
    --min_channel_samples 15 ^
    --hidden_dim 256 ^
    --num_layers 3 ^
    --dropout 0.2 ^
    --batch_size 8 ^
    --epochs 100 ^
    --lr 0.0001 ^
    --warmup_epochs 10 ^
    --weight_per_positive 3.0 ^
    --base_gamma 2.0 ^
    --threshold 0.40 ^
    --auto_threshold ^
    --save_dir "%SAVE_DIR%" ^
    --device cuda

echo.
echo ================================================================================
echo Training Complete!
echo ================================================================================
echo.
echo 分析要点:
echo 1. Jaccard 是否 ^> 0.30?
echo 2. Per-channel 是否更均衡?
echo 3. 是否还有很多通道 F1=0.00?
echo.
echo 如果基线效果好 (Jaccard ^> 0.30):
echo   - 说明特征本身可用
echo   - 逐步加回优化 (先加 multi-matrix, 再加采样策略)
echo.
echo 如果基线效果差 (Jaccard ^< 0.20):
echo   - 说明特征质量问题
echo   - 需要重新提取特征 (V2版本)
echo.

pause

