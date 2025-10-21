@echo off
REM Improved Multi-Label Training Script for Imbalanced Data
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ============================================================
echo Improved Multi-Label Training (Optimized for Imbalanced Data)
echo ============================================================
echo.
echo Optimizations:
echo   - Focal Loss (gamma=2.5)
echo   - Weighted Sampling (oversample positive samples)
echo   - Dynamic Threshold Finding
echo   - Two-Stage Training
echo   - Larger Model (hidden=256, layers=4)
echo ============================================================
echo.

python training\train_multilabel_improved.py ^
    --features_root E:\output\connectivity_features ^
    --labels_csv E:\output\connectivity_features\labels.csv ^
    --matrix_keys plv_alpha coherence_alpha wpli_alpha ^
    --loss_type focal ^
    --focal_gamma 2.5 ^
    --focal_alpha 0.25 ^
    --use_weighted_sampler ^
    --find_best_threshold ^
    --two_stage ^
    --stage1_epochs 30 ^
    --hidden_dim 256 ^
    --num_layers 4 ^
    --dropout 0.3 ^
    --batch_size 16 ^
    --epochs 100 ^
    --lr 0.0005 ^
    --threshold 0.3 ^
    --device cuda ^
    --save_dir checkpoints_multilabel_improved ^
    --seed 42

echo.
echo ============================================================
echo Training Complete!
echo ============================================================
pause

