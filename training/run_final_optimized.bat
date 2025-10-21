@echo off
REM Final Optimized Multi-Label Training
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ========================================================================
echo FINAL OPTIMIZED Multi-Label Training
echo ========================================================================
echo.
echo Optimizations applied:
echo   [1] Channel Filtering: Only channels with ^>= 15 samples
echo   [2] Asymmetric Loss: gamma_neg=6.0 (强烈降低负类权重)
echo   [3] Weighted Sampling: 15x weight for positive samples
echo   [4] Larger Model: hidden_dim=512, layers=5
echo   [5] Higher Threshold: 0.45 (平衡Precision和Recall)
echo   [6] LR Warmup: 10 epochs warmup
echo.
echo Expected improvements:
echo   - Jaccard: 0.26 -^> 0.40-0.50
echo   - Precision: 0.27 -^> 0.40-0.50
echo   - More balanced P/R
echo ========================================================================
echo.

python training\train_multilabel_final.py ^
    --features_root E:\output\connectivity_features ^
    --labels_csv E:\output\connectivity_features\labels.csv ^
    --matrix_keys plv_alpha coherence_alpha wpli_alpha pearson_corr plv_beta ^
    --min_channel_samples 15 ^
    --hidden_dim 512 ^
    --num_layers 5 ^
    --dropout 0.2 ^
    --batch_size 8 ^
    --epochs 150 ^
    --lr 0.0003 ^
    --warmup_epochs 10 ^
    --weight_per_positive 15.0 ^
    --gamma_neg 6.0 ^
    --gamma_pos 0.0 ^
    --threshold 0.45 ^
    --device cuda ^
    --save_dir checkpoints_multilabel_final ^
    --seed 42

echo.
echo ========================================================================
echo Training Complete!
echo ========================================================================
echo.
echo Next steps:
echo   1. Check results in: checkpoints_multilabel_final\
echo   2. Run threshold optimization:
echo      python training\optimize_threshold.py --results_file checkpoints_multilabel_final\test_results.npz
echo.
pause

