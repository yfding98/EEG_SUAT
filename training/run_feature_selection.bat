@echo off
REM Feature Selection using LightGBM
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ============================================================
echo Feature Selection for Connectivity Matrices
echo ============================================================
echo.
echo This will:
echo   1. Load connectivity matrices from NPZ files
echo   2. Train LightGBM models for each channel
echo   3. Analyze feature importance
echo   4. Recommend top-k most important matrices
echo.
echo Estimated time: 5-10 minutes
echo ============================================================
echo.

python training\quick_feature_selection.py ^
    --features_root E:\output\connectivity_features ^
    --labels_csv E:\output\connectivity_features\labels.csv ^
    --max_samples 300 ^
    --top_k 5 ^
    --output_dir feature_selection_quick ^
    --seed 42

echo.
echo ============================================================
echo Feature Selection Complete!
echo ============================================================
echo.
echo Check the results in: feature_selection_quick\
echo.
echo Recommended matrix keys are saved in:
echo   feature_selection_quick\recommended_keys.txt
echo.
echo Use them in training with:
echo   --matrix_keys (paste from recommended_keys.txt)
echo.
pause

