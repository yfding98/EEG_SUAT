@echo off
chcp 65001 >nul
setlocal

echo ================================================================================
echo EEG Connectivity Feature Extraction V2
echo - 5 second windows (no concatenation)
echo - Individual normalization
echo - Safe features only
echo ================================================================================
echo.

set INPUT_DIR=E:\DataSet\EEG\EEG dataset_SUAT_processed
set OUTPUT_DIR=E:\output\connectivity_features_v2
set PATTERN=*_merged_*.set
set WINDOW_SIZE=5

echo Configuration:
echo   Input:  %INPUT_DIR%
echo   Output: %OUTPUT_DIR%
echo   Pattern: %PATTERN%
echo   Window: %WINDOW_SIZE%s
echo   Visualization: Disabled (for batch speed)
echo.

python extract_connectivity_features_v2.py ^
    --input_dir "%INPUT_DIR%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --pattern "%PATTERN%" ^
    --window_size %WINDOW_SIZE% ^
    --no_visualize

echo.
echo ================================================================================
echo Extraction Complete!
echo ================================================================================
echo.
echo Results saved to: %OUTPUT_DIR%
echo.
echo Next steps:
echo 1. Run: python generate_labels_csv_v2.py
echo 2. Train: training\run_ultimate_nocurriculum.bat
echo.

pause

