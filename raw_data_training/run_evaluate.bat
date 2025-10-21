@echo off
REM 评估脚本 - Windows批处理文件

echo ======================================
echo 模型评估
echo ======================================

REM 激活conda环境（如果需要）
REM call conda activate your_env_name

REM 检查是否提供了检查点路径
if "%1"=="" (
    echo 使用方法: run_evaluate.bat ^<checkpoint_path^>
    echo 示例: run_evaluate.bat checkpoints\lightweight_20241017_210000\best_model.pth
    pause
    exit /b
)

REM 评估命令
python evaluate.py ^
    --checkpoint %1 ^
    --data_root "E:\DataSet\EEG\EEG dataset_SUAT_processed" ^
    --labels_csv "E:\output\connectivity_features\labels.csv" ^
    --batch_size 32 ^
    --num_workers 0 ^
    --save_dir evaluation_results

echo.
echo 评估完成！
pause

