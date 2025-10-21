@echo off
REM 训练脚本 - Windows批处理文件

echo ======================================
echo 原始EEG数据训练
echo ======================================

REM 激活conda环境（如果需要）
REM call conda activate your_env_name

REM 设置Python路径（如果需要）
REM set PYTHONPATH=%PYTHONPATH%;E:\code_learn\SUAT\workspace\EEG-projects\LaBraM

REM 训练命令
python train.py ^
    --data_root "E:\DataSet\EEG\EEG dataset_SUAT_processed" ^
    --labels_csv "E:\output\connectivity_features\labels.csv" ^
    --model_type lightweight ^
    --batch_size 32 ^
    --n_epochs 100 ^
    --lr 0.001 ^
    --num_workers 0 ^
    --save_dir checkpoints

echo.
echo 训练完成！
pause

