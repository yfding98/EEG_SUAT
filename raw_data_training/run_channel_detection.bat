@echo off
REM 活跃通道检测 - 训练和推理一键脚本

echo ========================================
echo EEG活跃通道检测系统
echo 任务：识别哪些通道是发作源
echo ========================================

echo.
echo 说明：
echo   - 输入：21个通道的EEG数据
echo   - 输出：哪2-5个通道是活跃的（发作源）
echo   - 方法：多标签二分类（每个通道活跃/不活跃）
echo.

REM 检查是否已有训练好的模型
if exist checkpoints_channel_detection (
    echo 检测到已有训练模型
    echo.
    choice /C YN /M "是否重新训练(Y)或直接推理(N)"
    
    if errorlevel 2 goto INFERENCE
    if errorlevel 1 goto TRAIN
) else (
    echo 未找到训练模型，开始训练...
    goto TRAIN
)

:TRAIN
echo.
echo ========================================
echo 步骤1: 训练活跃通道检测器
echo ========================================
python train_channel_detection.py ^
    --batch_size 16 ^
    --n_epochs 150 ^
    --lr 0.0005 ^
    --weight_decay 0.01 ^
    --dropout 0.3 ^
    --normalization window_robust ^
    --focal_alpha 0.75 ^
    --focal_gamma 2.0 ^
    --early_stopping_patience 30

if errorlevel 1 (
    echo 训练失败！
    pause
    exit /b 1
)

echo.
echo 训练完成！
echo.

:INFERENCE
echo ========================================
echo 步骤2: 测试活跃通道检测
echo ========================================
python test_channel_detection.py 1

echo.
echo 完成！
echo.
echo 下一步:
echo   1. 查看 test_detection_results\detection_*.png
echo   2. 查看 test_detection_results\detection_*.json
echo   3. 使用自己的文件推理:
echo      python inference_channel_detection.py --checkpoint xxx\best_model.pth --set_file xxx.set
echo.
pause

