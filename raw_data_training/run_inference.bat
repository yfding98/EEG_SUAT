@echo off
REM EEG推理脚本

echo ========================================
echo EEG癫痫发作推理和活跃通道识别
echo ========================================

REM 检查是否提供了参数
if "%1"=="" (
    echo.
    echo 使用方法:
    echo   run_inference.bat ^<checkpoint_path^> ^<set_file_path^> [window_idx]
    echo.
    echo 示例:
    echo   run_inference.bat checkpoints_multitask\multitask_20241018_143000\best_model.pth "E:\DataSet\...\SZ1_F7_Fp1.set"
    echo   run_inference.bat checkpoints_multitask\multitask_20241018_143000\best_model.pth "E:\DataSet\...\SZ1_F7_Fp1.set" 0
    echo.
    echo 或运行快速测试:
    echo   python test_inference.py
    echo.
    pause
    exit /b
)

if "%2"=="" (
    echo 错误: 需要提供.set文件路径
    pause
    exit /b 1
)

set CHECKPOINT=%1
set SET_FILE=%2
set WINDOW_IDX=%3

REM 执行推理
echo.
echo 检查点: %CHECKPOINT%
echo 文件: %SET_FILE%
if not "%WINDOW_IDX%"=="" (
    echo 窗口索引: %WINDOW_IDX%
    set WINDOW_ARG=--window_idx %WINDOW_IDX%
) else (
    echo 窗口: 第一个
    set WINDOW_ARG=
)

echo.
echo 开始推理...
echo.

python inference.py ^
    --checkpoint %CHECKPOINT% ^
    --set_file %SET_FILE% ^
    %WINDOW_ARG% ^
    --save_dir inference_results

echo.
echo 推理完成！
echo.
echo 查看结果:
echo   - inference_results\prediction_*.json
echo   - inference_results\window_*_prediction.png
echo.
pause

