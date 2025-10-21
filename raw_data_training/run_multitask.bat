@echo off
REM 多任务学习训练脚本
REM 显式利用通道组合信息

echo ========================================
echo 多任务EEG分类训练
echo 任务1: 发作类型分类（主）
echo 任务2: 活跃通道预测（辅）
echo 任务3: 通道关系学习（辅）
echo ========================================

REM 步骤1: 测试模型
echo.
echo 步骤1: 测试多任务模型...
python model_multitask.py
if errorlevel 1 (
    echo 模型测试失败！
    pause
    exit /b 1
)

REM 步骤2: 测试数据集
echo.
echo 步骤2: 测试通道感知数据集...
python dataset_channel_aware.py
if errorlevel 1 (
    echo 数据集测试失败！
    pause
    exit /b 1
)

REM 步骤3: 开始训练
echo.
echo 步骤3: 开始多任务训练...
echo.
echo 说明：
echo - 主任务：发作类型分类（SZ1, SZ4等）
echo - 辅助任务1：预测哪些通道是活跃的（显式利用labels.csv）
echo - 辅助任务2：学习通道间关系（建模容积传导）
echo.
python train_multitask.py ^
    --batch_size 16 ^
    --n_epochs 150 ^
    --lr 0.0005 ^
    --weight_decay 0.01 ^
    --dropout 0.3 ^
    --d_model 256 ^
    --n_heads 8 ^
    --n_layers 4 ^
    --seizure_weight 1.0 ^
    --channel_weight 0.5 ^
    --relation_weight 0.3 ^
    --normalization window_robust ^
    --early_stopping_patience 30

echo.
echo 训练完成！
echo.
echo 查看结果：
echo 1. TensorBoard: tensorboard --logdir checkpoints_multitask
echo 2. 检查点: checkpoints_multitask\multitask_xxxxxx\
echo 3. 注意监控"通道预测准确率"（表明模型是否学会识别活跃通道）
echo.
pause

