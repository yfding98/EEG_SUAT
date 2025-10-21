@echo off
REM 通道感知训练脚本

echo ========================================
echo 通道感知EEG分类训练
echo 考虑容积传导效应和活跃通道信息
echo ========================================

REM 步骤1: 测试数据集
echo.
echo 步骤1: 测试通道感知数据集...
python dataset_channel_aware.py
if errorlevel 1 (
    echo 数据集测试失败！
    pause
    exit /b 1
)

REM 步骤2: 测试模型
echo.
echo 步骤2: 测试模型结构...
python model_channel_aware.py
if errorlevel 1 (
    echo 模型测试失败！
    pause
    exit /b 1
)

REM 步骤3: 开始训练
echo.
echo 步骤3: 开始训练...
echo 使用通道掩码标记活跃通道(发作源)
echo 保留所有21个通道，学习时空传播模式
echo.
python train_channel_aware.py ^
    --batch_size 16 ^
    --n_epochs 150 ^
    --lr 0.0005 ^
    --weight_decay 0.01 ^
    --dropout 0.3 ^
    --d_model 256 ^
    --n_heads 8 ^
    --n_layers 4 ^
    --normalization window_robust ^
    --early_stopping_patience 30

echo.
echo 训练完成！
echo.
echo 下一步：
echo 1. 查看训练曲线: tensorboard --logdir checkpoints_channel_aware
echo 2. 分析注意力权重（在模型中已保存）
echo 3. 可视化通道间关系
echo.
pause

