@echo off
REM 分析badcase并使用过滤后的数据集训练

echo ========================================
echo 步骤1: 分析当前模型找出badcase
echo ========================================
python quick_analyze.py

if not exist badcase_analysis.json (
    echo 未生成badcase分析文件，使用原始数据集训练
    set BAD_WINDOWS_ARG=
) else (
    echo 找到badcase分析文件，将使用过滤后的数据集
    set BAD_WINDOWS_ARG=--bad_windows_file badcase_analysis.json
)

echo.
echo ========================================
echo 步骤2: 使用改进的训练策略
echo ========================================
echo 归一化方法: window_robust (对被试者差异鲁棒)
echo.
python train_improved.py ^
    --batch_size 16 ^
    --n_epochs 150 ^
    --lr 0.0005 ^
    --weight_decay 0.01 ^
    --dropout 0.5 ^
    --gradient_clip 1.0 ^
    --warmup_epochs 10 ^
    --early_stopping_patience 30 ^
    --use_augmentation ^
    --normalization window_robust ^
    %BAD_WINDOWS_ARG%

echo.
echo 训练完成！
pause

