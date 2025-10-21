@echo off
REM Multi-Label Classification Training Script
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ============================================================
echo Multi-Label Channel Classification Training
echo ============================================================

python train_multilabel.py ^
    --features_root E:\output\connectivity_features ^
    --labels_csv E:\output\connectivity_features\labels.csv ^
    --matrix_keys plv_alpha ^
    --model_type basic ^
    --hidden_dim 128 ^
    --num_layers 3 ^
    --dropout 0.5 ^
    --batch_size 32 ^
    --epochs 100 ^
    --lr 0.001 ^
    --use_pos_weight ^
    --threshold 0.5 ^
    --device cuda ^
    --save_dir checkpoints_multilabel ^
    --seed 42

pause

