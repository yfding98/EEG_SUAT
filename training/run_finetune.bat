@echo off
REM Set UTF-8 encoding for Python
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM Run supervised finetuning with multi-matrix attention fusion
python -m training.finetune ^
    --features_root E:\output\connectivity_features ^
    --labels_csv E:\output\connectivity_features\labels.csv ^
    --matrix_keys plv_alpha coherence_alpha wpli_alpha ^
    --fusion_method attention ^
    --pretrain_ckpt checkpoints_pretrain\best.pt ^
    --batch_size 32 ^
    --epochs 100 ^
    --lr 0.0005 ^
    --hidden 128 ^
    --device cuda ^
    --save_dir checkpoints_finetune

pause
