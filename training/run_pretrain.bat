@echo off
REM Set UTF-8 encoding for Python
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM Run contrastive pretraining with multi-matrix attention fusion
python -m training.contrastive_pretrain ^
    --features_root E:\output\connectivity_features ^
    --labels_csv E:\output\connectivity_features\labels.csv ^
    --matrix_keys plv_alpha coherence_alpha wpli_alpha ^
    --fusion_method attention ^
    --batch_size 32 ^
    --epochs 50 ^
    --lr 0.001 ^
    --hidden 128 ^
    --proj 128 ^
    --num_heads 4 ^
    --device cuda ^
    --save_dir checkpoints_pretrain

pause
