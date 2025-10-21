# Set UTF-8 encoding for Python
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "Starting Supervised Finetuning with Multi-Matrix Attention Fusion..." -ForegroundColor Green

# Run supervised finetuning
python -m training.finetune `
    --features_root "E:\output\connectivity_features" `
    --labels_csv "E:\output\connectivity_features\labels.csv" `
    --matrix_keys plv_alpha coherence_alpha wpli_alpha `
    --fusion_method attention `
    --pretrain_ckpt "checkpoints_pretrain\best.pt" `
    --batch_size 32 `
    --epochs 100 `
    --lr 0.0005 `
    --hidden 128 `
    --device cuda `
    --save_dir checkpoints_finetune

if ($LASTEXITCODE -eq 0) {
    Write-Host "Finetuning completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Finetuning failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}

Read-Host "Press Enter to exit"
