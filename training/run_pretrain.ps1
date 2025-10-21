# Set UTF-8 encoding for Python
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "Starting Contrastive Pretraining with Multi-Matrix Attention Fusion..." -ForegroundColor Green

# Run contrastive pretraining
python -m training.contrastive_pretrain `
    --features_root "E:\output\connectivity_features" `
    --labels_csv "E:\output\connectivity_features\labels.csv" `
    --matrix_keys plv_alpha coherence_alpha wpli_alpha `
    --fusion_method attention `
    --batch_size 32 `
    --epochs 50 `
    --lr 0.001 `
    --hidden 128 `
    --proj 128 `
    --num_heads 4 `
    --device cuda `
    --save_dir checkpoints_pretrain

if ($LASTEXITCODE -eq 0) {
    Write-Host "Pretraining completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Pretraining failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}

Read-Host "Press Enter to exit"
