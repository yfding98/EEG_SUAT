#!/usr/bin/env python3
"""
Training script for connectivity-based EEG analysis with GNN.

This script demonstrates how to run the complete training pipeline:
1. Contrastive pretraining on connectivity graphs
2. Supervised finetuning for classification

Usage:
    python training/run_training.py --features_root E:\output\connectivity_features --labels_csv E:\output\connectivity_features\labels.csv
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_pretraining(features_root: str, labels_csv: str, **kwargs):
    """Run contrastive pretraining."""
    matrix_keys = kwargs.get('matrix_keys', ['plv_alpha'])
    cmd = [
        sys.executable, '-m', 'training.contrastive_pretrain',
        '--features_root', features_root,
        '--labels_csv', labels_csv,
        '--matrix_keys'] + matrix_keys + [
        '--fusion_method', kwargs.get('fusion_method', 'attention'),
        '--batch_size', str(kwargs.get('batch_size', 32)),
        '--epochs', str(kwargs.get('epochs', 50)),
        '--lr', str(kwargs.get('lr', 1e-3)),
        '--hidden', str(kwargs.get('hidden', 128)),
        '--proj', str(kwargs.get('proj', 128)),
        '--device', kwargs.get('device', 'cuda' if os.system('nvidia-smi') == 0 else 'cpu'),
        '--save_dir', kwargs.get('save_dir', 'checkpoints_pretrain'),
        '--seed', str(kwargs.get('seed', 42))
    ]
    
    print("Running contrastive pretraining...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Don't capture output - let it stream to terminal for progress bar
    result = subprocess.run(cmd, encoding='utf-8', errors='replace')
    
    print("-" * 60)
    if result.returncode != 0:
        print(f"Pretraining failed with return code {result.returncode}")
        return False
    
    print("Pretraining completed successfully!")
    return True

def run_finetuning(features_root: str, labels_csv: str, pretrained_path: str, **kwargs):
    """Run supervised finetuning."""
    matrix_keys = kwargs.get('matrix_keys', ['plv_alpha'])
    cmd = [
        sys.executable, '-m', 'training.finetune',
        '--features_root', features_root,
        '--labels_csv', labels_csv,
        '--matrix_keys'] + matrix_keys + [
        '--fusion_method', kwargs.get('fusion_method', 'attention'),
        '--batch_size', str(kwargs.get('batch_size', 32)),
        '--epochs', str(kwargs.get('finetune_epochs', 30)),
        '--lr', str(kwargs.get('finetune_lr', 2e-4)),
        '--hidden', str(kwargs.get('hidden', 128)),
        '--num_classes', str(kwargs.get('num_classes', 2)),
        '--device', kwargs.get('device', 'cuda' if os.system('nvidia-smi') == 0 else 'cpu'),
        '--save_dir', kwargs.get('finetune_save_dir', 'checkpoints_finetune'),
        '--pretrained', pretrained_path,
        '--seed', str(kwargs.get('seed', 42))
    ]
    
    print("Running supervised finetuning...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Don't capture output - let it stream to terminal for progress bar
    result = subprocess.run(cmd, encoding='utf-8', errors='replace')
    
    print("-" * 60)
    if result.returncode != 0:
        print(f"Finetuning failed with return code {result.returncode}")
        return False
    
    print("Finetuning completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run complete training pipeline')
    parser.add_argument('--features_root', required=True, help='Root directory containing connectivity features')
    parser.add_argument('--labels_csv', required=True, help='CSV file with patient and channel combination information')
    parser.add_argument('--matrix_keys', nargs='+', default=['plv_alpha'], 
                       help='List of matrix keys to use (e.g., plv_alpha coherence_alpha wpli_alpha)')
    parser.add_argument('--fusion_method', default='attention', choices=['attention', 'concat', 'weighted'],
                       help='Method to fuse multiple matrices')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Pretraining epochs')
    parser.add_argument('--finetune_epochs', type=int, default=30, help='Finetuning epochs')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='Pretraining learning rate')
    parser.add_argument('--finetune_lr', type=float, default=2e-4, help='Finetuning learning rate')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--proj', type=int, default=128, help='Projection dimension')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--device', default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--pretrain_save_dir', default='checkpoints_pretrain', help='Pretraining save directory')
    parser.add_argument('--finetune_save_dir', default='checkpoints_finetune', help='Finetuning save directory')
    parser.add_argument('--skip_pretrain', action='store_true', help='Skip pretraining step')
    parser.add_argument('--skip_finetune', action='store_true', help='Skip finetuning step')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Features root: {args.features_root}")
    print(f"Labels CSV: {args.labels_csv}")
    
    # Check if files exist
    if not os.path.exists(args.features_root):
        print(f"Error: Features root directory does not exist: {args.features_root}")
        return 1
    
    if not os.path.exists(args.labels_csv):
        print(f"Error: Labels CSV file does not exist: {args.labels_csv}")
        return 1
    
    # Run pretraining
    if not args.skip_pretrain:
        success = run_pretraining(
            args.features_root, args.labels_csv,
            matrix_keys=args.matrix_keys,
            fusion_method=args.fusion_method,
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
            hidden=args.hidden,
            proj=args.proj,
            device=device,
            save_dir=args.pretrain_save_dir,
            seed=args.seed
        )
        if not success:
            print("Pretraining failed, stopping pipeline")
            return 1
    else:
        print("Skipping pretraining...")
    
    # Run finetuning
    if not args.skip_finetune:
        pretrained_path = os.path.join(args.pretrain_save_dir, 'best.pt')
        if not os.path.exists(pretrained_path):
            print(f"Warning: Pretrained model not found at {pretrained_path}")
            print("Proceeding with finetuning without pretrained weights...")
            pretrained_path = None
        
        success = run_finetuning(
            args.features_root, args.labels_csv, pretrained_path or '',
            matrix_keys=args.matrix_keys,
            fusion_method=args.fusion_method,
            batch_size=args.batch_size,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
            hidden=args.hidden,
            num_classes=args.num_classes,
            device=device,
            finetune_save_dir=args.finetune_save_dir,
            seed=args.seed
        )
        if not success:
            print("Finetuning failed")
            return 1
    else:
        print("Skipping finetuning...")
    
    print("Training pipeline completed successfully!")
    return 0

if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend([
            '--features_root', r'E:\output\connectivity_features',
            '--labels_csv', r'E:\output\connectivity_features\labels.csv',
            '--matrix_keys', 'plv_alpha', 'coherence_alpha', 'wpli_alpha',
            '--batch_size', '16',
            '--pretrain_epochs', '50',
            '--device', 'cuda',
            '--fusion_method', 'attention'
        ])
    sys.exit(main())
