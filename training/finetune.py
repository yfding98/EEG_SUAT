import os
import argparse
import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from .datasets import ConnectivityGraphDataset, discover_patient_segments, discover_patient_segments_from_csv, make_patient_splits, load_labels_csv
from .models import SupervisedModel, ContrastiveModel


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_graph(batch):
    max_n = max(int(item['n']) for item in batch)
    B = len(batch)
    adjs = torch.zeros(B, max_n, max_n)
    xs = torch.zeros(B, max_n, batch[0]['x'].shape[-1])
    ys = torch.tensor([item['y'] for item in batch], dtype=torch.long)
    paths = [item['path'] for item in batch]
    
    # Check if we have multi-matrix data
    has_matrices = 'matrices' in batch[0]
    matrices_dict = None
    
    if has_matrices:
        # Collect all matrix keys
        all_keys = set()
        for item in batch:
            all_keys.update(item['matrices'].keys())
        
        # Create padded matrices for each key
        matrices_dict = {}
        for key in all_keys:
            key_matrices = torch.zeros(B, max_n, max_n)
            for i, item in enumerate(batch):
                if key in item['matrices']:
                    n = int(item['n'])
                    key_matrices[i, :n, :n] = item['matrices'][key]
            matrices_dict[key] = key_matrices
    
    for i, it in enumerate(batch):
        n = int(it['n'])
        adjs[i, :n, :n] = it['adj']
        xs[i, :n] = it['x']
    
    result = {"adj": adjs, "x": xs, "y": ys, "paths": paths}
    if matrices_dict is not None:
        result["matrices"] = matrices_dict
    
    return result


@torch.no_grad()
def evaluate(model: SupervisedModel, loader: DataLoader, device: str, num_classes: int):
    model.eval()
    logits_all = []
    y_all = []
    for batch in loader:
        adj = batch['adj'].to(device)
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        
        # Check if we have multi-matrix data
        matrices = None
        if 'matrices' in batch:
            matrices = {k: v.to(device) for k, v in batch['matrices'].items()}
        
        logits, _ = model(x, adj, matrices)
        logits_all.append(logits.cpu())
        y_all.append(y.cpu())
    logits_all = torch.cat(logits_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    pred = logits_all.argmax(dim=-1).numpy()
    y_true = y_all.numpy()
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, average='macro')
    try:
        if num_classes == 2:
            proba = torch.softmax(logits_all, dim=-1)[:, 1].numpy()
            auroc = roc_auc_score(y_true, proba)
        else:
            proba = torch.softmax(logits_all, dim=-1).numpy()
            auroc = roc_auc_score(y_true, proba, multi_class='ovr')
    except Exception:
        auroc = float('nan')
    return acc, f1, auroc


def load_pretrained(encoder: SupervisedModel, ckpt_path: str):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    z = ContrastiveModel().state_dict()
    z.update(state['model'])
    # load only encoder weights if possible
    enc_state = {k.replace('encoder.', ''): v for k, v in z.items() if k.startswith('encoder.')}
    model_state = encoder.encoder.state_dict()
    intersect = {k: v for k, v in enc_state.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(intersect)
    encoder.encoder.load_state_dict(model_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_root', required=True, help='Root directory containing connectivity features')
    parser.add_argument('--labels_csv', required=True, help='CSV file with patient and channel combination information')
    parser.add_argument('--matrix_keys', nargs='+', default=['plv_alpha'], 
                       help='List of matrix keys to use (e.g., plv_alpha coherence_alpha wpli_alpha)')
    parser.add_argument('--fusion_method', default='attention', choices=['attention', 'concat', 'weighted'],
                       help='Method to fuse multiple matrices')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', default='checkpoints_finetune')
    parser.add_argument('--pretrained', default='checkpoints_pretrain/best.pt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load labels CSV
    labels_df = load_labels_csv(args.labels_csv)
    
    # Discover patient segments using CSV
    patient_to_files = discover_patient_segments_from_csv(args.labels_csv, args.features_root)
    split = make_patient_splits(patient_to_files, test_ratio=0.2, val_ratio=0.1, seed=args.seed)
    train_files, val_files, test_files = split['train'], split['val'], split['test']

    train_ds = ConnectivityGraphDataset(
        train_files, labels_df, 
        matrix_keys=args.matrix_keys, 
        fusion_method=args.fusion_method,
        augment=False
    )
    val_ds = ConnectivityGraphDataset(
        val_files, labels_df, 
        matrix_keys=args.matrix_keys, 
        fusion_method=args.fusion_method,
        augment=False
    )
    test_ds = ConnectivityGraphDataset(
        test_files, labels_df, 
        matrix_keys=args.matrix_keys, 
        fusion_method=args.fusion_method,
        augment=False
    )
    
    # Auto-detect number of classes if not specified correctly
    actual_num_classes = train_ds.get_num_classes()
    if args.num_classes != actual_num_classes:
        print(f"⚠ WARNING: --num_classes was set to {args.num_classes}, but dataset has {actual_num_classes} classes")
        print(f"  Automatically using {actual_num_classes} classes")
        args.num_classes = actual_num_classes

    collate = collate_graph
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    model = SupervisedModel(
        in_dim=2, 
        hidden_dim=args.hidden, 
        num_classes=args.num_classes,
        matrix_keys=args.matrix_keys,
        fusion_type=args.fusion_method
    ).to(args.device)
    
    print(f"\n{'='*60}")
    print(f"Model Configuration:")
    print(f"  - Input dim: 2")
    print(f"  - Hidden dim: {args.hidden}")
    print(f"  - Number of classes: {args.num_classes}")
    print(f"  - Matrix keys: {args.matrix_keys}")
    print(f"  - Fusion method: {args.fusion_method}")
    print(f"{'='*60}\n")
    load_pretrained(model, args.pretrained)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, batch in enumerate(pbar):
            adj = batch['adj'].to(args.device)
            x = batch['x'].to(args.device)
            y = batch['y'].to(args.device)
            
            # Check if we have multi-matrix data
            matrices = None
            if 'matrices' in batch:
                matrices = {k: v.to(args.device) for k, v in batch['matrices'].items()}
            
            logits, _ = model(x, adj, matrices)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total_loss += float(loss.item())
            
            # Update progress bar with current loss
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

        avg_train_loss = total_loss / max(1, len(train_loader))
        acc, f1, auroc = evaluate(model, val_loader, args.device, args.num_classes)
        
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Acc: {acc:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}')
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'best_f1': best_f1}, 
                      os.path.join(args.save_dir, 'best.pt'))
            print(f'  ✓ Saved best model (F1={best_f1:.4f})')
        
        torch.save({'epoch': epoch, 'model': model.state_dict()}, os.path.join(args.save_dir, 'last.pt'))

    # final test
    state = torch.load(os.path.join(args.save_dir, 'best.pt'), map_location=args.device, weights_only=False)
    model.load_state_dict(state['model'])
    acc, f1, auroc = evaluate(model, test_loader, args.device, args.num_classes)
    print(f'Test: acc={acc:.4f} f1={f1:.4f} auroc={auroc:.4f}')


if __name__ == '__main__':
    main()


