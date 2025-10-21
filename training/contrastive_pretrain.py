import os
import argparse
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import ConnectivityGraphDataset, discover_patient_segments, discover_patient_segments_from_csv, make_patient_splits, load_labels_csv
from .models import ContrastiveModel


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def graph_augment(adj: torch.Tensor, x: torch.Tensor, drop_edge: float = 0.2, feat_noise: float = 0.05):
    """
    Graph augmentation with support for multi-matrix inputs.
    Note: fill_diagonal_ does not work with batched tensors, so we use indexing.
    """
    A = adj.clone()
    
    if drop_edge > 0:
        mask = (torch.rand_like(A) > drop_edge).float()
        A = A * mask
        
        # Symmetrize the adjacency matrix
        if A.dim() == 3:  # [batch_size, n_nodes, n_nodes]
            A = (A + A.transpose(-1, -2)) / 2.0
            # Fill diagonal using indexing (works with batched tensors)
            if A.shape[-1] == A.shape[-2]:
                n = A.shape[-1]
                A[:, range(n), range(n)] = 0.0
        elif A.dim() == 4:  # [batch_size, n_nodes, n_nodes, n_matrices]
            A = (A + A.transpose(-2, -3)) / 2.0
            # Fill diagonal for each matrix using indexing
            n = A.shape[1]
            A[:, range(n), range(n), :] = 0.0
    
    X = x.clone()
    if feat_noise > 0:
        X = X + torch.randn_like(X) * feat_noise
    return A, X


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    B = z1.size(0)
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels])
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))
    loss = F.cross_entropy(sim, labels)
    return loss


def collate_graph(batch):
    # variable-size graphs -> pad to max N in batch
    max_n = max(int(item['n']) for item in batch)
    B = len(batch)
    adjs = torch.zeros(B, max_n, max_n)
    xs = torch.zeros(B, max_n, batch[0]['x'].shape[-1])
    ys = torch.tensor([item['y'] for item in batch], dtype=torch.long)
    
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
    
    result = {"adj": adjs, "x": xs, "y": ys}
    if matrices_dict is not None:
        result["matrices"] = matrices_dict
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_root', required=True, help='Root directory containing connectivity features')
    parser.add_argument('--labels_csv', required=True, help='CSV file with patient and channel combination information')
    parser.add_argument('--matrix_keys', nargs='+', default=['plv_alpha'], 
                       help='List of matrix keys to use (e.g., plv_alpha coherence_alpha wpli_alpha)')
    parser.add_argument('--fusion_method', default='attention', choices=['attention', 'concat', 'weighted'],
                       help='Method to fuse multiple matrices')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--proj', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', default='checkpoints_pretrain')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load labels CSV
    labels_df = load_labels_csv(args.labels_csv)
    
    # Discover patient segments using CSV
    patient_to_files = discover_patient_segments_from_csv(args.labels_csv, args.features_root)
    split = make_patient_splits(patient_to_files, test_ratio=0.2, val_ratio=0.1, seed=args.seed)
    train_files = split['train']

    dataset = ConnectivityGraphDataset(
        train_files, labels_df, 
        matrix_keys=args.matrix_keys, 
        fusion_method=args.fusion_method,
        augment=False
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_graph)

    model = ContrastiveModel(
        in_dim=2, 
        hidden_dim=args.hidden, 
        proj_dim=args.proj,
        matrix_keys=args.matrix_keys,
        fusion_type=args.fusion_method
    ).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, batch in enumerate(pbar):
            adj = batch['adj'].to(args.device)
            x = batch['x'].to(args.device)
            
            # Check if we have multi-matrix data
            matrices = None
            if 'matrices' in batch:
                matrices = {k: v.to(args.device) for k, v in batch['matrices'].items()}

            adj1, x1 = graph_augment(adj, x, drop_edge=0.2, feat_noise=0.05)
            adj2, x2 = graph_augment(adj, x, drop_edge=0.2, feat_noise=0.05)
            
            # Augment matrices if available
            matrices1, matrices2 = None, None
            if matrices is not None:
                matrices1 = {k: graph_augment(v, x, drop_edge=0.2, feat_noise=0.0)[0] 
                            for k, v in matrices.items()}
                matrices2 = {k: graph_augment(v, x, drop_edge=0.2, feat_noise=0.0)[0] 
                            for k, v in matrices.items()}

            z1, _ = model(x1, adj1, matrices1)
            z2, _ = model(x2, adj2, matrices2)
            loss = nt_xent(z1, z2, temperature=0.2)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            epoch_loss += float(loss.item())
            
            # Update progress bar with current loss
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
            })

        avg = epoch_loss / max(1, len(loader))
        print(f'\nEpoch {epoch} Summary: avg_loss={avg:.4f}, best={best:.4f}')
        
        if avg < best:
            best = avg
            ckpt = os.path.join(args.save_dir, 'best.pt')
            torch.save({'epoch': epoch, 'model': model.state_dict()}, ckpt)
            print(f'  ✓ Saved best model (loss={best:.4f})')
        
        ckpt_last = os.path.join(args.save_dir, 'last.pt')
        torch.save({'epoch': epoch, 'model': model.state_dict()}, ckpt_last)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--features_root', r'E:\output\connectivity_features',
            '--labels_csv', r'E:\output\connectivity_features\labels.csv',
            '--matrix_keys', 'plv_alpha', 'coherence_alpha', 'wpli_alpha',
            '--fusion_method', 'attention'
        ])
    sys.exit(main())


