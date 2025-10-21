from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Simple dense graph convolution with symmetric normalization."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # adj: [B, N, N] or [N, N]; x: [B, N, F] or [N, F]
        if x.dim() == 2:
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)
        # Add self-loops
        I = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        A = adj + I
        # D^{-1/2} A D^{-1/2}
        deg = A.sum(-1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
        A_hat = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
        h = self.lin(x)
        out = torch.matmul(A_hat, h)
        return out.squeeze(0)


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 128, out_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList([GraphConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        h = x
        for i, conv in enumerate(self.layers):
            h = conv(h, adj)
            if i < len(self.layers) - 1:
                h = self.act(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        # global mean pooling
        # h shape: [B, N, D] or [N, D]
        if h.dim() == 2:
            # Single graph case: [N, D] -> [D]
            # But we need to keep batch dimension for contrastive learning
            # This should not happen in training, add a safeguard
            g = h.mean(dim=0, keepdim=True)  # [1, D]
        else:
            # Batch case: [B, N, D] -> [B, D]
            g = h.mean(dim=1)
        return g  # [B, D] or [1, D]


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveModel(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 128, proj_dim: int = 128, 
                 matrix_keys: List[str] = None, fusion_type: str = "attention"):
        super().__init__()
        self.matrix_keys = matrix_keys or ["plv_alpha"]
        
        if len(self.matrix_keys) > 1 and fusion_type == "attention":
            # Use multi-matrix attention fusion
            from .attention_fusion import MultiMatrixGCNEncoder
            self.encoder = MultiMatrixGCNEncoder(
                matrix_keys=self.matrix_keys,
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                fusion_type="cross_attention"
            )
        else:
            # Use standard GCN encoder
            self.encoder = GCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
        
        self.proj = ProjectionHead(hidden_dim, proj_dim)

    def forward(self, x, adj, matrices=None):
        if len(self.matrix_keys) > 1 and matrices is not None:
            # Multi-matrix case
            g = self.encoder(x, matrices)
        else:
            # Single matrix case
            g = self.encoder(x, adj)
        
        z = self.proj(g)
        z = F.normalize(z, dim=-1)
        return z, g


class SupervisedModel(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 128, num_classes: int = 2,
                 matrix_keys: List[str] = None, fusion_type: str = "attention"):
        super().__init__()
        self.matrix_keys = matrix_keys or ["plv_alpha"]
        
        if len(self.matrix_keys) > 1 and fusion_type == "attention":
            # Use multi-matrix attention fusion
            from .attention_fusion import MultiMatrixGCNEncoder
            self.encoder = MultiMatrixGCNEncoder(
                matrix_keys=self.matrix_keys,
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                fusion_type="cross_attention"
            )
        else:
            # Use standard GCN encoder
            self.encoder = GCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
        
        self.cls = ClassifierHead(hidden_dim, num_classes)

    def forward(self, x, adj, matrices=None):
        if len(self.matrix_keys) > 1 and matrices is not None:
            # Multi-matrix case
            g = self.encoder(x, matrices)
        else:
            # Single matrix case
            g = self.encoder(x, adj)
        
        logits = self.cls(g)
        return logits, g


