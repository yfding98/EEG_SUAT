"""
Attention-based matrix fusion module for multi-matrix connectivity analysis.

This module provides attention mechanisms to dynamically fuse multiple connectivity
matrices (e.g., PLV, coherence, wPLI) based on their importance and context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class MatrixAttentionFusion(nn.Module):
    """
    Attention-based fusion of multiple connectivity matrices.
    
    This module learns to dynamically weight different connectivity matrices
    based on their relevance to the current task.
    """
    
    def __init__(
        self,
        matrix_keys: List[str],
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        fusion_type: str = "cross_attention"
    ):
        super().__init__()
        self.matrix_keys = matrix_keys
        self.num_matrices = len(matrix_keys)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.fusion_type = fusion_type
        
        # Matrix-specific encoders
        self.matrix_encoders = nn.ModuleDict({
            key: nn.Linear(1, hidden_dim) for key in matrix_keys
        })
        
        if fusion_type == "cross_attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_type == "self_attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_type == "gated":
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_dim * self.num_matrices, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_matrices),
                nn.Softmax(dim=-1)
            )
            self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, matrices: Dict[str, torch.Tensor], context: Optional[torch.Tensor] = None):
        """
        Fuse multiple connectivity matrices using attention.
        
        Args:
            matrices: Dictionary mapping matrix keys to tensors [B, N, N]
            context: Optional context tensor [B, hidden_dim]
            
        Returns:
            Fused matrix representation [B, N, N]
        """
        batch_size = next(iter(matrices.values())).shape[0]
        n_nodes = next(iter(matrices.values())).shape[1]
        
        # Encode each matrix
        matrix_embeddings = []
        for key in self.matrix_keys:
            if key in matrices:
                matrix = matrices[key]  # [B, N, N]
                # Flatten to [B, N*N, 1] then encode
                flat_matrix = matrix.view(batch_size, -1, 1)
                encoded = self.matrix_encoders[key](flat_matrix)  # [B, N*N, hidden_dim]
                # Reshape back to [B, N, N, hidden_dim]
                encoded = encoded.view(batch_size, n_nodes, n_nodes, self.hidden_dim)
                matrix_embeddings.append(encoded)
        
        if not matrix_embeddings:
            raise ValueError("No valid matrices provided")
        
        # Stack matrices [B, N, N, num_matrices, hidden_dim]
        stacked_matrices = torch.stack(matrix_embeddings, dim=3)  # [B, N, N, num_matrices, hidden_dim]
        
        if self.fusion_type == "cross_attention":
            return self._cross_attention_fusion(stacked_matrices, context)
        elif self.fusion_type == "self_attention":
            return self._self_attention_fusion(stacked_matrices)
        elif self.fusion_type == "gated":
            return self._gated_fusion(stacked_matrices)
    
    def _cross_attention_fusion(self, stacked_matrices: torch.Tensor, context: torch.Tensor):
        """Cross-attention fusion with context."""
        batch_size, n_nodes, _, num_matrices, hidden_dim = stacked_matrices.shape
        
        # Reshape for attention: [B*N*N, num_matrices, hidden_dim]
        query = stacked_matrices.view(-1, num_matrices, hidden_dim)
        
        if context is not None:
            # Use context as key/value
            context_expanded = context.unsqueeze(1).expand(-1, n_nodes*n_nodes, -1)
            context_reshaped = context_expanded.reshape(-1, 1, hidden_dim)
            
            # Cross-attention: matrices attend to context
            attended, _ = self.attention(query, context_reshaped, context_reshaped)
        else:
            # Self-attention among matrices
            attended, _ = self.attention(query, query, query)
        
        # Project and reshape back
        fused = self.fusion_proj(attended)  # [B*N*N, num_matrices, hidden_dim]
        fused = fused.mean(dim=1)  # [B*N*N, hidden_dim]
        fused = fused.view(batch_size, n_nodes, n_nodes, hidden_dim)
        
        # Convert back to connectivity matrix
        fused_matrix = fused.mean(dim=-1)  # [B, N, N]
        return fused_matrix
    
    def _self_attention_fusion(self, stacked_matrices: torch.Tensor):
        """Self-attention fusion among matrices."""
        batch_size, n_nodes, _, num_matrices, hidden_dim = stacked_matrices.shape
        
        # Reshape for attention: [B*N*N, num_matrices, hidden_dim]
        matrices_reshaped = stacked_matrices.view(-1, num_matrices, hidden_dim)
        
        # Self-attention among matrices
        attended, attention_weights = self.attention(
            matrices_reshaped, matrices_reshaped, matrices_reshaped
        )
        
        # Project and reshape back
        fused = self.fusion_proj(attended)  # [B*N*N, num_matrices, hidden_dim]
        fused = fused.mean(dim=1)  # [B*N*N, hidden_dim]
        fused = fused.view(batch_size, n_nodes, n_nodes, hidden_dim)
        
        # Convert back to connectivity matrix
        fused_matrix = fused.mean(dim=-1)  # [B, N, N]
        return fused_matrix, attention_weights
    
    def _gated_fusion(self, stacked_matrices: torch.Tensor):
        """Gated fusion with learned weights."""
        batch_size, n_nodes, _, num_matrices, hidden_dim = stacked_matrices.shape
        
        # Flatten for gating network
        flat_matrices = stacked_matrices.view(batch_size, n_nodes, n_nodes, -1)
        
        # Compute gates
        gates = self.gate_network(flat_matrices)  # [B, N, N, num_matrices]
        gates = gates.unsqueeze(-1)  # [B, N, N, num_matrices, 1]
        
        # Apply gates
        gated_matrices = stacked_matrices * gates  # [B, N, N, num_matrices, hidden_dim]
        fused = gated_matrices.sum(dim=3)  # [B, N, N, hidden_dim]
        
        # Project
        fused = self.fusion_proj(fused)  # [B, N, N, hidden_dim]
        
        # Convert back to connectivity matrix
        fused_matrix = fused.mean(dim=-1)  # [B, N, N]
        return fused_matrix


class MultiMatrixGCNEncoder(nn.Module):
    """
    GCN encoder with multi-matrix attention fusion.
    """
    
    def __init__(
        self,
        matrix_keys: List[str],
        in_dim: int = 2,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        fusion_type: str = "cross_attention"
    ):
        super().__init__()
        self.matrix_keys = matrix_keys
        self.num_layers = num_layers
        
        # Multi-matrix attention fusion
        self.matrix_fusion = MatrixAttentionFusion(
            matrix_keys=matrix_keys,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            fusion_type=fusion_type
        )
        
        # GCN layers
        from .models import GraphConv
        self.gcn_layers = nn.ModuleList([
            GraphConv(in_dim if i == 0 else hidden_dim, 
                     hidden_dim if i < num_layers - 1 else out_dim)
            for i in range(num_layers)
        ])
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, matrices: Dict[str, torch.Tensor]):
        """
        Forward pass with multi-matrix fusion.
        
        Args:
            x: Node features [B, N, F]
            matrices: Dictionary of connectivity matrices [B, N, N]
            
        Returns:
            Graph representation [B, hidden_dim]
        """
        # Fuse multiple matrices
        fused_adj = self.matrix_fusion(matrices)  # [B, N, N]
        
        # Apply GCN layers
        h = x
        for i, gcn in enumerate(self.gcn_layers):
            h = gcn(h, fused_adj)
            if i < len(self.gcn_layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        
        # Global pooling
        # h shape: [B, N, D] or [N, D]
        if h.dim() == 2:
            # Single graph case: [N, D] -> [1, D] (keep batch dimension)
            g = h.mean(dim=0, keepdim=True)
        else:
            # Batch case: [B, N, D] -> [B, D]
            g = h.mean(dim=1)
        
        return g  # [B, D] or [1, D]


class MatrixTypeClassifier(nn.Module):
    """
    Auxiliary task: classify matrix types.
    """
    
    def __init__(self, hidden_dim: int, num_matrix_types: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_matrix_types)
        )
    
    def forward(self, x: torch.Tensor):
        return self.classifier(x)


if __name__ == "__main__":
    # Test the attention fusion
    batch_size, n_nodes = 2, 10
    matrix_keys = ["plv_alpha", "coherence_alpha", "wpli_alpha"]
    
    # Create dummy matrices
    matrices = {
        key: torch.randn(batch_size, n_nodes, n_nodes) 
        for key in matrix_keys
    }
    
    # Test attention fusion
    fusion = MatrixAttentionFusion(matrix_keys, hidden_dim=64, num_heads=2)
    fused_matrix = fusion(matrices)
    print(f"Fused matrix shape: {fused_matrix.shape}")
    
    # Test multi-matrix GCN
    gcn = MultiMatrixGCNEncoder(
        matrix_keys=matrix_keys,
        in_dim=2,
        hidden_dim=64,
        out_dim=32
    )
    
    x = torch.randn(batch_size, n_nodes, 2)
    output = gcn(x, matrices)
    print(f"GCN output shape: {output.shape}")
