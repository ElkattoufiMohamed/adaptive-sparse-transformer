import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSparseAttention(nn.Module):
    """
    Adaptive Sparse Attention that dynamicaly selects attention patterns based in input content and learned parameteres.
    """

    def __init__(
        self,
        dim: int,  # Size of each token vector
        num_heads: int = 8,  # Multi_head attention splits dim into smaller heads, each head look at a different relationship
        dropout: float = 0.1,  # Randomly Zero out connection to avoid overfitting
        local_window_size: int = 32,  # How many token (Left & Right) we allow for Local attention
        global_ratio: float = 0.1,  # How much global attention is encouraged
        learnable_sparsity: bool = True,  # if True, we let the model learn how to select sparse pattern
        temperature: float = 1.0,  # Scale logit before softmax -> Controle the sharpness of the attention
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout
        self.local_window_size = local_window_size
        self.global_ratio = global_ratio
        self.temperature = temperature

        # Standard Q, K, V projections
        self.qkv = nn.Linear(
            dim, dim * 3, bias=False
        )  # A linear layer maps each token embedding of dim into 3 * dim. because we need Q, K ,V  (Query, Value, Key)
        self.proj = nn.Linear(
            dim, dim
        )  # After attention, we recombine the heads and project them back into the original dimension.
        self.dropout_layer = nn.Dropout(
            dropout
        )  # Aply dropout to attention weights (randomly dropping some connection btween token)

        # Pattern Selection Netwrok (PSN), this is a simple MLP Classifier
        self.pattern_selector = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 3),  # 3 patterns: local, global, sparse
            nn.Softmax(dim=-1),  # To ensure probabilities sum to 1
        )

        if learnable_sparsity:
            self.sparse_pattern_weights = nn.Parameter(torch.randn(num_heads, 1, 1))
            self.sparse_bias = nn.Parameter(
                torch.zeros(num_heads, 1, 1)
            )  # Means these are trainable weights, they let each head adjust how sparse attention is applied

        self.learnable_sparsity = learnable_sparsity
        
        # ADD THIS INITIALIZATION CODE HERE
        # Initialize pattern selector weights more conservatively
        for module in self.pattern_selector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)  # Smaller gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Also initialize other components more conservatively
        nn.init.xavier_normal_(self.qkv.weight, gain=0.1)
        nn.init.xavier_normal_(self.proj.weight, gain=0.1)
        
        # Initialize sparse pattern weights more conservatively too
        if learnable_sparsity:
            nn.init.normal_(self.sparse_pattern_weights, mean=0.0, std=0.1)  # Smaller std

    def create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Creat local attention mask with sliding window
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        for i in range(seq_len):
            start = max(0, i - self.local_window_size // 2)
            end = min(seq_len, i + self.local_window_size // 2 + 1)
            mask[i, start:end] = 0
        return mask

    def create_global_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Create global attention mask (Full attention).
        return torch.zeros(
            (seq_len, seq_len), device=device
        )  # Just return a matrix of zeros. No restriction, full global attention.

    def create_learned_sparse_mask(
        self, attention_scores: torch.Tensor, sparsity_ratio: float = 0.3
    ) -> torch.Tensor:
        # Create learned sparse attention mask.
        batch_size, num_heads, seq_len, _ = attention_scores.shape

        # Apply learnable transformation
        if self.learnable_sparsity:
            scores = attention_scores * self.sparse_pattern_weights + self.sparse_bias
        else:
            scores = attention_scores

        # Keep top-k values, masl others
        k = max(1, int(seq_len * (1 - sparsity_ratio)))

        # Ensure k doesn't exceed sequence length
        seq_len = scores.size(-1)
        k = min(k, seq_len)
        topk_values, topk_indices = torch.topk(scores, k, dim=-1)

        mask = torch.full_like(scores, float("-inf"))
        mask.scatter_(-1, topk_indices, 0)
        return mask

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        batch_size, seq_len, dim = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Pattern selection base on the input content
        # Use average pooled features for pattern selection
        pooled_features = torch.mean(x, dim=1)
        pattern_weights = self.pattern_selector(pooled_features)

        # Create different attention masks
        local_mask = self.create_local_mask(seq_len, x.device)
        global_mask = self.create_global_mask(seq_len, x.device)
        sparse_mask = self.create_learned_sparse_mask(attention_scores)

        # Combine mask based on learned weights
        combined_mask = (
            pattern_weights[:, 0:1].unsqueeze(1).unsqueeze(-1) * local_mask
            + pattern_weights[:, 1:2].unsqueeze(1).unsqueeze(-1) * global_mask
            + pattern_weights[:, 2:3].unsqueeze(1).unsqueeze(-1) * sparse_mask
        )

        # Apply combined mask
        attention_scores = attention_scores + combined_mask

        # Apply imput mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(1) == 0, float("-inf")
            )

        # Compute attention weights and apply dropout
        attention_weights = F.softmax(attention_scores / self.temperature, dim=-1)
        attention_weights = self.dropout_layer(attention_scores)

        # Applt attention to values
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)

        # Final projection
        out = self.proj(out)

        # Return output and attention analysis
        attention_info = {
            "pattern_weights": pattern_weights,
            "attention_weights": attention_weights,
            "local_ratio": pattern_weights[:, 0].mean().item(),
            "global_ratio": pattern_weights[:, 1].mean().item(),
            "sparse_ration": pattern_weights[:, 2].mean().item(),
        }

        return out, attention_info


class MultiHeadAdaptiveAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.attention = AdaptiveSparseAttention(
            dim=dim, num_heads=num_heads, dropout=dropout, **kwargs
        )

    def forward(self, x, mask=None):
        return self.attention(x, mask)
