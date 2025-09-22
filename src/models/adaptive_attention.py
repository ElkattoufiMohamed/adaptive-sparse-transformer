# src/models/adaptive_attention.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class AdaptiveSparseAttention(nn.Module):
    """
    Adaptive Sparse Attention that dynamically selects attention patterns
    (local / global / learned-sparse) per sequence via a small pattern selector.
    This implementation uses binary masks (0/1) for patterns and combines them
    with a soft-OR rule so positions are allowed if any pattern permits them.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        local_window_size: int = 32,
        global_ratio: float = 0.1,
        learnable_sparsity: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout
        self.local_window_size = local_window_size
        self.global_ratio = global_ratio
        self.temperature = temperature

        # QKV and output projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout_layer = nn.Dropout(self.dropout_p)

        # Pattern selector: sequence-level MLP -> (local, global, sparse)
        self.pattern_selector = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 3),
            nn.Softmax(dim=-1),
        )

        # Per-head learnable sparsity parameters (applied to attention scores)
        self.learnable_sparsity = learnable_sparsity
        if learnable_sparsity:
            # store as (H,1,1) then broadcast to (1,H,1,1)
            self.sparse_pattern_weights = nn.Parameter(torch.randn(num_heads, 1, 1))
            self.sparse_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

        # Conservative initialization for stability
        for module in self.pattern_selector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.xavier_normal_(self.qkv.weight, gain=0.1)
        nn.init.xavier_normal_(self.proj.weight, gain=0.1)
        if learnable_sparsity:
            nn.init.normal_(self.sparse_pattern_weights, mean=0.0, std=0.1)
            nn.init.zeros_(self.sparse_bias)

    # --- Mask constructors returning binary masks (0/1) ---
    def create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Binary local mask of shape (L, L) with 1 where attention allowed inside window.
        """
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
        half = self.local_window_size // 2
        for i in range(seq_len):
            start = max(0, i - half)
            end = min(seq_len, i + half + 1)
            mask[i, start:end] = 1.0
        return mask

    def create_global_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Binary global mask (all ones)."""
        return torch.ones((seq_len, seq_len), device=device, dtype=torch.float32)

    def create_learned_sparse_mask(
        self, attention_scores: torch.Tensor, sparsity_ratio: float = 0.3
    ) -> torch.Tensor:
        """
        Create a learned sparse binary mask (B, H, L, L) from attention_scores.
        Returns 1 where keys are kept, 0 where masked.
        """
        # sanitize inputs
        attention_scores = torch.nan_to_num(attention_scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        attention_scores = attention_scores.clamp(min=-1e9, max=1e9)

        B, H, L, _ = attention_scores.shape
        # k = number of keys to keep per query (at least 1)
        k = max(1, int(L * (1 - float(sparsity_ratio))))
        k = min(k, L)

        # tiny jitter to avoid tie issues
        if attention_scores.is_floating_point():
            attention_scores = attention_scores + (torch.rand_like(attention_scores) * 1e-12)

        # apply per-head transform if enabled (broadcast to (B, H, L, L))
        if self.learnable_sparsity:
            w = self.sparse_pattern_weights.view(1, H, 1, 1)  # (1,H,1,1)
            b = self.sparse_bias.view(1, H, 1, 1)
            scores = attention_scores * w + b
        else:
            scores = attention_scores

        # topk along last dim
        topk_values, topk_indices = torch.topk(scores, k, dim=-1)

        # build binary mask: zeros default, ones at topk indices
        mask = torch.zeros_like(scores, dtype=torch.float32)
        mask.scatter_(-1, topk_indices, 1.0)  # ones where kept
        return mask  # shape: (B, H, L, L)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, L, D)
            mask: (B, L) with 1 for valid tokens, 0 for pad (optional)
        Returns:
            out: (B, L, D)
            attention_info: dict with pattern weights and attention weights
        """
        B, L, D = x.shape
        device = x.device

        # QKV -> (3, B, H, L, head_dim)
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, L, head_dim)

        # scaled dot-product attention scores: (B, H, L, L)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # sanitize numeric edge cases early
        attention_scores = torch.nan_to_num(attention_scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        attention_scores = attention_scores.clamp(min=-1e9, max=1e9)

        # pattern selection based on pooled features: (B, 3)
        pooled_features = torch.mean(x, dim=1)
        pattern_weights = self.pattern_selector(pooled_features)  # (B, 3)

        pooled_features = torch.mean(x, dim=1)
        pattern_weights = self.pattern_selector(pooled_features)
        
        # ADD DEBUGGING HERE:
        if self.training:  # Only debug during training
            print(f"DEBUG Pooled features stats: mean={pooled_features.mean().item():.6f}, std={pooled_features.std().item():.6f}")
            print(f"DEBUG Pattern weights sample: {pattern_weights[0].detach().cpu().numpy()}")
            print(f"DEBUG Pattern weights std: {pattern_weights.std().item():.6f}")
            
            # Check pattern selector network outputs before softmax
            with torch.no_grad():
                pre_softmax = self.pattern_selector[:-1](pooled_features)  # Without softmax
                print(f"DEBUG Pre-softmax logits: {pre_softmax[0].detach().cpu().numpy()}")
            
            # Register gradient hook
            def grad_hook(grad):
                if grad is not None:
                    print(f"DEBUG Pattern weights gradient norm: {grad.norm().item():.8f}")
                else:
                    print("DEBUG Pattern weights gradient is None!")
            pattern_weights.register_hook(grad_hook)
    

        # create pattern masks (binary), expand to broadcast shapes
        local_mask_bin = self.create_local_mask(L, device).unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
        global_mask_bin = self.create_global_mask(L, device).unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
        sparse_mask_bin = self.create_learned_sparse_mask(attention_scores)  # (B,H,L,L) binary

        # expand pattern weights for broadcasting
        pw_local = pattern_weights[:, 0].view(B, 1, 1, 1)   # (B,1,1,1)
        pw_global = pattern_weights[:, 1].view(B, 1, 1, 1)
        pw_sparse = pattern_weights[:, 2].view(B, 1, 1, 1)

        # weighted binary masks -> (B,H,L,L)
        weighted_local = (pw_local * local_mask_bin).expand(B, self.num_heads, L, L)
        weighted_global = (pw_global * global_mask_bin).expand(B, self.num_heads, L, L)
        weighted_sparse = (pw_sparse * sparse_mask_bin)  # (B,H,L,L)

        # combine masks using soft-OR semantics (sum then threshold)
        combined_score = weighted_local + weighted_global + weighted_sparse  # (B,H,L,L)
        allowed_mask = combined_score > 0.0  # bool: allowed if any pattern allows

        # mask out disallowed positions by setting -inf
        attention_scores = attention_scores.masked_fill(~allowed_mask, float("-inf"))

        # apply input-level attention mask (keys) if given
        if mask is not None:
            mask = mask.to(dtype=torch.long, device=device)
            mask_sum = mask.sum(dim=1)
            if (mask_sum == 0).any():
                # avoid mutating external tensor
                mask = mask.clone()
                zero_idx = (mask_sum == 0).nonzero(as_tuple=False).squeeze(-1)
                mask[zero_idx, 0] = 1
                logger.debug(f"AdaptiveSparseAttention: fixed {zero_idx.numel()} all-zero attention_mask samples by enabling token 0")
            # broadcast mask for keys: (B,1,1,L)
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))

        # final sanitize before softmax
        attention_scores = torch.nan_to_num(attention_scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        attention_scores = attention_scores.clamp(min=-1e9, max=1e9)

        # find any fully-masked rows (all -inf) and repair (set one key to 0)
        row_valid = (attention_scores > -9e8).any(dim=-1)  # (B,H,L)
        if not row_valid.all():
            bad_idx = (~row_valid).nonzero(as_tuple=False)  # (N, 3) with (b,h,q)
            if bad_idx.numel() > 0:
                b_idx = bad_idx[:, 0]
                h_idx = bad_idx[:, 1]
                q_idx = bad_idx[:, 2]
                attention_scores[b_idx, h_idx, q_idx, 0] = 0.0
                logger.debug(f"AdaptiveSparseAttention: repaired {bad_idx.size(0)} fully-masked attention rows")

        # final numeric safety
        attention_scores = torch.nan_to_num(attention_scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        attention_scores = attention_scores.clamp(min=-1e9, max=1e9)

        # softmax -> attention weights
        attention_weights = F.softmax(attention_scores / float(self.temperature), dim=-1)
        attention_weights = self.dropout_layer(attention_weights)  # apply dropout to weights

        # attention output
        out = torch.matmul(attention_weights, v)  # (B,H,L,head_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, L, D)  # (B,L,D)

        out = self.proj(out)

        attention_info = {
            "pattern_weights": pattern_weights,  # (B,3) tensor
            "attention_weights": attention_weights,  # (B,H,L,L)
            "local_ratio": float(pattern_weights[:, 0].mean().item()),
            "global_ratio": float(pattern_weights[:, 1].mean().item()),
            "sparse_ratio": float(pattern_weights[:, 2].mean().item()),
        }

        return out, attention_info


class MultiHeadAdaptiveAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.attention = AdaptiveSparseAttention(dim=dim, num_heads=num_heads, dropout=dropout, **kwargs)

    def forward(self, x, mask=None):
        return self.attention(x, mask)
