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
    (local / global / learned-sparse) per sequence via a pattern selector network.
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
        pattern_temperature: float = 0.3,  # Lower temperature for sharper selection
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
        self.pattern_temperature = pattern_temperature

        # QKV and output projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout_layer = nn.Dropout(self.dropout_p)

        # Pattern selector: deeper network for better pattern recognition
        self.pattern_selector = nn.Sequential(
            nn.Linear(dim, dim),          # Larger first layer
            nn.ReLU(),
            nn.Dropout(0.2),             # Higher dropout
            nn.Linear(dim, dim // 2),    # Second layer
            nn.ReLU(),
            nn.Linear(dim // 2, 3),      # Output logits
        )

        # Learnable pattern bias to break symmetry
        self.pattern_bias = nn.Parameter(torch.tensor([0.2, -0.1, -0.1]))

        # Per-head learnable sparsity parameters
        self.learnable_sparsity = learnable_sparsity
        if learnable_sparsity:
            self.sparse_pattern_weights = nn.Parameter(torch.randn(num_heads, 1, 1))
            self.sparse_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

        # Improved initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with aggressive biases to break pattern symmetry."""
        # Pattern selector - use larger initialization and biased final layer
        for i, module in enumerate(self.pattern_selector):
            if isinstance(module, nn.Linear):
                if i == len(self.pattern_selector) - 1:  # Final output layer
                    # Large gain to ensure strong initial logits
                    nn.init.xavier_normal_(module.weight, gain=3.0)
                    if module.bias is not None:
                        # Strong bias toward local pattern initially
                        module.bias.data = torch.tensor([1.0, -0.5, -0.5])
                else:
                    nn.init.xavier_normal_(module.weight, gain=1.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # QKV and projection layers
        nn.init.xavier_normal_(self.qkv.weight, gain=0.5)
        nn.init.xavier_normal_(self.proj.weight, gain=0.5)
        
        # Sparse pattern parameters
        if self.learnable_sparsity:
            nn.init.normal_(self.sparse_pattern_weights, mean=0.0, std=0.3)
            nn.init.zeros_(self.sparse_bias)

    def create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Binary local mask (seq_len, seq_len) with 1 where attention is allowed."""
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
        """Create learned sparse binary mask from attention scores."""
        B, H, L, _ = attention_scores.shape
        k = max(1, min(L, int(L * (1 - sparsity_ratio))))

        # Apply per-head learnable transformation
        if self.learnable_sparsity:
            w = self.sparse_pattern_weights.view(1, H, 1, 1)
            b = self.sparse_bias.view(1, H, 1, 1)
            scores = attention_scores * w + b
        else:
            scores = attention_scores

        # Top-k selection with jitter to break ties
        jitter = torch.randn_like(scores) * 1e-6
        scores_jittered = scores + jitter
        
        _, topk_indices = torch.topk(scores_jittered, k, dim=-1)
        
        # Create binary mask
        mask = torch.zeros_like(scores, dtype=torch.float32)
        mask.scatter_(-1, topk_indices, 1.0)
        return mask

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, L, D) input tokens
            mask: (B, L) attention mask with 1 for valid tokens, 0 for padding
        Returns:
            output: (B, L, D) 
            attention_info: dict with pattern statistics
        """
        B, L, D = x.shape
        device = x.device

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Enhanced pattern selection
        pooled_features = torch.mean(x, dim=1)  # (B, D)
        
        # Add sequence length as additional feature
        seq_length_feature = torch.full((B, 1), L / 512.0, device=device, dtype=x.dtype)
        
        # For now, use only pooled features (can enhance later)
        pattern_logits = self.pattern_selector(pooled_features)  # (B, 3)
        
        # Add learnable bias to break symmetry
        pattern_logits = pattern_logits + self.pattern_bias.unsqueeze(0)
        
        # Apply sharp temperature for decisive selection
        pattern_weights = F.softmax(pattern_logits / self.pattern_temperature, dim=-1)

        # Enhanced debugging with gradient tracking
        if self.training and torch.rand(1).item() < 0.05:  # Debug 5% of batches
            print(f"DEBUG Pattern logits: {pattern_logits[0].detach().cpu().numpy()}")
            print(f"DEBUG Pattern weights: {pattern_weights[0].detach().cpu().numpy()}")
            print(f"DEBUG Pattern logits std: {pattern_logits.std().item():.6f}")
            print(f"DEBUG Pooled features std: {pooled_features.std().item():.6f}")
            
            # Track gradients
            if pattern_logits.requires_grad:
                def grad_hook(grad):
                    if grad is not None:
                        print(f"DEBUG Pattern logits grad norm: {grad.norm().item():.8f}")
                    else:
                        print("DEBUG Pattern logits grad is None!")
                pattern_logits.register_hook(grad_hook)

        # Create binary pattern masks
        local_mask = self.create_local_mask(L, device)  # (L, L)
        global_mask = self.create_global_mask(L, device)  # (L, L)  
        sparse_mask = self.create_learned_sparse_mask(attention_scores)  # (B, H, L, L)

        # Expand pattern weights for broadcasting
        pw_local = pattern_weights[:, 0].view(B, 1, 1, 1)   # (B, 1, 1, 1)
        pw_global = pattern_weights[:, 1].view(B, 1, 1, 1)
        pw_sparse = pattern_weights[:, 2].view(B, 1, 1, 1)

        # Combine masks using weighted combination
        combined_mask = (
            pw_local * local_mask.unsqueeze(0).unsqueeze(0) +    # (B, H, L, L)
            pw_global * global_mask.unsqueeze(0).unsqueeze(0) +  # (B, H, L, L)
            pw_sparse * sparse_mask                               # (B, H, L, L)
        )

        # Apply combined mask - threshold for attention positions
        threshold = 0.05  # Lower threshold for more selective attention
        attention_mask = combined_mask > threshold

        # Mask attention scores
        attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf'))

        # Apply input padding mask if provided
        if mask is not None:
            # Ensure at least one token is unmasked per sequence
            mask_sum = mask.sum(dim=1)
            if (mask_sum == 0).any():
                mask = mask.clone()
                zero_mask_idx = (mask_sum == 0).nonzero(as_tuple=True)[0]
                mask[zero_mask_idx, 0] = 1  # Unmask first token
            
            # Apply padding mask to keys
            key_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attention_scores = attention_scores.masked_fill(key_mask == 0, float('-inf'))

        # Handle fully masked rows - repair by unmasking first position
        all_masked = (attention_scores == float('-inf')).all(dim=-1)  # (B, H, L)
        if all_masked.any():
            attention_scores = attention_scores.clone()
            attention_scores[all_masked, 0] = 0.0

        # Apply attention with temperature
        attention_weights = F.softmax(attention_scores / self.temperature, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        # Compute output
        out = torch.matmul(attention_weights, v)  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        out = self.proj(out)

        # Enhanced diversity losses
        pattern_entropy = -(pattern_weights * torch.log(pattern_weights + 1e-8)).sum(dim=-1)
        avg_entropy = pattern_entropy.mean()
        
        # Strong diversity penalty
        max_entropy = math.log(3.0)
        diversity_loss = -(avg_entropy - max_entropy * 0.3)  # Encourage high entropy
        
        # Pattern specialization loss - penalize uniform usage across batch
        pattern_mean = pattern_weights.mean(dim=0)  # (3,) - average usage per pattern
        specialization_loss = -((pattern_mean - 1.0/3.0)**2).sum()  # Penalize uniform

        # Attention info for logging and loss computation
        attention_info = {
            "pattern_weights": pattern_weights,  # (B, 3)
            "attention_weights": attention_weights,  # (B, H, L, L)
            "local_ratio": float(pattern_weights[:, 0].mean().item()),
            "global_ratio": float(pattern_weights[:, 1].mean().item()),
            "sparse_ratio": float(pattern_weights[:, 2].mean().item()),
            "diversity_loss": diversity_loss,
            "specialization_loss": specialization_loss,
            "pattern_entropy": avg_entropy,
            "pattern_logits_std": pattern_logits.std().item(),
            "total_pattern_loss": diversity_loss + 0.1 * specialization_loss,  # Combined loss
        }

        return out, attention_info


class MultiHeadAdaptiveAttention(nn.Module):
    """Wrapper for compatibility."""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.attention = AdaptiveSparseAttention(
            dim=dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            **kwargs
        )

    def forward(self, x, mask=None):
        return self.attention(x, mask)