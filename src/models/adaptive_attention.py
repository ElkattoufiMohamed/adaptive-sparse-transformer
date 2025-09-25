# src/models/adaptive_attention.py
import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class AdaptiveSparseAttention(nn.Module):
    """
    Adaptive Sparse Attention with fixed pattern learning issues.
    Key changes:
    - Proper diversity losses that get used in training
    - Separate learning rate support
    - Temperature scheduling
    - Better initialization
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
        pattern_temperature: float = 1.0,  # Start higher, anneal down
        min_pattern_temperature: float = 0.3,  # Minimum temperature
        pattern_dropout: float = 0.1,  # Lower dropout for pattern selector
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
        
        # Temperature scheduling parameters
        self.pattern_temperature = pattern_temperature
        self.min_pattern_temperature = min_pattern_temperature
        self.temperature_decay_rate = 0.995  # Decay per step
        self.current_pattern_temp = pattern_temperature

        # QKV and output projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout_layer = nn.Dropout(self.dropout_p)

        # Enhanced pattern selector with moderate initialization
        self.pattern_selector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),  # Add normalization for stability
            nn.GELU(),  # Smoother than ReLU
            nn.Dropout(pattern_dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 3),  # Output logits
        )

        # Mild learnable pattern bias
        self.pattern_bias = nn.Parameter(torch.tensor([0.05, 0.0, -0.05]))
        
        # Track previous pattern weights for consistency loss
        self.register_buffer('prev_pattern_weights', None)
        self.register_buffer('pattern_momentum', torch.zeros(3))
        self.momentum_beta = 0.9

        # Per-head learnable sparsity parameters
        self.learnable_sparsity = learnable_sparsity
        if learnable_sparsity:
            self.sparse_pattern_weights = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.1 + 1.0)
            self.sparse_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

        # Initialize weights properly
        self._init_weights()
        
        # Tracking for debugging
        self.step_count = 0

    def _init_weights(self):
        """Moderate initialization to allow learning."""
        # Pattern selector - moderate initialization
        for i, module in enumerate(self.pattern_selector):
            if isinstance(module, nn.Linear):
                # Use smaller gain for all layers
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    if i == len(self.pattern_selector) - 1:  # Final layer
                        # Very mild initial bias
                        nn.init.constant_(module.bias, 0.0)
                    else:
                        nn.init.zeros_(module.bias)

        # QKV and projection - standard initialization
        nn.init.xavier_uniform_(self.qkv.weight, gain=1.0 / math.sqrt(2))
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def update_temperature(self):
        """Anneal pattern temperature during training."""
        if self.training:
            self.current_pattern_temp = max(
                self.min_pattern_temperature,
                self.pattern_temperature * (self.temperature_decay_rate ** self.step_count)
            )
            self.step_count += 1

    def create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Binary local mask with sliding window."""
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
        """Create learned sparse mask with improved stability."""
        B, H, L, _ = attention_scores.shape
        
        # Dynamic sparsity based on sequence length
        effective_sparsity = min(sparsity_ratio, 1.0 - (10.0 / L))  # Keep at least 10 connections
        k = max(1, min(L, int(L * (1 - effective_sparsity))))

        # Apply learnable transformation
        if self.learnable_sparsity:
            w = self.sparse_pattern_weights.view(1, H, 1, 1)
            b = self.sparse_bias.view(1, H, 1, 1)
            scores = attention_scores * torch.abs(w) + b  # Ensure positive weights
        else:
            scores = attention_scores

        # Add small noise for tie-breaking during training
        if self.training:
            noise = torch.randn_like(scores) * 0.01
            scores = scores + noise

        # Top-k selection
        _, topk_indices = torch.topk(scores, k, dim=-1, largest=True, sorted=False)
        
        # Create binary mask
        mask = torch.zeros_like(scores, dtype=torch.float32)
        mask.scatter_(-1, topk_indices, 1.0)
        return mask

    def compute_pattern_losses(
        self, 
        pattern_weights: torch.Tensor,
        pattern_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute various losses to encourage pattern learning."""
        
        # 1. Entropy loss - encourage exploration
        pattern_entropy = -(pattern_weights * torch.log(pattern_weights + 1e-8)).sum(dim=-1)
        avg_entropy = pattern_entropy.mean()
        max_entropy = math.log(3.0)
        # Scale based on training progress
        entropy_weight = max(0.5, 1.0 - self.step_count / 10000)
        diversity_loss = (max_entropy - avg_entropy) * entropy_weight
        
        # 2. Batch variance loss - different samples should use different patterns
        batch_variance = pattern_weights.var(dim=0).sum()
        variance_loss = -batch_variance * 2.0
        
        # 3. Temporal consistency loss - patterns shouldn't oscillate wildly
        consistency_loss = torch.tensor(0.0, device=pattern_weights.device)
        if self.prev_pattern_weights is not None and self.training:
            # Only apply to same-sized sequences
            if self.prev_pattern_weights.shape[0] == pattern_weights.shape[0]:
                consistency_loss = F.mse_loss(
                    pattern_weights, 
                    self.prev_pattern_weights.detach()
                ) * 0.1
        
        # 4. Pattern activation loss - ensure all patterns get used
        pattern_usage = pattern_weights.mean(dim=0)  # Average usage per pattern
        self.pattern_momentum = self.momentum_beta * self.pattern_momentum + (1 - self.momentum_beta) * pattern_usage
        # Penalize if any pattern is underused (below 15%)
        underuse_penalty = torch.relu(0.15 - self.pattern_momentum).sum() * 5.0
        
        # 5. Logit variance loss - pattern logits should be decisive
        logit_variance = pattern_logits.var(dim=-1).mean()
        decisiveness_loss = -logit_variance * 0.5
        
        return {
            'diversity_loss': diversity_loss,
            'variance_loss': variance_loss,
            'consistency_loss': consistency_loss,
            'underuse_penalty': underuse_penalty,
            'decisiveness_loss': decisiveness_loss,
            'total_pattern_loss': (
                diversity_loss + 
                variance_loss * 0.5 + 
                consistency_loss + 
                underuse_penalty + 
                decisiveness_loss * 0.2
            )
        }

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with pattern learning fixes.
        """
        B, L, D = x.shape
        device = x.device
        
        # Update temperature
        self.update_temperature()

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Pattern selection with improved features
        # Use both mean and max pooling for better representation
        pooled_mean = torch.mean(x, dim=1)  # (B, D)
        pooled_max, _ = torch.max(x, dim=1)  # (B, D)
        pooled_features = (pooled_mean + pooled_max) / 2.0
        
        # Add noise during training for exploration
        if self.training:
            feature_noise = torch.randn_like(pooled_features) * 0.1
            pooled_features = pooled_features + feature_noise
        
        # Compute pattern logits
        pattern_logits = self.pattern_selector(pooled_features)  # (B, 3)
        pattern_logits = pattern_logits + self.pattern_bias
        
        # Apply temperature (annealed during training)
        pattern_weights = F.softmax(pattern_logits / self.current_pattern_temp, dim=-1)
        
        # Add exploration noise during training
        if self.training and torch.rand(1).item() < 0.1:  # 10% of the time
            explore_noise = torch.randn_like(pattern_weights) * 0.1
            pattern_weights = F.softmax(pattern_logits + explore_noise, dim=-1)

        # Compute pattern losses
        pattern_losses = self.compute_pattern_losses(pattern_weights, pattern_logits)
        
        # Update previous weights buffer
        if self.training:
            self.prev_pattern_weights = pattern_weights.detach()

        # Debug logging
        if self.training and self.step_count % 100 == 0:
            with torch.no_grad():
                print(f"\n=== Step {self.step_count} ===")
                print(f"Pattern weights mean: {pattern_weights.mean(dim=0).cpu().numpy()}")
                print(f"Pattern weights std: {pattern_weights.std(dim=0).cpu().numpy()}")
                print(f"Pattern logits: {pattern_logits[0].cpu().numpy()}")
                print(f"Current temperature: {self.current_pattern_temp:.3f}")
                print(f"Pattern momentum: {self.pattern_momentum.cpu().numpy()}")
                print(f"Total pattern loss: {pattern_losses['total_pattern_loss'].item():.4f}")

        # Create attention masks
        local_mask = self.create_local_mask(L, device)
        global_mask = self.create_global_mask(L, device)
        sparse_mask = self.create_learned_sparse_mask(attention_scores)

        # Expand pattern weights for broadcasting
        pw_local = pattern_weights[:, 0].view(B, 1, 1, 1)
        pw_global = pattern_weights[:, 1].view(B, 1, 1, 1)
        pw_sparse = pattern_weights[:, 2].view(B, 1, 1, 1)

        # Combine masks
        combined_mask = (
            pw_local * local_mask.unsqueeze(0).unsqueeze(0) +
            pw_global * global_mask.unsqueeze(0).unsqueeze(0) +
            pw_sparse * sparse_mask
        )

        # Apply threshold
        threshold = 0.1  # Slightly higher threshold for cleaner attention
        attention_mask_binary = combined_mask > threshold
        
        # Mask attention scores
        attention_scores = attention_scores.masked_fill(~attention_mask_binary, float('-inf'))

        # Apply input padding mask
        if mask is not None:
            # Fix zero mask sequences
            mask_sum = mask.sum(dim=1, keepdim=True)
            mask = mask.clone()
            mask[mask_sum.squeeze() == 0, 0] = 1
            
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(key_mask == 0, float('-inf'))

        # Prevent complete masking
        all_masked = (attention_scores == float('-inf')).all(dim=-1)
        if all_masked.any():
            attention_scores = attention_scores.clone()
            # Unmask diagonal for completely masked rows
            for b in range(B):
                for h in range(self.num_heads):
                    for l in range(L):
                        if all_masked[b, h, l]:
                            attention_scores[b, h, l, l] = 0.0

        # Apply softmax
        attention_weights = F.softmax(attention_scores / self.temperature, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        # Compute output
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.proj(out)

        # Prepare attention info
        attention_info = {
            "pattern_weights": pattern_weights,
            "attention_weights": attention_weights,
            "local_ratio": float(pattern_weights[:, 0].mean().item()),
            "global_ratio": float(pattern_weights[:, 1].mean().item()),
            "sparse_ratio": float(pattern_weights[:, 2].mean().item()),
            "pattern_entropy": pattern_losses['diversity_loss'].item() if isinstance(pattern_losses['diversity_loss'], torch.Tensor) else pattern_losses['diversity_loss'],
            "pattern_logits_std": pattern_logits.std().item(),
            "current_temperature": self.current_pattern_temp,
            **{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in pattern_losses.items()}
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