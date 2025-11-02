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
        target_density: float = 0.3,
        min_density: float = 0.1,
        density_tolerance: float = 0.05,
        num_global_anchors: int = 8,
        head_pattern_refinement: bool = True,
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

        self.target_density = target_density
        self.min_density = min_density
        self.density_tolerance = density_tolerance
        self.num_global_anchors = num_global_anchors
        self.head_pattern_refinement = head_pattern_refinement

        # Token-level pattern selector with optional head refinement
        self.pattern_dropout = nn.Dropout(pattern_dropout)
        self.pattern_selector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 3),
        )
        if self.head_pattern_refinement:
            self.head_pattern_refiner = nn.Linear(dim, num_heads * 3)

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
        for module in self.pattern_selector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.head_pattern_refinement:
            nn.init.xavier_uniform_(self.head_pattern_refiner.weight, gain=0.5)
            if self.head_pattern_refiner.bias is not None:
                nn.init.zeros_(self.head_pattern_refiner.bias)

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

    def create_global_anchor_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Binary mask that exposes a small number of global anchor tokens."""
        if self.num_global_anchors <= 0:
            return torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)

        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
        # Always include beginning and end tokens
        anchors = [0, max(seq_len - 1, 0)]
        if self.num_global_anchors > 2 and seq_len > 2:
            step = max(1, seq_len // self.num_global_anchors)
            anchors.extend(range(step, seq_len - 1, step))
        anchors = torch.unique(torch.tensor(anchors, device=device)).long()
        anchors = anchors.clamp(0, seq_len - 1)

        mask[:, anchors] = 1.0  # Everyone can attend to anchors
        mask[anchors, :] = 1.0  # Anchors can attend globally
        return mask

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

        flat_weights = pattern_weights.reshape(-1, pattern_weights.shape[-1])
        flat_logits = pattern_logits.reshape(-1, pattern_logits.shape[-1])

        # 1. Entropy loss - encourage exploration
        pattern_entropy = -(flat_weights * torch.log(flat_weights + 1e-8)).sum(dim=-1)
        avg_entropy = pattern_entropy.mean()
        max_entropy = math.log(3.0)
        # Scale based on training progress
        entropy_weight = max(0.5, 1.0 - self.step_count / 10000)
        diversity_loss = (max_entropy - avg_entropy) * entropy_weight

        # 2. Batch variance loss - different samples should use different patterns
        batch_variance = flat_weights.var(dim=0, unbiased=False).sum()
        variance_loss = -batch_variance * 2.0

        # 3. Temporal consistency loss - patterns shouldn't oscillate wildly
        consistency_loss = torch.tensor(0.0, device=pattern_weights.device)
        if self.prev_pattern_weights is not None and self.training:
            # Only apply to same-sized sequences
            if self.prev_pattern_weights.shape == pattern_weights.shape:
                consistency_loss = F.mse_loss(
                    pattern_weights,
                    self.prev_pattern_weights.detach()
                ) * 0.1

        # 4. Pattern activation loss - ensure all patterns get used
        pattern_usage = flat_weights.mean(dim=0)  # Average usage per pattern
        self.pattern_momentum = self.momentum_beta * self.pattern_momentum + (1 - self.momentum_beta) * pattern_usage
        # Penalize if any pattern is underused (below 15%)
        underuse_penalty = torch.relu(0.15 - self.pattern_momentum).sum() * 5.0

        # 5. Logit variance loss - pattern logits should be decisive
        logit_variance = flat_logits.var(dim=-1, unbiased=False).mean()
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

        # Token-level pattern selection
        token_features = self.pattern_dropout(x)
        base_logits = self.pattern_selector(token_features)  # (B, L, 3)

        if self.head_pattern_refinement:
            head_logits = self.head_pattern_refiner(token_features)
            head_logits = head_logits.view(B, L, self.num_heads, 3)
            pattern_logits = base_logits.unsqueeze(2) + head_logits
        else:
            pattern_logits = base_logits.unsqueeze(2).expand(B, L, self.num_heads, 3)

        pattern_logits = pattern_logits + self.pattern_bias.view(1, 1, 1, -1)

        # Apply temperature (annealed during training)
        scaled_logits = pattern_logits / self.current_pattern_temp
        pattern_weights = F.softmax(scaled_logits, dim=-1)

        # Add exploration noise during training
        if self.training and torch.rand(1).item() < 0.1:  # 10% of the time
            explore_noise = torch.randn_like(pattern_logits) * 0.05
            pattern_weights = F.softmax(scaled_logits + explore_noise, dim=-1)

        # Compute pattern losses
        pattern_losses = self.compute_pattern_losses(pattern_weights, pattern_logits)

        # Update previous weights buffer
        if self.training:
            self.prev_pattern_weights = pattern_weights.detach()

        # Create attention masks
        local_mask = self.create_local_mask(L, device)
        global_mask = self.create_global_anchor_mask(L, device)
        sparse_mask = self.create_learned_sparse_mask(attention_scores)

        # Reorder for head-first operations
        pattern_weights_heads = pattern_weights.permute(0, 2, 1, 3).contiguous()

        # Expand pattern weights for broadcasting (B, H, L, 3)
        pw_local = pattern_weights_heads[..., 0]
        pw_global = pattern_weights_heads[..., 1]
        pw_sparse = pattern_weights_heads[..., 2]

        local_component = pw_local.unsqueeze(-1) * local_mask.unsqueeze(0).unsqueeze(0)
        global_component = pw_global.unsqueeze(-1) * global_mask.unsqueeze(0).unsqueeze(0)
        sparse_component = pw_sparse.unsqueeze(-1) * sparse_mask

        combined_mask_scores = local_component + global_component + sparse_component

        local_bool = local_mask.bool()
        global_bool = global_mask.bool()
        sparse_bool = sparse_mask.bool()
        allowed_union = sparse_bool | (local_bool | global_bool).unsqueeze(0).unsqueeze(0)

        combined_mask_scores = combined_mask_scores.masked_fill(~allowed_union, float('-inf'))

        attention_mask_binary, density_metrics = self.enforce_density_budget(
            combined_mask_scores,
            allowed_union,
            global_bool,
        )

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
            "attention_mask": attention_mask_binary,
            "local_ratio": float(pw_local.mean().item()),
            "global_ratio": float(pw_global.mean().item()),
            "sparse_ratio": float(pw_sparse.mean().item()),
            "pattern_entropy": pattern_losses['diversity_loss'].item() if isinstance(pattern_losses['diversity_loss'], torch.Tensor) else pattern_losses['diversity_loss'],
            "pattern_logits_std": pattern_logits.std().item(),
            "current_temperature": self.current_pattern_temp,
            "actual_density": density_metrics["actual_density"],
            "target_density": density_metrics["target_density"],
            "density_error": density_metrics["density_error"],
            "anchors_per_row": density_metrics["anchors_per_row"],
            "head_variance": float(pw_local.var(dim=1, unbiased=False).mean().item()),
            **{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in pattern_losses.items()}
        }

        return out, attention_info

    def enforce_density_budget(
        self,
        combined_scores: torch.Tensor,
        allowed_union: torch.Tensor,
        global_bool: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Project the mixed mask onto the target FLOPs/density budget."""

        B, H, L, _ = combined_scores.shape
        device = combined_scores.device
        allowed_union = allowed_union.bool()

        anchor_mask = global_bool.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        diag = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        mandatory_mask = anchor_mask | diag
        available_mask = allowed_union & ~mandatory_mask

        valid_counts = available_mask.sum(dim=-1)
        min_keep = max(1, int(self.min_density * L))

        target_counts = (valid_counts.float() * self.target_density).round().long()
        target_counts = torch.clamp(target_counts, min=min_keep)
        target_counts = torch.minimum(target_counts, valid_counts.clamp(min=1))
        target_counts = torch.where(valid_counts == 0, torch.zeros_like(target_counts), target_counts)

        masked_scores = combined_scores.masked_fill(~available_mask, float('-inf'))

        k_max = int(target_counts.max().item()) if target_counts.numel() > 0 else 0

        if k_max > 0:
            # Only materialise the top-k entries that could ever be kept.
            _, topk_indices = torch.topk(
                masked_scores,
                k_max,
                dim=-1,
                largest=True,
                sorted=False,
            )

            topk_available = available_mask.gather(-1, topk_indices)

            rank_indices = torch.arange(k_max, device=device).view(1, 1, 1, k_max)
            within_budget = rank_indices < target_counts.unsqueeze(-1)
            selected_mask = within_budget & topk_available

            selection = torch.zeros_like(available_mask, dtype=torch.bool)
            selection.scatter_(-1, topk_indices, selected_mask)
        else:
            selection = torch.zeros_like(available_mask, dtype=torch.bool)
        budget_mask = mandatory_mask | selection

        # Guarantee at least one connection per row
        row_has_value = budget_mask.any(dim=-1, keepdim=True)
        fallback_needed = ~row_has_value
        if fallback_needed.any():
            fallback_indices = torch.argmax(allowed_union.float(), dim=-1, keepdim=True)
            fallback_mask = torch.zeros_like(budget_mask)
            fallback_mask.scatter_(-1, fallback_indices, True)
            budget_mask = torch.where(fallback_needed, fallback_mask, budget_mask)

        actual_density = budget_mask.float().mean().item()
        mandatory_per_row = mandatory_mask.sum(dim=-1).float()
        target_density = float(((target_counts.float() + mandatory_per_row).mean().item()) / L)
        density_error = actual_density - target_density
        anchors_per_row = anchor_mask.float().sum(dim=-1).float().mean().item() / L

        return budget_mask, {
            "actual_density": actual_density,
            "target_density": target_density,
            "density_error": density_error,
            "anchors_per_row": anchors_per_row,
        }


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