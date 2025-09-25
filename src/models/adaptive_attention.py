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
        enable_pattern_perturbation: bool = True,  # Enable random pattern perturbation
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
        self.enable_pattern_perturbation = enable_pattern_perturbation

        # QKV and output projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout_layer = nn.Dropout(self.dropout_p)

        # Pattern selector: deeper network for better pattern recognition
        self.pattern_selector = nn.Sequential(
            nn.Linear(dim, dim),          # Larger first layer
            nn.ReLU(),
            nn.Dropout(0.3),             # Higher dropout to prevent overfitting
            nn.Linear(dim, dim // 2),    # Second layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, 3),      # Output logits
        )

        # Learnable pattern bias to break symmetry - initialized with stronger bias
        self.pattern_bias = nn.Parameter(torch.tensor([0.5, -0.3, -0.2]))

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
                    # Very large gain to ensure strong initial logits
                    nn.init.xavier_normal_(module.weight, gain=5.0)
                    if module.bias is not None:
                        # Very strong bias toward local pattern initially
                        module.bias.data = torch.tensor([1.5, -0.8, -0.7])
                else:
                    nn.init.xavier_normal_(module.weight, gain=2.0)
                    if module.bias is not None:
                        nn.init.normal_(module.bias, mean=0.0, std=0.1)

        # QKV and projection layers
        nn.init.xavier_normal_(self.qkv.weight, gain=0.5)
        nn.init.xavier_normal_(self.proj.weight, gain=0.5)
        
        # Sparse pattern parameters
        if self.learnable_sparsity:
            nn.init.normal_(self.sparse_pattern_weights, mean=0.0, std=0.5)
            nn.init.normal_(self.sparse_bias, mean=0.0, std=0.1)

    def get_pattern_parameters(self):
        """Return parameters specific to pattern selection for separate optimization."""
        pattern_params = []
        pattern_params.extend(self.pattern_selector.parameters())
        pattern_params.append(self.pattern_bias)
        return pattern_params

    def get_non_pattern_parameters(self):
        """Return all parameters except pattern selection ones."""
        pattern_param_ids = {id(p) for p in self.get_pattern_parameters()}
        return [p for p in self.parameters() if id(p) not in pattern_param_ids]

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

        # Enhanced pattern selection with features
        pooled_features = torch.mean(x, dim=1)  # (B, D)
        
        # Add sequence length and variance as additional features
        seq_length_feature = torch.full((B, 1), L / 512.0, device=device, dtype=x.dtype)
        seq_variance = torch.var(x, dim=1, keepdim=True).mean(dim=-1, keepdim=True)  # (B, 1)
        
        # Enhance pooled features with additional information
        enhanced_features = torch.cat([
            pooled_features,
            seq_length_feature,
            seq_variance
        ], dim=1)
        
        # Project enhanced features back to original dimension for pattern selector
        if enhanced_features.size(1) != self.dim:
            # Simple projection to match expected input size
            projection = nn.Linear(enhanced_features.size(1), self.dim).to(device)
            pattern_input = projection(enhanced_features)
        else:
            pattern_input = enhanced_features
        
        # Get pattern logits
        pattern_logits = self.pattern_selector(pattern_input)  # (B, 3)
        
        # Add learnable bias to break symmetry
        pattern_logits = pattern_logits + self.pattern_bias.unsqueeze(0)
        
        # Apply random perturbation during training to prevent freezing
        if self.training and self.enable_pattern_perturbation:
            if torch.rand(1).item() < 0.02:  # 2% of training steps
                perturbation = torch.randn_like(self.pattern_bias) * 0.2
                self.pattern_bias.data += perturbation
                logger.debug(f"Applied pattern bias perturbation: {perturbation}")
        
        # Apply sharp temperature for decisive selection
        pattern_weights = F.softmax(pattern_logits / self.pattern_temperature, dim=-1)

        # Enhanced debugging with comprehensive gradient tracking
        if self.training and torch.rand(1).item() < 0.1:  # Debug 10% of batches
            print(f"DEBUG Pattern logits: {pattern_logits[0].detach().cpu().numpy()}")
            print(f"DEBUG Pattern weights: {pattern_weights[0].detach().cpu().numpy()}")
            print(f"DEBUG Pattern logits std: {pattern_logits.std().item():.6f}")
            print(f"DEBUG Pattern bias: {self.pattern_bias.detach().cpu().numpy()}")
            print(f"DEBUG Pooled features std: {pooled_features.std().item():.6f}")
            
            # Track gradients for pattern selector
            def pattern_grad_hook(grad):
                if grad is not None:
                    print(f"DEBUG Pattern logits grad norm: {grad.norm().item():.8f}")
                    print(f"DEBUG Pattern logits grad mean: {grad.mean().item():.8f}")
                else:
                    print("DEBUG Pattern logits grad is None!")
            
            if pattern_logits.requires_grad:
                pattern_logits.register_hook(pattern_grad_hook)
            
            # Track pattern bias gradients
            def bias_grad_hook(grad):
                if grad is not None:
                    print(f"DEBUG Pattern bias grad norm: {grad.norm().item():.8f}")
                else:
                    print("DEBUG Pattern bias grad is None!")
                    
            if self.pattern_bias.requires_grad:
                self.pattern_bias.register_hook(bias_grad_hook)

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
        threshold = 0.02  # Very low threshold for more selective attention
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

        # Enhanced diversity losses with stronger penalties
        pattern_entropy = -(pattern_weights * torch.log(pattern_weights + 1e-8)).sum(dim=-1)
        avg_entropy = pattern_entropy.mean()
        
        # Very strong diversity penalty - heavily penalize low entropy
        max_entropy = math.log(3.0)
        diversity_loss = -2.0 * (avg_entropy - max_entropy * 0.5)  # Stronger penalty
        
        # Pattern specialization loss - encourage different usage across batch
        pattern_mean = pattern_weights.mean(dim=0)  # (3,) - average usage per pattern
        target_dist = torch.tensor([0.5, 0.25, 0.25], device=device)  # Target distribution
        specialization_loss = -5.0 * ((pattern_mean - target_dist)**2).sum()  # Strong penalty
        
        # Pattern variance loss - encourage high variance in pattern selection
        pattern_var = pattern_weights.var(dim=0).sum()  # Sum of variances across patterns
        variance_loss = -pattern_var  # Encourage high variance
        
        # Combined pattern learning loss
        total_pattern_loss = diversity_loss + specialization_loss + 0.5 * variance_loss

        # Attention info for logging and loss computation
        attention_info = {
            "pattern_weights": pattern_weights,  # (B, 3)
            "attention_weights": attention_weights,  # (B, H, L, L)
            "local_ratio": float(pattern_weights[:, 0].mean().item()),
            "global_ratio": float(pattern_weights[:, 1].mean().item()),
            "sparse_ratio": float(pattern_weights[:, 2].mean().item()),
            "diversity_loss": diversity_loss,
            "specialization_loss": specialization_loss,
            "variance_loss": variance_loss,
            "pattern_entropy": avg_entropy,
            "pattern_logits_std": pattern_logits.std().item(),
            "total_pattern_loss": total_pattern_loss,
            "pattern_bias": self.pattern_bias.clone(),  # For monitoring
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

    def get_pattern_parameters(self):
        """Expose pattern parameters for separate optimization."""
        return self.attention.get_pattern_parameters()
    
    def get_non_pattern_parameters(self):
        """Expose non-pattern parameters for separate optimization."""
        return self.attention.get_non_pattern_parameters()


# Helper function to create separate optimizers
def create_separate_optimizers(model, main_lr=1e-4, pattern_lr=1e-2):
    """
    Create separate optimizers for pattern selector and other parameters.
    
    Args:
        model: The transformer model containing adaptive attention
        main_lr: Learning rate for main model parameters
        pattern_lr: Learning rate for pattern selector (should be much higher)
    
    Returns:
        main_optimizer, pattern_optimizer
    """
    # Collect pattern selector parameters from all adaptive attention modules
    pattern_params = []
    main_params = []
    
    for module in model.modules():
        if isinstance(module, (AdaptiveSparseAttention, MultiHeadAdaptiveAttention)):
            if hasattr(module, 'get_pattern_parameters'):
                pattern_params.extend(module.get_pattern_parameters())
            if hasattr(module, 'get_non_pattern_parameters'):
                main_params.extend(module.get_non_pattern_parameters())
    
    # If no adaptive attention modules found, fall back to parameter name filtering
    if not pattern_params:
        pattern_params = [p for name, p in model.named_parameters() 
                         if any(keyword in name for keyword in ['pattern_selector', 'pattern_bias'])]
        main_params = [p for name, p in model.named_parameters() 
                      if not any(keyword in name for keyword in ['pattern_selector', 'pattern_bias'])]
    
    main_optimizer = torch.optim.AdamW(main_params, lr=main_lr, weight_decay=0.01)
    pattern_optimizer = torch.optim.AdamW(pattern_params, lr=pattern_lr, weight_decay=0.001)
    
    return main_optimizer, pattern_optimizer