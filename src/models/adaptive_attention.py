import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class AdaptiveSparseAttention(nn.Module):
    """
    Adaptive Sparse Attention that dynamically selects attention patterns
    (local / global / learned-sparse) per sequence via a pattern selector network.

    Key improvements:
      - learnable pattern temperature (log_pattern_tau) to scale logits
      - optional Gumbel-Softmax selection (use_gumbel)
      - entropy penalty and diversity loss with configurable weights
      - helper param_groups() to expose selector params for larger LR
      - debug flag for printing activation/grad checks
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
        pattern_temperature: float = 0.5,
        # New options:
        use_gumbel: bool = False,
        gumbel_init_temp: float = 1.0,
        entropy_lambda: float = 0.0,
        diversity_lambda: float = 0.0,
        debug: bool = False,
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

        # pattern temperature (learnable log)
        self.pattern_temperature = pattern_temperature
        self.log_pattern_tau = nn.Parameter(torch.tensor(math.log(max(pattern_temperature, 1e-6))))
        # gumbel-softmax option
        self.use_gumbel = use_gumbel
        self.gumbel_temp = nn.Parameter(torch.tensor(gumbel_init_temp))
        # regularization weights
        self.entropy_lambda = entropy_lambda
        self.diversity_lambda = diversity_lambda
        self.debug = debug

        # QKV and output projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout_layer = nn.Dropout(self.dropout_p)

        # Pattern selector: sequence-level MLP -> (local, global, sparse) logits
        self.pattern_selector = nn.Sequential(
            nn.Linear(dim, max(64, dim // 2)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(max(64, dim // 2), 3),
        )

        # Per-head learnable sparsity parameters
        self.learnable_sparsity = learnable_sparsity
        if learnable_sparsity:
            # shape (H, 1, 1) broadcastable to (B, H, L, L) when needed
            self.sparse_pattern_weights = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.2)
            self.sparse_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

        # Improved initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling for learning."""
        for module in self.pattern_selector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    # small non-zero bias to break symmetry
                    nn.init.normal_(module.bias, mean=1e-3, std=1e-3)

        nn.init.xavier_normal_(self.qkv.weight, gain=0.5)
        nn.init.xavier_normal_(self.proj.weight, gain=0.5)

        if self.learnable_sparsity:
            nn.init.normal_(self.sparse_pattern_weights, mean=0.0, std=0.2)
            nn.init.zeros_(self.sparse_bias)

    @torch.no_grad()
    def create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Binary local mask (seq_len, seq_len) with 1 where attention is allowed."""
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
        half = self.local_window_size // 2
        # vectorized fill for speed
        idxs = torch.arange(seq_len, device=device)
        starts = (idxs - half).clamp(min=0)
        ends = (idxs + half + 1).clamp(max=seq_len)
        for i in range(seq_len):
            mask[i, starts[i]:ends[i]] = 1.0
        return mask

    @torch.no_grad()
    def create_global_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Binary global mask (all ones)."""
        return torch.ones((seq_len, seq_len), device=device, dtype=torch.float32)

    def create_learned_sparse_mask(
        self, attention_scores: torch.Tensor, sparsity_ratio: float = 0.3
    ) -> torch.Tensor:
        """Create learned sparse binary mask from attention scores.
        attention_scores: (B, H, L, L)
        returns: (B, H, L, L) binary mask float32
        """
        B, H, L, _ = attention_scores.shape
        # k is number of positions to keep; ensure >=1
        k = max(1, int(L * (1 - sparsity_ratio)))

        if self.learnable_sparsity:
            # broadcast weights/bias (1, H, 1, 1)
            w = self.sparse_pattern_weights.view(1, H, 1, 1)
            b = self.sparse_bias.view(1, H, 1, 1)
            scores = attention_scores * w + b
        else:
            scores = attention_scores

        # small jitter to break exact ties
        jitter = torch.randn_like(scores) * 1e-6
        scores_jittered = scores + jitter

        # topk across last dim
        topk_vals, topk_indices = torch.topk(scores_jittered, k=min(k, L), dim=-1)

        mask = torch.zeros_like(scores, dtype=torch.float32)
        # scatter 1s into mask
        mask.scatter_(-1, topk_indices, 1.0)
        return mask

    def param_groups(self, base_lr: float = 1e-4, selector_lr_scale: float = 8.0):
        """
        Return param groups for optimizer so selector can have larger LR.
        Usage:
            optimizer = torch.optim.Adam(model.param_groups(base_lr=1e-4, selector_lr_scale=8.0))
        """
        selector_params = list(self.pattern_selector.parameters()) + [self.log_pattern_tau, self.gumbel_temp] if self.use_gumbel else list(self.pattern_selector.parameters()) + [self.log_pattern_tau]
        other_params = [p for n, p in self.named_parameters() if p not in selector_params]
        groups = [
            {"params": other_params, "lr": base_lr},
            {"params": selector_params, "lr": base_lr * selector_lr_scale},
        ]
        return groups

    def _sample_pattern_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 3)
        returns probabilities/proportions (B, 3)
        """
        # temperature (learnable) but clamp for stability
        tau = (self.log_pattern_tau.exp()).clamp(min=1e-4, max=10.0)

        if self.use_gumbel and self.training:
            # gumbel-softmax sampling for peaky selection (differentiable)
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            y = (logits + g) / (self.gumbel_temp.clamp(min=1e-4))
            probs = F.softmax(y / tau, dim=-1)
        else:
            probs = F.softmax(logits / tau, dim=-1)
        return probs, tau

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        x: (B, L, D)
        mask: (B, L) token mask with 1 for valid, 0 for padding
        returns: out (B, L, D), attention_info dict
        """
        B, L, D = x.shape
        device = x.device

        # QKV
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, head_dim)

        # attention scores (B, H, L, L)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # pooled features -> pattern logits
        pooled_features = torch.mean(x, dim=1)  # (B, D)
        pattern_logits = self.pattern_selector(pooled_features)  # (B, 3)

        pattern_weights, tau = self._sample_pattern_weights(pattern_logits)  # (B, 3), scalar tau

        # Debug printing and basic grad/param checks
        if self.debug:
            # activation stats
            logger.info("Pattern selector input mean/std: %s / %s", pooled_features.mean().item(), pooled_features.std().item())
            logger.info("Pattern logits sample: %s", pattern_logits.detach().cpu().numpy()[0])
            logger.info("Pattern weights sample: %s", pattern_weights.detach().cpu().numpy()[0])
            logger.info("Pattern tau (learned): %s", tau.item() if isinstance(tau, torch.Tensor) else float(tau))
            for n, p in self.pattern_selector.named_parameters():
                if p.grad is None:
                    logger.debug("Selector param %s requires_grad=%s", n, p.requires_grad)

        # Construct masks (float)
        local_mask = self.create_local_mask(L, device).unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        global_mask = self.create_global_mask(L, device).unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        sparse_mask = self.create_learned_sparse_mask(attention_scores)  # (B, H, L, L)

        # Expand pattern weights for broadcasting over heads and positions
        pw_local = pattern_weights[:, 0].view(B, 1, 1, 1)   # (B,1,1,1)
        pw_global = pattern_weights[:, 1].view(B, 1, 1, 1)
        pw_sparse = pattern_weights[:, 2].view(B, 1, 1, 1)

        # make local/global broadcast to (B, H, L, L)
        combined_mask = (
            pw_local * local_mask.expand(B, self.num_heads, L, L) +
            pw_global * global_mask.expand(B, self.num_heads, L, L) +
            pw_sparse * sparse_mask
        )

        # threshold to decide allowed connections; keep threshold small to avoid over-masking
        threshold = 0.05
        attention_mask = combined_mask > threshold  # boolean (B, H, L, L)

        # Mask attention scores; use -1e9 instead of -inf for numerical stability
        neg_inf = -1e9
        attention_scores = attention_scores.masked_fill(~attention_mask, neg_inf)

        # Apply padding mask (keys)
        if mask is not None:
            # mask: (B, L) -> (B, 1, 1, L)
            key_mask = mask.view(B, 1, 1, L)
            attention_scores = attention_scores.masked_fill(key_mask == 0, neg_inf)

        # Handle rows where everything became masked: allow first position
        # (B, H, L) boolean where all positions are neg_inf
        all_masked = (attention_scores <= neg_inf + 1e-6).all(dim=-1)
        if all_masked.any():
            # set first column to 0 (score for position 0) for those entries
            # attention_scores is shape (B, H, L, L)
            dummy_scores = attention_scores.clone()
            # set safe finite value
            dummy_scores[all_masked, 0] = 0.0
            attention_scores = dummy_scores

        # Softmax with temperature (for attention distribution)
        attn_tau = max(self.temperature, 1e-6)
        attention_weights = F.softmax(attention_scores / attn_tau, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        # Output
        out = torch.matmul(attention_weights, v)  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.proj(out)

        # Compute pattern regularizers (entropy, diversity)
        # pattern_weights: (B, 3)
        eps = 1e-8
        pattern_entropy = -(pattern_weights * (pattern_weights + eps).log()).sum(dim=-1)  # (B,)
        avg_entropy = pattern_entropy.mean()
        entropy_loss = avg_entropy  # if we want to penalize entropy, we add +lambda * entropy_loss
        diversity_loss = -torch.var(pattern_weights.mean(dim=0))  # negative var to encourage spread across patterns

        # pack attention_info
        attention_info = {
            "pattern_weights": pattern_weights.detach(),  # (B, 3)
            "pattern_logits": pattern_logits.detach(),
            "attention_weights": attention_weights.detach(),  # (B, H, L, L)
            "local_ratio": float(pattern_weights[:, 0].mean().item()),
            "global_ratio": float(pattern_weights[:, 1].mean().item()),
            "sparse_ratio": float(pattern_weights[:, 2].mean().item()),
            "pattern_entropy": float(avg_entropy.item()),
            "entropy_loss": entropy_loss,
            "diversity_loss": diversity_loss,
            "tau": float(tau.item()) if isinstance(tau, torch.Tensor) else float(tau),
        }

        # Note: we do NOT add the entropy/diversity terms into the forward return loss automatically,
        # instead the training loop should add them (so you control weighting and logging).
        # Example:
        #   loss = task_loss + model.entropy_lambda * attention_info['entropy_loss'] + model.diversity_lambda * attention_info['diversity_loss']

        return out, attention_info


class MultiHeadAdaptiveAttention(nn.Module):
    """Wrapper for compatibility (same API as before)."""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.attention = AdaptiveSparseAttention(dim=dim, num_heads=num_heads, dropout=dropout, **kwargs)

    def forward(self, x, mask=None):
        return self.attention(x, mask)
