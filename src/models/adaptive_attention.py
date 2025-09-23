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

    Improvements included:
      - pattern_bias registered in __init__ (no CPU/CUDA mismatch)
      - learnable pattern temperature (log_pattern_tau)
      - optional gumbel-softmax path (use_gumbel)
      - entropy/diversity regularizers (weights exposed)
      - helper param_groups() to give selector a larger LR easily
      - robust handling of masked rows and numerical stability
      - debug flag for printing activation / gradient info
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        local_window_size: int = 32,
        learnable_sparsity: bool = True,
        temperature: float = 1.0,
        pattern_temperature: float = 0.5,
        use_gumbel: bool = False,
        gumbel_init_temp: float = 1.0,
        entropy_lambda: float = 0.0,
        diversity_lambda: float = 0.0,
        debug: bool = False,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        # Core config
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout
        self.local_window_size = local_window_size
        self.temperature = temperature

        # Pattern temperature (learnable in log-space for stability)
        self.log_pattern_tau = nn.Parameter(torch.tensor(math.log(max(pattern_temperature, 1e-6)), dtype=torch.float32))
        # Gumbel option
        self.use_gumbel = use_gumbel
        self.gumbel_temp = nn.Parameter(torch.tensor(gumbel_init_temp, dtype=torch.float32))
        # Regularization coefficients (exposed so trainer can read/change)
        self.entropy_lambda = entropy_lambda
        self.diversity_lambda = diversity_lambda

        self.debug = debug

        # QKV and projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout_layer = nn.Dropout(self.dropout_p)

        # Pattern selector: sequence-level MLP -> logits for [local, global, sparse]
        hidden = max(64, dim // 2)
        self.pattern_selector = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(max(32, hidden // 2), 3),  # outputs raw logits
        )

        # Learnable bias for pattern logits (registered here, avoids device mismatch)
        # small non-zero init to break symmetry
        self.pattern_bias = nn.Parameter(torch.tensor([0.2, -0.1, -0.1], dtype=torch.float32))

        # Per-head learnable sparsity parameters (broadcastable)
        self.learnable_sparsity = learnable_sparsity
        if learnable_sparsity:
            # shape (H, 1, 1), scaled small
            self.sparse_pattern_weights = nn.Parameter(torch.randn(num_heads, 1, 1) * 0.2)
            self.sparse_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

        # Initialization
        self._init_weights()

    def _init_weights(self):
        # Pattern selector init: slightly larger gain on final layer to encourage exploration
        for i, module in enumerate(self.pattern_selector):
            if isinstance(module, nn.Linear):
                gain = 1.0
                # last linear -> higher gain
                if i == len(self.pattern_selector) - 1:
                    gain = 1.6
                    nn.init.xavier_normal_(module.weight, gain=gain)
                    if module.bias is not None:
                        # small bias already set by self.pattern_bias, keep zeros here
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_normal_(module.weight, gain=gain)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # QKV and proj
        nn.init.xavier_normal_(self.qkv.weight, gain=0.5)
        nn.init.xavier_normal_(self.proj.weight, gain=0.5)

        # Sparse params
        if self.learnable_sparsity:
            nn.init.normal_(self.sparse_pattern_weights, mean=0.0, std=0.2)
            nn.init.zeros_(self.sparse_bias)

    @torch.no_grad()
    def create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Binary local mask (L, L) with ones where local attention allowed."""
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
        half = self.local_window_size // 2
        # vectorized ranges for each row (simple loop is fine for moderate seq_len)
        for i in range(seq_len):
            start = max(0, i - half)
            end = min(seq_len, i + half + 1)
            mask[i, start:end] = 1.0
        return mask

    @torch.no_grad()
    def create_global_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.ones((seq_len, seq_len), device=device, dtype=torch.float32)

    def create_learned_sparse_mask(self, attention_scores: torch.Tensor, sparsity_ratio: float = 0.3) -> torch.Tensor:
        """
        attention_scores: (B, H, L, L)
        returns binary mask (B, H, L, L) float32 with 1 where kept.
        """
        B, H, L, _ = attention_scores.shape
        k = max(1, int(L * (1 - sparsity_ratio)))

        # apply per-head transformation if enabled
        if self.learnable_sparsity:
            w = self.sparse_pattern_weights.view(1, H, 1, 1)
            b = self.sparse_bias.view(1, H, 1, 1)
            scores = attention_scores * w + b
        else:
            scores = attention_scores

        # jitter to break ties
        jitter = torch.randn_like(scores) * 1e-6
        scores_jittered = scores + jitter

        # topk along last dim
        topk_vals, topk_idx = torch.topk(scores_jittered, k=min(k, L), dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.float32)
        mask.scatter_(-1, topk_idx, 1.0)
        return mask

    def param_groups(self, base_lr: float = 1e-4, selector_lr_scale: float = 8.0):
        """
        Return optimizer param groups so the selector and its temps can have larger LR.
        Usage:
            optimizer = torch.optim.AdamW(model.param_groups(base_lr=1e-4), ...)
        """
        selector_names = {"pattern_selector", "pattern_bias", "log_pattern_tau", "gumbel_temp"}
        selector_params = []
        other_params = []
        for name, p in self.named_parameters():
            if any(name.startswith(sn) for sn in selector_names):
                selector_params.append(p)
            else:
                other_params.append(p)
        groups = [
            {"params": other_params, "lr": base_lr},
            {"params": selector_params, "lr": base_lr * selector_lr_scale},
        ]
        return groups

    def _sample_pattern_weights(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        logits: (B, 3)
        returns: (probs (B,3), tau scalar Tensor)
        """
        tau = (self.log_pattern_tau.exp()).clamp(min=1e-4, max=10.0)
        if self.use_gumbel and self.training:
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            y = (logits + g) / (self.gumbel_temp.clamp(min=1e-4))
            probs = F.softmax(y / tau, dim=-1)
        else:
            probs = F.softmax(logits / tau, dim=-1)
        return probs, tau

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        x: (B, L, D)
        mask: (B, L)
        returns: out (B, L, D), attention_info dict
        """
        B, L, D = x.shape
        device = x.device

        # QKV
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention scores (B, H, L, L)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # pooled features (sequence-level)
        pooled = torch.mean(x, dim=1)  # (B, D)

        # safe: pattern_selector expects input dim; if changed, user should adapt selector architecture
        pattern_logits = self.pattern_selector(pooled)  # (B, 3)

        # add registered bias (already a Parameter on correct device)
        pattern_logits = pattern_logits + self.pattern_bias.unsqueeze(0)

        # sample / compute pattern weights using learnable temperature
        pattern_weights, tau = self._sample_pattern_weights(pattern_logits)  # (B, 3), scalar

        # debugging prints and gradient hook option
        if self.debug and self.training and torch.rand(1).item() < 0.05:
            logger.info("Pattern logits sample: %s", pattern_logits.detach().cpu().numpy()[0])
            logger.info("Pattern weights sample: %s", pattern_weights.detach().cpu().numpy()[0])
            logger.info("Pattern logits std: %.6f", pattern_logits.std().item())
            # hook grads on logits (diagnostic); hook returns None for non-grad
            if pattern_logits.requires_grad:
                def _hook(g):
                    if g is None:
                        logger.warning("Pattern logits grad is None")
                    else:
                        logger.info("Pattern logits grad norm: %.6e", g.norm().item())
                pattern_logits.register_hook(_hook)

        # construct masks
        local_mask = self.create_local_mask(L, device).unsqueeze(0).unsqueeze(0)   # (1,1,L,L)
        global_mask = self.create_global_mask(L, device).unsqueeze(0).unsqueeze(0) # (1,1,L,L)
        sparse_mask = self.create_learned_sparse_mask(attention_scores)           # (B,H,L,L)

        # expand pattern weights
        pw_local = pattern_weights[:, 0].view(B, 1, 1, 1)
        pw_global = pattern_weights[:, 1].view(B, 1, 1, 1)
        pw_sparse = pattern_weights[:, 2].view(B, 1, 1, 1)

        combined_mask = (
            pw_local * local_mask.expand(B, self.num_heads, L, L) +
            pw_global * global_mask.expand(B, self.num_heads, L, L) +
            pw_sparse * sparse_mask
        )  # (B, H, L, L) float

        # threshold to decide allowed connections; small threshold to avoid over-masking
        threshold = 0.05
        allowed = combined_mask > threshold  # boolean mask

        # mask attention scores (use large negative number for numerical stability)
        neg_inf = -1e9
        attention_scores = attention_scores.masked_fill(~allowed, neg_inf)

        # apply padding mask (key mask)
        if mask is not None:
            key_mask = mask.view(B, 1, 1, L)
            attention_scores = attention_scores.masked_fill(key_mask == 0, neg_inf)

        # fix fully masked rows (all -inf) by setting index 0 to 0.0 (safe finite)
        all_masked = (attention_scores <= neg_inf + 1e-6).all(dim=-1)  # (B,H,L)
        if all_masked.any():
            attention_scores = attention_scores.clone()
            # set score of position 0 to 0.0 for masked entries
            attention_scores[all_masked, 0] = 0.0

        # softmax to get attention weights (with attention temperature)
        attn_tau = max(self.temperature, 1e-6)
        attention_weights = F.softmax(attention_scores / attn_tau, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        # output
        out = torch.matmul(attention_weights, v)  # (B,H,L,head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # (B,L,D)
        out = self.proj(out)

        # pattern regularizers: entropy & diversity (computed but not added automatically)
        eps = 1e-8
        pattern_entropy = -(pattern_weights * (pattern_weights + eps).log()).sum(dim=-1)  # (B,)
        avg_entropy = pattern_entropy.mean()
        entropy_loss = avg_entropy  # trainer may multiply by entropy_lambda

        diversity_loss = -torch.var(pattern_weights.mean(dim=0))  # negative var to encourage spread

        attention_info = {
            "pattern_weights": pattern_weights.detach(),
            "pattern_logits": pattern_logits.detach(),
            "attention_weights": attention_weights.detach(),
            "local_ratio": float(pattern_weights[:, 0].mean().item()),
            "global_ratio": float(pattern_weights[:, 1].mean().item()),
            "sparse_ratio": float(pattern_weights[:, 2].mean().item()),
            "pattern_entropy": float(avg_entropy.item()),
            "entropy_loss": entropy_loss,
            "diversity_loss": diversity_loss,
            "tau": float(tau.item()) if isinstance(tau, torch.Tensor) else float(tau),
        }

        return out, attention_info


class MultiHeadAdaptiveAttention(nn.Module):
    """Wrapper for backwards compatibility / easy swapping."""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.attention = AdaptiveSparseAttention(dim=dim, num_heads=num_heads, dropout=dropout, **kwargs)

    def forward(self, x, mask=None):
        return self.attention(x, mask)
