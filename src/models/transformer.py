# src/models/transformer.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple

from .adaptive_attention import MultiHeadAdaptiveAttention


class AdaptiveTransformerBlock(nn.Module):
    """Transformer block with adaptive sparse attention (pre-norm)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        **attention_kwargs
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attention = MultiHeadAdaptiveAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            **attention_kwargs
        )

        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.FloatTensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.FloatTensor, dict]:
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out, attn_info = self.attention(x_norm, mask)
        # Apply dropout to attention output and residual
        x = x + self.dropout(attn_out)

        # Feed-forward (pre-norm on the residual input)
        x = x + self.mlp(self.norm2(x))

        return x, attn_info


class AdaptiveSparseTransformer(nn.Module):
    """Full Adaptive Sparse Transformer."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 2,
        **attention_kwargs
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                AdaptiveTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    **attention_kwargs,
                )
                for _ in range(depth)
            ]
        )

        # Output head
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

        # Initialize weights and guard against NaN/infs
        self.apply(self._init_weights)
        self._reinit_bad_parameters()

    def _reinit_bad_parameters(self) -> None:
        """Reinitialize any params that end up NaN/Inf after initialization (defensive)."""
        for name, p in self.named_parameters():
            if p is None:
                continue
            if torch.isnan(p).any() or torch.isinf(p).any():
                # Use a conservative re-init
                print(f"WARNING: {name} had NaN/Inf after init; reinitializing.")
                if p.dim() >= 2:
                    nn.init.xavier_normal_(p, gain=0.02)
                else:
                    nn.init.zeros_(p)

    def _init_weights(self, module: nn.Module) -> None:
        """Conservative initialization for numeric stability."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _check_seq_len(self, seq_len: int) -> None:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds model max_seq_len ({self.max_seq_len}). "
                "Either increase max_seq_len in config or truncate inputs."
            )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_info: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: (B, L) long tensor of token ids
            attention_mask: (B, L) binary mask with 1 for valid tokens, 0 for pad
            return_attention_info: whether to return attention diagnostics
        Returns:
            dict with 'logits', 'last_hidden_state', 'pooled_output' and optionally 'attention_info'
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Sanity check: seq len within positional embedding range
        self._check_seq_len(seq_len)

        # Position ids
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_emb = self.token_embedding(input_ids)  # (B, L, D)
        pos_emb = self.position_embedding(position_ids)  # (B, L, D)
        x = self.dropout(token_emb + pos_emb)

        # Collect attention info if requested (list per-layer)
        all_attention_info: List[dict] = []

        # Pass through layers
        for block in self.blocks:
            x, attn_info = block(x, attention_mask)
            if return_attention_info:
                all_attention_info.append(attn_info)

        # Final normalization
        x = self.norm(x)  # (B, L, D)

        # Masked pooling with robust fallback:
        # If attention_mask present, compute sum_embeddings / sum_mask; if a sample has sum_mask==0,
        # fallback to simple mean pooling across tokens for that sample (avoids division explosion).
        if attention_mask is not None:
            # ensure mask dtype and device
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x).to(dtype=x.dtype, device=device)  # (B, L, D)
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)  # (B, D)
            sum_mask = torch.sum(mask_expanded, dim=1)  # (B, D) (same value across D)
            # per-sample scalar mask counts (shape B,)
            sum_mask_counts = sum_mask[:, 0]  # (B,)
            zero_mask = (sum_mask_counts == 0)

            # Avoid divide-by-zero by using a clamped denominator for safe division,
            # then replace zero-mask samples with mean pooling fallback.
            sum_mask_safe = sum_mask_counts.clamp(min=1.0).unsqueeze(-1)  # (B,1)
            pooled = sum_embeddings / sum_mask_safe  # (B, D)

            if zero_mask.any():
                # fallback mean pooling for zero-mask samples
                mean_pool = torch.mean(x, dim=1)  # (B, D)
                pooled[zero_mask] = mean_pool[zero_mask]
        else:
            pooled = torch.mean(x, dim=1)

        logits = self.classifier(pooled)

        output: Dict[str, Any] = {
            "logits": logits,
            "last_hidden_state": x,
            "pooled_output": pooled,
        }

        if return_attention_info:
            output["attention_info"] = all_attention_info

        return output

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
