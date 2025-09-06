import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .adaptive_attention import MultiHeadAdaptiveAttention

class AdaptiveTransformerBlock(nn.Module):
    """Transformer block with adaptive sparse attention."""
    
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
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, attn_info = self.attention(x_norm, mask)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_info

class AdaptiveSparseTransformer(nn.Module):
    """Complete Adaptive Sparse Transformer model."""
    
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
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdaptiveTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                **attention_kwargs
            )
            for _ in range(depth)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)

        self._fix_nan_parameters()

    def _fix_nan_parameters(self):
        """Fix any NaN parameters that appear during initialization."""
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"WARNING: NaN detected in {name} during initialization - reinitializing")
                if 'weight' in name:
                    nn.init.xavier_normal_(param, gain=0.01)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            
            # Also check for inf values
            if torch.isinf(param).any():
                print(f"WARNING: Inf detected in {name} during initialization - reinitializing")
                if 'weight' in name:
                    nn.init.xavier_normal_(param, gain=0.01)
                elif 'bias' in name:
                    nn.init.zeros_(param)

        
    def _init_weights(self, module):
        """Initialize weights conservatively to prevent NaN."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.02)  # Very small gain
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_info: bool = False
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = self.dropout(token_emb + pos_emb)
        
        # Track attention information
        all_attention_info = []
        
        # Pass through transformer blocks
        for block in self.blocks:
            x, attn_info = block(x, attention_mask)
            if return_attention_info:
                all_attention_info.append(attn_info)
        
        # Final normalization
        x = self.norm(x)
        
        # Classification (using [CLS] token or mean pooling)
        if attention_mask is not None:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            pooled = sum_embeddings / sum_mask
        else:
            # Simple mean pooling
            pooled = torch.mean(x, dim=1)
        
        logits = self.classifier(pooled)
        
        output = {
            'logits': logits,
            'last_hidden_state': x,
            'pooled_output': pooled
        }
        
        if return_attention_info:
            output['attention_info'] = all_attention_info
            
        return output