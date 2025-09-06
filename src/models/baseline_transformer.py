import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

class StandardMultiHeadAttention(nn.Module):
    """Standard multi-head attention for baseline comparison."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(1) == 0, float('-inf')
            )
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj(out)
        
        return out, {'attention_weights': attention_weights}

class BaselineTransformerBlock(nn.Module):
    """Standard transformer block for baseline."""
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attention = StandardMultiHeadAttention(dim, num_heads, dropout)
        
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
        attn_out, attn_info = self.attention(self.norm1(x), mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_info

class BaselineTransformer(nn.Module):
    """Standard transformer for performance comparison."""
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            BaselineTransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None, return_attention_info=False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.dropout(x)
        
        all_attention_info = []
        
        for block in self.blocks:
            x, attn_info = block(x, attention_mask)
            if return_attention_info:
                all_attention_info.append(attn_info)
        
        x = self.norm(x)
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
            pooled = torch.sum(x * mask_expanded, dim=1) / torch.sum(mask_expanded, dim=1)
        else:
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
