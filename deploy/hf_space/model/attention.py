"""
Multi-Head Causal Self-Attention with RoPE, SDPA, and KV Cache Support.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embeddings import apply_rotary_emb


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    
    Tensor Shapes across forward pass:
        Input x:           (B, T, C) where C = n_embd
        Linear Q, K, V:    (B, T, 3 * C)
        Split into heads:  (B, n_head, T, head_dim) where head_dim = C // n_head
        RoPE Q, K:         (B, n_head, T, head_dim)
        Attention output:  (B, n_head, T, head_dim)
        Recombined:        (B, T, C)
        Output projection: (B, T, C)
    """
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_sdpa: bool = True,
    ):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.dropout = dropout
        self.use_rope = use_rope
        self.use_sdpa = use_sdpa

        # Fused linear projection for Query, Key, and Value
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Explicit lower-triangular causal mask buffer for manual attention path
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(
                1, 1, block_size, block_size
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape (B, T, C)
            cos, sin: Precomputed RoPE frequency tensors (max_seq_len, head_dim // 2)
            kv_cache: Optional past key, value tensors (B, n_head, past_T, head_dim)
            offset: Current token position offset in the sequence (for KV cache decoding)
            use_cache: If True, returns updated (key, value) tensors even on initial prefill
            
        Returns:
            out: (B, T, C)
            new_kv_cache: (key, value) tensors if kv_cache is provided or use_cache is True
        """
        B, T, C = x.shape

        # 1. Project to Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape into heads: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE if enabled
        if self.use_rope and cos is not None and sin is not None:
            q = apply_rotary_emb(q, cos, sin, offset=offset)
            k = apply_rotary_emb(k, cos, sin, offset=offset)

        # 4. Handle KV Cache
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
            new_kv_cache = (k, v)
        elif use_cache:
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        total_kv_len = k.shape[2]

        # 5. Attention mechanism
        if self.use_sdpa:
            is_causal = (T > 1) and (total_kv_len == T)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Manual attention path
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if T > 1 and total_kv_len == T:
                mask = self.bias[:, :, :T, :T]
                att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # 6. Recombine heads: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 7. Final output projection with residual dropout
        out = self.resid_dropout(self.c_proj(y))
        return out, new_kv_cache
