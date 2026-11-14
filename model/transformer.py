"""
Transformer Block and Feed-Forward Networks (SwiGLU and standard ReLU MLP).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import CausalSelfAttention
from model.normalization import get_norm


class SwiGLUMLP(nn.Module):
    """
    SwiGLU Feed-Forward Network:
        hidden_dim ~ 8/3 * n_embd, rounded to multiple of 64.
        out = W_down(silu(W_gate(x)) * W_up(x))
        
    Tensor Shapes:
        Input x:      (B, T, C)
        Gate & Up:    (B, T, hidden_dim)
        Down proj:    (B, T, C)
    """
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        # 8/3 * n_embd rounded up to multiple of 64
        raw_hidden = int(2.0 * (4.0 * n_embd) / 3.0)
        hidden_dim = ((raw_hidden + 63) // 64) * 64
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_up = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu(w_gate(x)) * w_up(x)
        h = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.dropout(self.w_down(h))


class ReLUMLP(nn.Module):
    """
    Standard ReLU MLP Feed-Forward Network:
        hidden_dim = 4 * n_embd.
        out = W_down(relu(W_up(x)))
    """
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        hidden_dim = 4 * n_embd
        self.c_fc = nn.Linear(n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.c_fc(x))
        return self.dropout(self.c_proj(h))


class TransformerBlock(nn.Module):
    """
    Decoder-only Transformer Block with Pre-Normalization and Residual Connections:
        x = x + Attention(Norm_1(x))
        x = x + MLP(Norm_2(x))
        
    Tensor Shapes:
        Input x:  (B, T, C)
        Output:   (B, T, C)
    """
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        activation: str = "swiglu",
        use_rope: bool = True,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.ln_1 = get_norm(norm_type, n_embd)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
            use_rope=use_rope,
            use_sdpa=use_sdpa,
        )
        self.ln_2 = get_norm(norm_type, n_embd)
        if activation.lower() == "swiglu":
            self.mlp = SwiGLUMLP(n_embd=n_embd, dropout=dropout)
        elif activation.lower() == "relu":
            self.mlp = ReLUMLP(n_embd=n_embd, dropout=dropout)
        else:
            raise ValueError(f"Unknown activation: {activation}. Must be 'swiglu' or 'relu'.")

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm architecture
        normed_1 = self.ln_1(x)
        attn_out, new_kv_cache = self.attn(
            normed_1, cos=cos, sin=sin, kv_cache=kv_cache, offset=offset
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache
