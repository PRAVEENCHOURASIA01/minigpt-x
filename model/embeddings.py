"""
Token and Positional Embeddings for MiniGPT-X.

Includes:
- Standard Token Embedding lookup.
- Precomputed Rotary Position Embeddings (RoPE) and complex rotation application.
- Fallback learned absolute positional embeddings for ablation studies.
"""

from typing import Tuple
import torch
import torch.nn as nn


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes cosine and sine frequency tensors for Rotary Position Embeddings (RoPE).
    
    Args:
        head_dim: Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length (block_size).
        theta: Frequency base scaling factor (default: 10000.0).
        
    Returns:
        cos: Tensor of shape (max_seq_len, head_dim // 2)
        sin: Tensor of shape (max_seq_len, head_dim // 2)
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    # dim_indices: 0, 2, 4, ..., head_dim - 2
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
    # freqs: theta^(-2i / head_dim)
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))
    # pos: 0, 1, ..., max_seq_len - 1
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # outer product: (max_seq_len, head_dim // 2)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """
    Applies Rotary Position Embeddings (RoPE) to an input query or key tensor.
    
    Tensor Shapes:
        x: (B, n_head, T, head_dim)
        cos: (max_seq_len, head_dim // 2) or sliced to (T, head_dim // 2)
        sin: (max_seq_len, head_dim // 2) or sliced to (T, head_dim // 2)
        
    Returns:
        Rotated tensor x of shape (B, n_head, T, head_dim) with preserved L2 norm.
    """
    B, n_head, T, head_dim = x.shape
    half_dim = head_dim // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    # Slice cos and sin for current sequence positions [offset : offset + T]
    # Expand shapes for broadcasting: (1, 1, T, half_dim)
    c = cos[offset : offset + T, :].unsqueeze(0).unsqueeze(0).to(x.device, dtype=x.dtype)
    s = sin[offset : offset + T, :].unsqueeze(0).unsqueeze(0).to(x.device, dtype=x.dtype)

    # Complex multiplication equivalent:
    # (x1 + i*x2) * (cos + i*sin) = (x1*cos - x2*sin) + i*(x1*sin + x2*cos)
    x_rot1 = x1 * c - x2 * s
    x_rot2 = x1 * s + x2 * c
    return torch.cat([x_rot1, x_rot2], dim=-1)
