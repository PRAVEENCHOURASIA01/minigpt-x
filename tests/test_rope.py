"""
Tests for Rotary Position Embeddings (RoPE).
"""

import math
import pytest
import torch
from model.embeddings import precompute_rope_freqs, apply_rotary_emb


def test_rope_preserves_vector_norm():
    """Verify RoPE rotation operation preserves the L2 norm of vectors."""
    head_dim = 64
    seq_len = 32
    cos, sin = precompute_rope_freqs(head_dim, seq_len)
    
    torch.manual_seed(42)
    # Shape: (B=2, n_head=4, T=32, head_dim=64)
    x = torch.randn(2, 4, seq_len, head_dim)
    orig_norm = torch.norm(x, p=2, dim=-1)

    rotated = apply_rotary_emb(x, cos, sin)
    rot_norm = torch.norm(rotated, p=2, dim=-1)

    diff = (orig_norm - rot_norm).abs().max().item()
    assert diff < 1e-5, f"RoPE failed to preserve vector norm: diff={diff}"


def test_rope_relative_shift_invariance():
    """Verify RoPE inner product depends only on relative distance (m - n)."""
    head_dim = 32
    max_len = 64
    cos, sin = precompute_rope_freqs(head_dim, max_len)

    torch.manual_seed(123)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    # Compute inner product at positions (m=5, n=2), diff = 3
    q_m = apply_rotary_emb(q, cos, sin, offset=5)
    k_n = apply_rotary_emb(k, cos, sin, offset=2)
    score1 = (q_m * k_n).sum().item()

    # Compute inner product shifted by offset delta=10 -> positions (m'=15, n'=12), diff = 3
    q_m_shifted = apply_rotary_emb(q, cos, sin, offset=15)
    k_n_shifted = apply_rotary_emb(k, cos, sin, offset=12)
    score2 = (q_m_shifted * k_n_shifted).sum().item()

    assert math.isclose(score1, score2, rel_tol=1e-4, abs_tol=1e-4), (
        f"RoPE relative shift invariance violated: {score1} vs {score2}"
    )


def test_rope_tensor_shape_preservation():
    """Verify applying RoPE preserves tensor shape."""
    head_dim = 64
    cos, sin = precompute_rope_freqs(head_dim, 128)
    x = torch.randn(3, 8, 50, head_dim)
    out = apply_rotary_emb(x, cos, sin)
    assert out.shape == x.shape
