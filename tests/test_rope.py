"""
Tests for Rotary Position Embeddings (RoPE).
"""

import pytest


def test_rope_preserves_vector_norm():
    """Verify RoPE rotation operation preserves the L2 norm of query and key vectors."""
    # TODO: Check that ||RoPE(x)||_2 == ||x||_2 within floating point precision, since rotation is an orthogonal transformation
    pytest.skip("TODO: implement after the module exists")


def test_rope_relative_shift_invariance():
    """Verify RoPE inner product depends only on relative distance (m - n), not absolute positions."""
    # TODO: Verify <RoPE(q, pos=m), RoPE(k, pos=n)> == <RoPE(q, pos=m+k), RoPE(k, pos=n+k)> for any integer offset k
    pytest.skip("TODO: implement after the module exists")


def test_rope_tensor_shape_preservation():
    """Verify applying RoPE preserves the original tensor dimensions (B, n_head, T, head_dim)."""
    # TODO: Ensure output shape after rotary embedding is identical to input query and key tensor shape
    pytest.skip("TODO: implement after the module exists")
