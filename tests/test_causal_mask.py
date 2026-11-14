"""
Tests for causal attention masking.
"""

import pytest


def test_causal_mask_structure():
    """Verify causal mask is lower triangular with zeros on/below diagonal and -inf above."""
    # TODO: Verify that token at index t can attend to tokens <= t and cannot attend to tokens > t
    pytest.skip("TODO: implement after the module exists")


def test_attention_future_token_invariance():
    """Verify altering future token values does not affect past token attention outputs."""
    # TODO: Pass sequence X and modified sequence X' where token t+1 is changed; outputs at position t must match
    pytest.skip("TODO: implement after the module exists")


def test_attention_output_shape():
    """Verify multi-head causal attention output shape matches (B, T, C)."""
    # TODO: Check that input tensor (B, T, C) produces output tensor (B, T, C) with correct batch and sequence dimensions
    pytest.skip("TODO: implement after the module exists")
