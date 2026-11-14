"""
Tests for Root Mean Square Normalization (RMSNorm).
"""

import math
import pytest
import torch
from model.normalization import RMSNorm


def test_rmsnorm_output_rms_scale():
    """Verify normalized tensor output has an RMS approximately equal to 1.0 (with unit weights)."""
    dim = 128
    norm = RMSNorm(dim)
    torch.manual_seed(42)
    x = torch.randn(4, 32, dim) * 5.0 + 3.0  # arbitrary scale and offset
    y = norm(x)

    rms = torch.sqrt(torch.mean(y.pow(2), dim=-1))
    diff = (rms - 1.0).abs().max().item()
    assert diff < 1e-3, f"RMSNorm output RMS was not close to 1.0: diff={diff}"


def test_rmsnorm_scale_invariance():
    """Verify multiplying inputs by an arbitrary positive scalar c does not change RMSNorm output."""
    dim = 64
    norm = RMSNorm(dim)
    torch.manual_seed(42)
    x = torch.randn(2, 10, dim)
    c = 7.5
    y1 = norm(x)
    y2 = norm(c * x)

    diff = (y1 - y2).abs().max().item()
    assert diff < 1e-4, f"RMSNorm violated scale invariance: diff={diff}"


def test_rmsnorm_tensor_shape_preservation():
    """Verify input shape matches output shape through RMSNorm."""
    norm = RMSNorm(256)
    x = torch.randn(2, 5, 256)
    y = norm(x)
    assert y.shape == x.shape
