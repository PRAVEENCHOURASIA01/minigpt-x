"""
Environment and hardware tests for MiniGPT-X.
Verifies PyTorch import, MPS backend availability, tensor movement to MPS,
and numerical equivalence between CPU and MPS matrix multiplication.
"""

import pytest
import torch


def test_torch_imports():
    """Verify PyTorch imports properly and has a valid version."""
    assert torch is not None
    assert hasattr(torch, "__version__")
    assert len(torch.__version__) > 0


def test_mps_available():
    """Verify Apple Silicon MPS backend is built and available."""
    assert torch.backends.mps.is_built(), "PyTorch was not built with MPS support."
    assert torch.backends.mps.is_available(), "Apple Silicon MPS backend is not available."


def test_tensor_to_mps():
    """Verify a tensor can be created and moved to the mps device."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    device = torch.device("mps")
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    assert x.device.type == "mps"
    assert x.shape == (3,)
    assert (x.cpu() == torch.tensor([1.0, 2.0, 3.0])).all()


def test_mps_matmul_matches_cpu():
    """Verify MPS matrix multiplication matches CPU result within float32 tolerance."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    torch.manual_seed(42)
    a_cpu = torch.randn(64, 64, dtype=torch.float32)
    b_cpu = torch.randn(64, 64, dtype=torch.float32)
    c_cpu = torch.matmul(a_cpu, b_cpu)

    a_mps = a_cpu.to("mps")
    b_mps = b_cpu.to("mps")
    c_mps = torch.matmul(a_mps, b_mps)
    torch.mps.synchronize()

    assert torch.allclose(c_mps.cpu(), c_cpu, atol=1e-4, rtol=1e-4)
