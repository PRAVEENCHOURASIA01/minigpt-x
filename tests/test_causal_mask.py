"""
Tests for causal attention masking.
"""

import pytest
import torch
from model.gpt import GPTConfig, MiniGPTX


def test_causal_mask_structure():
    """Verify modifying future tokens has zero effect on earlier logits."""
    torch.manual_seed(42)
    cfg = GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=64, vocab_size=500, use_sdpa=False)
    model = MiniGPTX(cfg)
    model.eval()

    # Create sequence of length 16
    x1 = torch.randint(0, 500, (1, 16))
    x2 = x1.clone()
    # Modify future tokens (at index 10 and beyond)
    x2[0, 10:] = torch.randint(0, 500, (6,))

    with torch.no_grad():
        logits1, _, _ = model(x1)
        logits2, _, _ = model(x2)

    # Logits at positions 0..9 must match exactly
    diff = (logits1[:, :10, :] - logits2[:, :10, :]).abs().max().item()
    assert diff < 1e-5, f"Future token modification affected past logits: diff={diff}"


def test_attention_future_token_invariance_sdpa():
    """Verify future token invariance also holds under PyTorch SDPA."""
    torch.manual_seed(42)
    cfg = GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=64, vocab_size=500, use_sdpa=True)
    model = MiniGPTX(cfg)
    model.eval()

    x1 = torch.randint(0, 500, (1, 16))
    x2 = x1.clone()
    x2[0, 10:] = torch.randint(0, 500, (6,))

    with torch.no_grad():
        logits1, _, _ = model(x1)
        logits2, _, _ = model(x2)

    diff = (logits1[:, :10, :] - logits2[:, :10, :]).abs().max().item()
    assert diff < 1e-4, f"SDPA future tokens leaked into past: diff={diff}"


def test_sdpa_equals_manual_attention():
    """Verify SDPA output matches manual causal attention output within numerical tolerance."""
    torch.manual_seed(1337)
    cfg_manual = GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=32, vocab_size=200, use_sdpa=False)
    cfg_sdpa = GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=32, vocab_size=200, use_sdpa=True)

    m_manual = MiniGPTX(cfg_manual)
    m_sdpa = MiniGPTX(cfg_sdpa)
    # Copy exact weights
    m_sdpa.load_state_dict(m_manual.state_dict())

    m_manual.eval()
    m_sdpa.eval()

    x = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        out_manual, _, _ = m_manual(x)
        out_sdpa, _, _ = m_sdpa(x)

    diff = (out_manual - out_sdpa).abs().max().item()
    assert diff < 1e-4, f"SDPA output differed from manual attention: diff={diff}"
