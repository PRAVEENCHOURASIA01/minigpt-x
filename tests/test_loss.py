"""
Tests for cross-entropy loss computation and parameter behavior.
"""

import math
import pytest
import torch
from model.gpt import GPTConfig, MiniGPTX


def test_initial_loss_near_uniform_entropy():
    """Verify untrained model loss is close to theoretical uniform cross-entropy: ln(vocab_size)."""
    vocab_size = 4096
    cfg = GPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=64,
        vocab_size=vocab_size,
    )
    torch.manual_seed(42)
    model = MiniGPTX(cfg)
    model.eval()

    x = torch.randint(0, vocab_size, (4, 32))
    targets = torch.randint(0, vocab_size, (4, 32))

    with torch.no_grad():
        _, loss, _ = model(x, targets=targets)

    expected_loss = math.log(vocab_size)  # ln(4096) ≈ 8.3177
    actual_loss = loss.item()
    diff = abs(actual_loss - expected_loss)
    # At random initialization, loss should be within 0.5 of ln(vocab_size)
    assert diff < 0.5, f"Initial loss {actual_loss:.4f} too far from ln({vocab_size}) = {expected_loss:.4f}"


def test_loss_target_ignore_index():
    """Verify loss calculation properly ignores masked padding index (-1)."""
    vocab_size = 100
    cfg = GPTConfig(n_layer=1, n_head=2, n_embd=64, block_size=32, vocab_size=vocab_size)
    model = MiniGPTX(cfg)

    x = torch.randint(0, vocab_size, (2, 10))
    targets = torch.full((2, 10), -1, dtype=torch.long)
    targets[:, :5] = torch.randint(0, vocab_size, (2, 5))

    _, loss, _ = model(x, targets=targets)
    assert not torch.isnan(loss), "Loss was NaN with ignore_index=-1"
    loss.backward()
    assert model.tok_embeddings.weight.grad is not None


def test_weight_tying_and_optimizer_groups():
    """Verify tied weights and 2D parameter splitting for optimizer."""
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=200, tie_embeddings=True)
    model = MiniGPTX(cfg)
    assert model.lm_head.weight is model.tok_embeddings.weight

    opt = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu")
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 0.1
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_model_mps_forward_backward():
    """Verify complete forward and backward pass on Apple Silicon MPS backend."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    device = torch.device("mps")
    cfg = GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=64, vocab_size=500)
    model = MiniGPTX(cfg).to(device)
    model.train()

    x = torch.randint(0, 500, (2, 16), device=device)
    y = torch.randint(0, 500, (2, 16), device=device)

    logits, loss, _ = model(x, targets=y)
    assert logits.device.type == "mps"
    assert loss.device.type == "mps"

    loss.backward()
    torch.mps.synchronize()
    assert model.blocks[0].attn.c_attn.weight.grad is not None
