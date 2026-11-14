"""
Tests for Key-Value (KV) Cache and sampling functions during inference.
"""

import pytest
import torch
from inference.generate import sample_next_token
from inference.kv_cache import KVCache
from model.gpt import GPTConfig, MiniGPTX


def test_kv_cache_matches_uncached_generation():
    """Verify forward decoding with KV cache produces logits identical to full uncached re-forwarding."""
    torch.manual_seed(42)
    cfg = GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=64, vocab_size=500, use_rope=True, use_sdpa=True)
    model = MiniGPTX(cfg)
    model.eval()

    # Step 1: Prefill prompt tokens 0..5 with use_cache=True
    prompt = torch.randint(0, 500, (1, 6))
    with torch.no_grad():
        logits_prefill, _, kv_caches = model(prompt, offset=0, use_cache=True)

    # Next token to decode
    next_tok = torch.tensor([[42]], dtype=torch.long)

    # Step 2: Decode token with KV Cache
    with torch.no_grad():
        cached_logits, _, _ = model(next_tok, kv_caches=kv_caches, offset=6)

    # Step 3: Uncached full context forwarding
    full_prompt = torch.cat([prompt, next_tok], dim=1)  # shape (1, 7)
    with torch.no_grad():
        uncached_logits, _, _ = model(full_prompt)

    # Compare logits for the final newly decoded position
    diff = (cached_logits[0, -1, :] - uncached_logits[0, -1, :]).abs().max().item()
    assert diff < 1e-4, f"KV cache decoded logits differed from uncached re-forwarding: diff={diff}"


def test_sampling_greedy_matches_argmax():
    """Verify greedy temperature 0.0 always returns argmax."""
    logits = torch.tensor([1.2, 5.8, 2.1, 0.4])
    tok = sample_next_token(logits, temperature=0.0)
    assert tok == 1


def test_sampling_top_k_limits_choices():
    """Verify top-k=2 restricts selection strictly to the top 2 indices."""
    logits = torch.tensor([1.0, 10.0, 8.0, 0.5])
    # Top 2 are indices 1 (10.0) and 2 (8.0)
    for _ in range(30):
        tok = sample_next_token(logits, temperature=1.0, top_k=2, top_p=1.0)
        assert tok in (1, 2), f"Token {tok} chosen outside top-k"


def test_sampling_top_p_nucleus():
    """Verify top-p retains smallest subset exceeding probability mass."""
    # Extreme distribution: index 0 has ~90% probability
    logits = torch.tensor([10.0, 1.0, 0.5, 0.1])
    for _ in range(30):
        tok = sample_next_token(logits, temperature=1.0, top_k=50, top_p=0.8)
        assert tok == 0, f"Token {tok} selected when index 0 had dominant mass"


def test_kv_cache_reset():
    """Verify KVCache reset cleanly clears state."""
    cache = KVCache(n_layer=4)
    dummy = (torch.zeros(1, 2, 5, 16), torch.zeros(1, 2, 5, 16))
    cache.update_layer_cache(0, dummy)
    assert cache.current_seq_len() == 5
    cache.reset()
    assert cache.current_seq_len() == 0
    assert cache.get_layer_cache(0) is None
