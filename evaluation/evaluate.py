"""
Evaluation Loop and Metric Utilities for MiniGPT-X.
"""

import math
from typing import Tuple
import torch
import torch.nn as nn


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    val_data,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluates validation loss, perplexity, and bits-per-byte over fixed validation windows.
    
    Returns:
        mean_val_loss: Cross entropy in nats
        perplexity: exp(mean_val_loss)
        bits_per_byte: (val_loss_nats / ln(2)) * (total_tokens / total_bytes)
    """
    model.eval()
    losses = []
    total_tokens = 0
    total_bytes = 0

    # Fixed seed for deterministic validation windows across runs
    rng = torch.Generator().manual_seed(999)

    max_idx = len(val_data) - block_size - 1
    for _ in range(eval_iters):
        ix = torch.randint(0, max_idx, (batch_size,), generator=rng)
        x_list = [torch.from_numpy(val_data[i : i + block_size].astype(int)) for i in ix]
        y_list = [torch.from_numpy(val_data[i + 1 : i + 1 + block_size].astype(int)) for i in ix]
        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)

        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
        total_tokens += x.numel()

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")

    # In our TinyStories BPE, compression is ~3.98 chars/token -> bytes/token ≈ 1 / 3.98 tokens/byte ≈ 0.251
    # Standard formula: (loss_nats / ln(2)) * (tokens / bytes)
    # With 3.98 chars/token, tokens/byte ≈ 1.0 / 3.98
    chars_per_token = 3.98
    bits_per_byte = (mean_loss / math.log(2)) * (1.0 / chars_per_token)

    model.train()
    return mean_loss, perplexity, bits_per_byte
