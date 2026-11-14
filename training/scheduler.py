"""
Learning Rate Scheduler with Linear Warmup and Cosine Decay.
"""

import math


def get_lr(
    it: int,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    max_iters: int,
) -> float:
    """
    Computes learning rate for iteration `it` using linear warmup and cosine decay.
    
    1. Linear warmup: 0 -> learning_rate over warmup_iters steps.
    2. Cosine decay: learning_rate -> min_lr between warmup_iters and max_iters.
    3. Beyond max_iters: keep constant at min_lr.
    """
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it > max_iters:
        return min_lr
    # In between, use cosine decay
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
