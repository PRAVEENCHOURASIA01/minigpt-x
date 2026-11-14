"""
Learning Rate Scheduler (Linear Warmup + Cosine Decay).

This module implements a cosine decay learning rate schedule with warmup:
1. Linear warmup from 0.0 to learning_rate over warmup_iters
2. Cosine decay from learning_rate down to min_lr between warmup_iters and max_iters
3. Constant min_lr for any iterations beyond max_iters
"""

# TODO: Define get_lr(it: int, learning_rate: float, min_lr: float, warmup_iters: int, max_iters: int) -> float
# TODO: Implement linear warmup phase logic: lr = learning_rate * it / warmup_iters
# TODO: Implement cosine decay phase logic using math.cos
# TODO: Implement floor at min_lr for iterations exceeding max_iters
# TODO: Wrap in a PyTorch LambdaLR or custom scheduler step function
