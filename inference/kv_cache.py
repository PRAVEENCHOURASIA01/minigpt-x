"""
Key-Value (KV) Cache management for MiniGPT-X inference.
"""

from typing import List, Optional, Tuple
import torch


class KVCache:
    """
    Manages Key-Value tensor caches across all Transformer decoder layers.
    Eliminates redundant O(N^2) past key/value recomputations during autoregressive decoding.
    """
    def __init__(self, n_layer: int):
        self.n_layer = n_layer
        self.caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layer

    def get_layer_cache(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.caches[layer_idx]

    def update_layer_cache(self, layer_idx: int, cache: Tuple[torch.Tensor, torch.Tensor]):
        self.caches[layer_idx] = cache

    def reset(self):
        """Clears all stored key and value tensors."""
        self.caches = [None] * self.n_layer

    def current_seq_len(self) -> int:
        """Returns sequence length of the currently stored keys."""
        if self.caches[0] is None:
            return 0
        return self.caches[0][0].shape[2]
