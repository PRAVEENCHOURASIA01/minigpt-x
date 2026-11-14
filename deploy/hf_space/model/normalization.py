"""
Normalization layers for MiniGPT-X: RMSNorm and LayerNorm.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    Normalizes inputs by the root-mean-square of activations across the last dimension,
    followed by a learnable scaling parameter (gamma / weight).
    
    Tensor Shapes:
        Input x:  (B, T, C)
        Output:   (B, T, C)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rms: sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        # Compute in float32 for numerical stability, then cast back
        var = torch.mean(x.pow(2), dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm


def get_norm(norm_type: str, dim: int) -> nn.Module:
    """Factory helper returning either RMSNorm or standard LayerNorm."""
    if norm_type.lower() == "rmsnorm":
        return RMSNorm(dim)
    elif norm_type.lower() == "layernorm":
        return nn.LayerNorm(dim)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Must be 'rmsnorm' or 'layernorm'.")
