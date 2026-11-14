"""
Normalization Layers (RMSNorm and LayerNorm).

This module provides normalization layers used throughout the Transformer:
- RMSNorm (Root Mean Square Normalization): faster, mean-invariant normalization with learnable scale
- LayerNorm (Standard Layer Normalization): standard zero-mean unit-variance normalization with scale and bias
"""

# TODO: Define RMSNorm module inheriting from nn.Module
# TODO: Implement root-mean-square calculation over the final dimension with epsilon
# TODO: Apply learnable gain/weight parameter
# TODO: Implement standard LayerNorm or wrap torch.nn.LayerNorm for ablation comparisons
# TODO: Provide a helper factory get_norm(norm_type, dim)
