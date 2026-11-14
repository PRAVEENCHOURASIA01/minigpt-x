"""
Model Checkpointing and State Management.

This module handles saving and loading full training state:
- Model weights (state_dict)
- Optimizer state (momentum, variance)
- Training step and best validation loss
- Model hyperparameters / configuration metadata
"""

# TODO: Define save_checkpoint(model, optimizer, step, loss, config, filepath)
# TODO: Save torch.save dictionary containing model, optimizer, step, and config
# TODO: Define load_checkpoint(filepath, model, optimizer=None, device='cpu')
# TODO: Support strict and non-strict loading for inference vs training resumption
# TODO: Implement prune/cleanup utility to keep only top-k best checkpoints
