"""
Evaluation Loop and Metric Aggregation.

This module evaluates a trained MiniGPT-X model on validation datasets:
- Computes mean cross-entropy loss over multiple evaluation batches
- Reports token accuracy and validation perplexity
- Supports zero-shot or few-shot evaluation prompts
"""

# TODO: Define evaluate_loss(model, data_loader, eval_iters, device) -> float
# TODO: Set model to eval mode (model.eval()) and use torch.no_grad()
# TODO: Iterate over eval_iters batches, compute average cross-entropy loss
# TODO: Restore model to training mode (model.train())
# TODO: Calculate perplexity from average validation loss
