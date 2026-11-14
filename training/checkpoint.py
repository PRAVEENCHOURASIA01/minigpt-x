"""
Model Checkpointing and Atomic State Management.
"""

import os
import shutil
import tempfile
from typing import Any, Dict, Optional
import torch


def save_checkpoint(
    checkpoint_dir: str,
    tag: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    config: Dict[str, Any],
    tokenizer_path: str,
    is_best: bool = False,
):
    """
    Saves a training checkpoint atomically using a temporary file and rename.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "step": step,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "tokenizer_path": tokenizer_path,
    }

    # Save latest
    latest_path = os.path.join(checkpoint_dir, f"{tag}_latest.pt")
    temp_fd, temp_path = tempfile.mkstemp(dir=checkpoint_dir, prefix="ckpt_tmp_")
    os.close(temp_fd)
    torch.save(state, temp_path)
    shutil.move(temp_path, latest_path)

    # Save best
    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{tag}_best.pt")
        shutil.copyfile(latest_path, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Loads checkpoint state into model and optional optimizer.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
