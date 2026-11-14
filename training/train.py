"""
Main Training Script for MiniGPT-X.

CLI:
    python3 -m training.train --config configs/medium.yaml --tag main [--max_minutes N] [--max_iters N] [--resume] [--set key=val]
"""

import argparse
import csv
import datetime
import os
import signal
import sys
import time
from typing import Any, Dict
import numpy as np
import torch
import yaml

from evaluation.evaluate import estimate_loss
from model.gpt import GPTConfig, MiniGPTX
from training.checkpoint import load_checkpoint, save_checkpoint
from training.scheduler import get_lr


class TimeoutHandler:
    def __init__(self, max_minutes: float):
        self.max_seconds = max_minutes * 60.0 if max_minutes > 0 else float("inf")
        self.start_time = time.time()
        self.stop_requested = False

    def should_stop(self) -> bool:
        if self.stop_requested:
            return True
        if time.time() - self.start_time >= self.max_seconds:
            return True
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniGPT-X")
    parser.add_argument("--config", type=str, default="configs/medium.yaml", help="Path to config YAML")
    parser.add_argument("--tag", type=str, default="dev", help="Tag for run logs and checkpoints")
    parser.add_argument("--max_minutes", type=float, default=0.0, help="Wall clock budget in minutes")
    parser.add_argument("--max_iters", type=int, default=None, help="Override maximum iterations")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "bfloat16"], help="Precision")
    parser.add_argument("--set", dest="overrides", action="append", help="Override config keys, e.g. --set model.use_rope=false")
    return parser.parse_args()


def apply_overrides(cfg_dict: Dict[str, Any], overrides: list) -> Dict[str, Any]:
    if not overrides:
        return cfg_dict
    for item in overrides:
        k, v = item.split("=", 1)
        # Parse scalar types
        if v.lower() == "true":
            val = True
        elif v.lower() == "false":
            val = False
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
        # Split dotted hierarchy
        parts = k.split(".")
        d = cfg_dict
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg_dict


def get_batch(data: np.memmap, batch_size: int, block_size: int, device: torch.device):
    max_idx = len(data) - block_size - 1
    ix = np.random.randint(0, max_idx, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def main():
    args = parse_args()

    # Load YAML
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict = apply_overrides(cfg_dict, args.overrides)

    m_cfg = cfg_dict["model"]
    t_cfg = cfg_dict["training"]
    d_cfg = cfg_dict["data"]

    if args.max_iters is not None:
        t_cfg["max_iters"] = args.max_iters

    # Set seeds
    seed = t_cfg.get("seed", 1337)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Select hardware device
    if cfg_dict.get("device", "auto") == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg_dict["device"])

    print(f"=== MiniGPT-X Training ===")
    print(f"Config:     {args.config}")
    print(f"Tag:        {args.tag}")
    print(f"Device:     {device}")
    print(f"Precision:  {args.precision}")
    print(f"Max Iters:  {t_cfg['max_iters']:,}")
    print(f"Batch Size: {t_cfg['batch_size']} x {t_cfg.get('grad_accum_steps', 1)} accum")

    # Load memmap binary datasets
    train_bin = "data/processed/train.bin"
    val_bin = "data/processed/val.bin"
    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    print(f"Loaded train.bin: {len(train_data):,} tokens | val.bin: {len(val_data):,} tokens")

    # Initialize model
    model_config = GPTConfig(**m_cfg)
    model = MiniGPTX(model_config).to(device)
    print(f"Model parameters: {model.num_parameters(non_embedding=False):,} total ({model.num_parameters(non_embedding=True):,} non-emb)")

    # Optimizer & Scheduler
    optimizer = model.configure_optimizers(
        weight_decay=t_cfg.get("weight_decay", 0.1),
        learning_rate=t_cfg.get("learning_rate", 1e-3),
        betas=(0.9, 0.95),
        device_type=device.type,
    )

    # Checkpoint setup
    checkpoint_dir = "checkpoints"
    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        latest_ckpt = os.path.join(checkpoint_dir, f"{args.tag}_latest.pt")
        if os.path.exists(latest_ckpt):
            print(f"Resuming from {latest_ckpt}...")
            ckpt = load_checkpoint(latest_ckpt, model, optimizer, device=device.type)
            start_step = ckpt.get("step", 0) + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"Resumed at step {start_step} with best val_loss={best_val_loss:.4f}")

    # CSV Logging setup
    os.makedirs("experiments", exist_ok=True)
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    csv_path = os.path.join("experiments", f"{today_str}_{os.path.basename(args.config).replace('.yaml','')}_{args.tag}.csv")
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_file.tell() == 0:
        csv_writer.writerow(["step", "train_loss", "val_loss", "perplexity", "lr", "tokens_per_sec", "elapsed_s", "peak_mem_mb"])
        csv_file.flush()
    print(f"Logging metrics to: {csv_path}\n")

    timeout = TimeoutHandler(args.max_minutes)
    tokens_per_step = t_cfg["batch_size"] * t_cfg.get("grad_accum_steps", 1) * model_config.block_size
    train_start_time = time.time()

    # Training loop
    step = start_step
    grad_accum_steps = t_cfg.get("grad_accum_steps", 1)
    use_autocast = (args.precision == "bfloat16" and device.type == "mps")

    while step < t_cfg["max_iters"]:
        if timeout.should_stop():
            print(f"\nTime budget or stop requested. Exiting loop at step {step}.")
            break

        step_t0 = time.time()
        # Update learning rate
        lr = get_lr(
            step,
            learning_rate=t_cfg["learning_rate"],
            min_lr=t_cfg.get("min_lr", 1e-4),
            warmup_iters=t_cfg.get("warmup_iters", 100),
            max_iters=t_cfg["max_iters"],
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        train_loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = get_batch(train_data, t_cfg["batch_size"], model_config.block_size, device)
            if use_autocast:
                with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                    _, loss, _ = model(x, targets=y)
            else:
                _, loss, _ = model(x, targets=y)
            loss = loss / grad_accum_steps
            train_loss_accum += loss.item()
            loss.backward()

        if t_cfg.get("grad_clip", 1.0) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["grad_clip"])

        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()

        step_elapsed = time.time() - step_t0
        tokens_per_sec = tokens_per_step / step_elapsed if step_elapsed > 0 else 0.0
        total_elapsed = time.time() - train_start_time

        # Check memory
        peak_mem_mb = 0.0
        if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            peak_mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)

        # Periodic Evaluation
        is_eval_step = (step % t_cfg.get("eval_interval", 250) == 0) or (step == t_cfg["max_iters"] - 1)
        val_loss, ppl, bpb = None, None, None
        if is_eval_step:
            val_loss, ppl, bpb = estimate_loss(
                model,
                val_data,
                batch_size=t_cfg["batch_size"],
                block_size=model_config.block_size,
                eval_iters=t_cfg.get("eval_iters", 50),
                device=device,
            )
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                tag=args.tag,
                model=model,
                optimizer=optimizer,
                step=step,
                val_loss=val_loss,
                config=cfg_dict,
                tokenizer_path=d_cfg.get("tokenizer_path", "tokenizer/bpe_4096.json"),
                is_best=is_best,
            )
            print(
                f"Step {step:5d}/{t_cfg['max_iters']} | Train Loss: {train_loss_accum:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f} | BPB: {bpb:.2f} | lr: {lr:.2e} | {tokens_per_sec:,.0f} tok/s | Elapsed: {total_elapsed:.1f}s"
            )

        elif step % 25 == 0:
            print(
                f"Step {step:5d}/{t_cfg['max_iters']} | Train Loss: {train_loss_accum:.4f} | lr: {lr:.2e} | {tokens_per_sec:,.0f} tok/s | Elapsed: {total_elapsed:.1f}s"
            )

        # Append to CSV
        csv_writer.writerow([
            step,
            round(train_loss_accum, 4),
            round(val_loss, 4) if val_loss is not None else "",
            round(ppl, 2) if ppl is not None else "",
            f"{lr:.6e}",
            round(tokens_per_sec, 1),
            round(total_elapsed, 1),
            round(peak_mem_mb, 1),
        ])
        csv_file.flush()

        step += 1

    # Save final checkpoint on exit
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        tag=args.tag,
        model=model,
        optimizer=optimizer,
        step=step,
        val_loss=best_val_loss if best_val_loss != float("inf") else 0.0,
        config=cfg_dict,
        tokenizer_path=d_cfg.get("tokenizer_path", "tokenizer/bpe_4096.json"),
        is_best=False,
    )
    csv_file.close()
    print(f"\nTraining completed or saved at step {step}. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
