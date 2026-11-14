"""
Exports a training checkpoint to a lean weights-only checkpoint for deployment.
"""

import argparse
import os
import torch
from model.gpt import GPTConfig, MiniGPTX


def export_checkpoint(input_ckpt: str, output_ckpt: str):
    print(f"Loading {input_ckpt} on CPU...")
    ckpt = torch.load(input_ckpt, map_location="cpu")
    model_cfg = ckpt["config"]["model"]
    state_dict = ckpt["model_state_dict"]

    os.makedirs(os.path.dirname(output_ckpt) or ".", exist_ok=True)
    lean_payload = {
        "config": {"model": model_cfg},
        "model_state_dict": state_dict,
        "tokenizer_path": "tokenizer/bpe_4096.json",
    }
    torch.save(lean_payload, output_ckpt)
    in_size_mb = os.path.getsize(input_ckpt) / (1024 * 1024)
    out_size_mb = os.path.getsize(output_ckpt) / (1024 * 1024)
    print(f"Exported lean checkpoint to {output_ckpt}: {in_size_mb:.2f} MB -> {out_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export lean checkpoint for deployment")
    parser.add_argument("--input", type=str, default="checkpoints/smoke_latest.pt")
    parser.add_argument("--output", type=str, default="deploy/hf_space/model.pt")
    args = parser.parse_args()
    export_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()

