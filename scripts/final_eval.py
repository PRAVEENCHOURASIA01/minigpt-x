"""Final evaluation of a MiniGPT-X checkpoint. Reports only measured numbers."""
import argparse
import dataclasses
import json
import math
import os

import numpy as np
import torch

from model.gpt import GPTConfig, MiniGPTX
from tokenizer.encode import BPETokenizer
from evaluation.evaluate import estimate_loss

PROMPTS = [
    "Once upon a time, a little girl named Lily",
    "Tom and his dog went to the park",
    "One day, a small bird found a shiny key",
    "The sun was bright and Mia wanted to",
    "Ben was sad because his toy was broken. Then",
]


def load_tokenizer(path):
    try:
        tok = BPETokenizer.load(path)      # works if load is a classmethod
        if tok is not None:
            return tok
    except TypeError:
        pass
    tok = BPETokenizer()
    tok.load(path)                         # works if load is an instance method
    return tok


@torch.no_grad()
def sample(model, tok, prompt, device, block_size, max_new=150,
           temperature=0.8, top_p=0.9, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new):
        out = model(idx[:, -block_size:])
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[:, -1, :].float() / temperature
        probs = torch.softmax(logits, dim=-1).cpu()
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sp, dim=-1)
        sp[cum - sp > top_p] = 0                      # nucleus (top-p) filter
        sp = sp / sp.sum(dim=-1, keepdim=True)
        nxt = si.gather(-1, torch.multinomial(sp, 1, generator=g))
        idx = torch.cat([idx, nxt.to(device)], dim=1)
    return tok.decode(idx[0].tolist()).split("<|endoftext|>")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/main_best.pt")
    ap.add_argument("--val", default="data/processed/val.bin")
    ap.add_argument("--tokenizer", default="tokenizer/bpe_4096.json")
    ap.add_argument("--eval_iters", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_cfg = ckpt["config"]["model"]
    names = {f.name for f in dataclasses.fields(GPTConfig)}
    cfg = GPTConfig(**{k: v for k, v in model_cfg.items() if k in names})
    model = MiniGPTX(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    block_size = cfg.block_size

    tok = load_tokenizer(args.tokenizer)
    val = np.memmap(args.val, dtype=np.uint16, mode="r")

    # 1) validation loss and perplexity on fixed windows (seeded inside estimate_loss)
    loss, ppl, _assumed_bpb = estimate_loss(
        model, val, args.batch_size, block_size, args.eval_iters, torch.device(device))

    # 2) bits-per-byte using a MEASURED tokens-per-byte ratio from real validation text
    n_tok = min(1_000_000, len(val))
    text = tok.decode(val[:n_tok].tolist())
    tokens_per_byte = n_tok / len(text.encode("utf-8"))
    bpb = loss / math.log(2) * tokens_per_byte

    results = {
        "checkpoint": args.ckpt, "device": device, "parameters": n_params,
        "val_loss_nats": loss, "perplexity_per_token": ppl,
        "measured_tokens_per_byte": tokens_per_byte,
        "measured_bytes_per_token": 1 / tokens_per_byte,
        "bits_per_byte": bpb,
        "eval_windows": args.eval_iters * args.batch_size, "block_size": block_size,
    }
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/final_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # 3) sample stories (fixed seed, temperature 0.8, top-p 0.9)
    os.makedirs("docs", exist_ok=True)
    with open("docs/SAMPLES.md", "w") as f:
        f.write("# Sample generations\n\n"
                f"Checkpoint: `{args.ckpt}`. Settings: temperature 0.8, top-p 0.9, "
                "seed 0, 150 new tokens. Samples are not cherry-picked.\n\n")
        for p in PROMPTS:
            s = sample(model, tok, p, device, block_size)
            f.write(f"## Prompt: {p}\n\n{s}\n\n")
            print("\n--- " + p + "\n" + s)


if __name__ == "__main__":
    main()