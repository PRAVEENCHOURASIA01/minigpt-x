"""
Autoregressive Text Generation and Streaming for MiniGPT-X.

CLI:
    python3 -m inference.generate --ckpt checkpoints/main_best.pt --prompt "Once upon a time"
"""

import argparse
import sys
from typing import Generator, List, Optional
import torch
import torch.nn.functional as F

from model.gpt import GPTConfig, MiniGPTX
from tokenizer.encode import BPETokenizer
from training.checkpoint import load_checkpoint


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> int:
    """
    Samples next token id from logits using temperature, top-k, and top-p (nucleus) filtering.
    
    Args:
        logits: 1D Tensor of shape (vocab_size,)
        temperature: Scaling factor (0.0 or <= 1e-5 for greedy argmax)
        top_k: Truncates distribution to top k highest probability tokens
        top_p: Nucleus sampling cumulative probability threshold
    """
    if temperature <= 1e-5:
        return torch.argmax(logits, dim=-1).item()

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        val, _ = torch.topk(logits, k)
        min_val = val[-1]
        logits = torch.where(logits < min_val, torch.tensor(float("-inf"), device=logits.device), logits)

    # Top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token


@torch.no_grad()
def generate_stream(
    model: MiniGPTX,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    use_kv_cache: bool = True,
    device: Optional[torch.device] = None,
) -> Generator[str, None, None]:
    """
    Autoregressively generates text token-by-token and yields decoded string chunks.
    Supports KV-cache accelerated single-token decode.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        prompt_tokens = [tokenizer.special_token_id]

    idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    if use_kv_cache:
        # 1. Prefill prompt with use_cache=True
        logits, _, kv_caches = model(idx, kv_caches=None, offset=0, use_cache=True)
        last_logits = logits[0, -1, :]
        next_tok = sample_next_token(last_logits, temperature=temperature, top_k=top_k, top_p=top_p)
        if next_tok == tokenizer.special_token_id:
            return
        yield tokenizer.decode([next_tok])

        # 2. Sequential decode step-by-step
        curr_offset = idx.shape[1]
        for _ in range(max_new_tokens - 1):
            if curr_offset >= model.block_size:
                break
            tok_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            logits, _, kv_caches = model(tok_tensor, kv_caches=kv_caches, offset=curr_offset)
            curr_offset += 1
            next_tok = sample_next_token(logits[0, -1, :], temperature=temperature, top_k=top_k, top_p=top_p)
            if next_tok == tokenizer.special_token_id:
                break
            yield tokenizer.decode([next_tok])

    else:
        # Full context re-forwarding (uncached)
        curr_tokens = list(prompt_tokens)
        for _ in range(max_new_tokens):
            context = curr_tokens[-model.block_size :]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits, _, _ = model(x)
            next_tok = sample_next_token(logits[0, -1, :], temperature=temperature, top_k=top_k, top_p=top_p)
            if next_tok == tokenizer.special_token_id:
                break
            curr_tokens.append(next_tok)
            yield tokenizer.decode([next_tok])


def generate(
    model: MiniGPTX,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    use_kv_cache: bool = True,
    device: Optional[torch.device] = None,
) -> str:
    """Generates complete text response for a given prompt."""
    chunks = list(generate_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        use_kv_cache=use_kv_cache,
        device=device,
    ))
    return "".join(chunks)


def main():
    parser = argparse.ArgumentParser(description="Generate text with MiniGPT-X")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little dog named Spot.", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p filtering")
    parser.add_argument("--no_kv", action="store_true", help="Disable KV cache")
    parser.add_argument("--device", type=str, default="auto", help="Hardware device (auto, mps, cpu)")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = GPTConfig(**ckpt["config"]["model"])
    model = MiniGPTX(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tok_path = ckpt.get("tokenizer_path", "tokenizer/bpe_4096.json")
    tokenizer = BPETokenizer.load(tok_path)

    print(f"Loaded {args.ckpt} on {device}.")
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    sys.stdout.write(args.prompt)
    sys.stdout.flush()

    for chunk in generate_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        use_kv_cache=not args.no_kv,
        device=device,
    ):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
