import csv, math, os, time
import torch
from baseline.data import load_data, get_batch
from baseline.bigram import BigramLM

device = "mps"
batch_size, block_size = 32, 8
lr, max_iters, eval_interval, eval_iters = 1e-2, 3000, 250, 100

torch.manual_seed(1337)
tok, train, val = load_data()
model = BigramLM(tok.vocab_size).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for name, d in (("train", train), ("val", val)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(d, batch_size, block_size, device)
            losses[k] = model(x, y)[1].item()
        out[name] = losses.mean().item()
    model.train()
    return out

os.makedirs("experiments", exist_ok=True)
log_path = "experiments/bigram_baseline.csv"
with open(log_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "train_loss", "val_loss", "perplexity", "lr",
                "tokens_per_sec", "elapsed_s", "peak_mem_mb"])

    start = last = time.time()
    tokens_since = 0
    for step in range(max_iters + 1):
        if step % eval_interval == 0:
            torch.mps.synchronize()
            now = time.time()
            tps = tokens_since / (now - last) if step > 0 else 0
            losses = estimate_loss()
            w.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                        f"{math.exp(losses['val']):.2f}", lr, f"{tps:.0f}",
                        f"{now - start:.1f}",
                        f"{torch.mps.current_allocated_memory() / 1e6:.1f}"])
            f.flush()
            print(f"step {step}: train {losses['train']:.4f} val {losses['val']:.4f} tok/s {tps:.0f}")
            last, tokens_since = time.time(), 0

        x, y = get_batch(train, batch_size, block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tokens_since += batch_size * block_size

@torch.no_grad()
def generate(model, idx, n):
    for _ in range(n):
        logits, _ = model(idx)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx

ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tok.decode(generate(model, ctx, 300)[0].tolist()))