# MiniGPT-X

MiniGPT-X is a small (10–30M parameter) GPT-style autoregressive language model built completely from scratch in PyTorch, engineered and benchmarked natively on Apple Silicon Mac using the Metal Performance Shaders (MPS) backend. Designed as an in-depth portfolio and learning project, it covers the full lifecycle of a modern language model: Byte-Pair Encoding (BPE) tokenization, modern Transformer architecture innovations, an engineered training loop, accelerated inference with KV caching, a streaming FastAPI server, a chat interface, and rigorous empirical ablations.

## Planned Features

- **Byte-Pair Encoding (BPE)**: Custom-trained subword tokenizer from scratch with save/load merge tables.
- **Modern Transformer Architecture**:
  - Rotary Position Embeddings (RoPE)
  - Root Mean Square Normalization (RMSNorm)
  - SwiGLU feed-forward networks
  - PyTorch `scaled_dot_product_attention` (SDPA / Flash kernel)
  - Weight tying between token embeddings and language model output projection
- **Robust Training Pipeline**:
  - AdamW optimizer with 2D weight decay separation
  - Learning rate schedule with linear warmup and cosine decay
  - Gradient accumulation and gradient clipping
  - Complete checkpointing and metric tracking (loss, perplexity, throughput, memory)
- **High-Performance Inference**:
  - Greedy decoding, temperature scaling, top-k filtering, and nucleus (top-p) sampling
  - Key-Value (KV) cache for fast autoregressive generation
- **Production Deployment**:
  - FastAPI server with Server-Sent Events (SSE) token streaming
  - Interactive browser chat UI
- **Empirical Rigor**:
  - Benchmark reports comparing baseline vs modern techniques
  - Controlled ablation studies on RoPE, RMSNorm, SwiGLU, SDPA, and KV cache speedups

## Repository Structure

```
minigpt-x/
  configs/        # Model and training hyperparameter configurations (small.yaml, medium.yaml)
  data/           # Raw text datasets and processed/tokenized binary datasets
  tokenizer/      # BPE tokenizer training and encoding/decoding modules
  model/          # Transformer architecture (attention, RoPE, RMSNorm, SwiGLU, GPT)
  training/       # Training loop, optimizer, scheduler, and checkpoint management
  evaluation/     # Validation loss, perplexity calculation, and throughput benchmarks
  inference/      # Autoregressive generation sampling and KV cache implementation
  api/            # FastAPI REST and SSE streaming inference server
  app/            # Lightweight chat web UI
  scripts/        # Environment diagnostics and MPS benchmarking tools
  experiments/    # Experiment run logs (CSV metric files)
  checkpoints/    # Saved model weights and training checkpoints
  docs/           # Project roadmap, benchmark report, and ablation documentation
  notes/          # Personal learning log and architectural reflections
  tests/          # Pytest verification suite
```

## Setup

Ensure you are using the system Python interpreter: `/usr/local/bin/python3`.

1. **Install Dependencies**:
   ```bash
   /usr/local/bin/python3 -m pip install -r requirements.txt
   ```

2. **Verify Hardware Environment**:
   ```bash
   make env
   ```
   *(or `/usr/local/bin/python3 check_env.py`)*

3. **Run Test Suite**:
   ```bash
   make test
   ```
   *(or `/usr/local/bin/python3 -m pytest -q`)*

4. **Benchmark MPS Hardware**:
   ```bash
   make bench
   ```
   *(or `/usr/local/bin/python3 scripts/bench_mps.py`)*

## Status

See [ROADMAP.md](file:///Users/praveenchourasia/Projects/minigpt-x/docs/ROADMAP.md) for the active checklist across all 6 phases and Module 0.

## Results

TBD

## Limitations

MiniGPT-X is a small educational language model (10–30M parameters) developed strictly for learning and demonstration purposes. It is not an enterprise-grade or production conversational assistant, has limited world knowledge, and will exhibit hallucination.

