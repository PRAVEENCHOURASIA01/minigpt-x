# MiniGPT-X Project Roadmap

This document outlines the step-by-step roadmap for building, training, evaluating, and serving MiniGPT-X from scratch on Apple Silicon.

---

## Module 0: Environment & Foundations
- [ ] Verify Python 3.14 environment, PyTorch installation, and Apple Silicon MPS backend availability (`check_env.py`)
- [ ] Benchmark float32 matrix multiplication throughput on CPU vs MPS (`scripts/bench_mps.py`)
- [ ] Understand PyTorch tensors, storage layout, broadcasting, and automatic differentiation (`autograd`)
- [ ] Pass baseline environment tests (`pytest tests/test_env.py`)

---

## Phase 1: Rebuild the Foundation
- [ ] Implement configurable Transformer architectural skeleton
- [ ] Map out and verify tensor shapes across all intermediate layers
- [ ] Implement causal lower-triangular attention mask
- [ ] Implement cross-entropy loss computation
- [ ] Pass tests in `tests/test_causal_mask.py` and `tests/test_loss.py`

---

## Phase 2: Modern Architecture
- [ ] Build and train Byte-Pair Encoding (BPE) tokenizer from scratch (`tokenizer/train_tokenizer.py`, `tokenizer/encode.py`)
- [ ] Implement Rotary Position Embeddings (RoPE) (`model/embeddings.py`)
- [ ] Implement Root Mean Square Normalization (RMSNorm) (`model/normalization.py`)
- [ ] Implement SwiGLU feed-forward networks (`model/transformer.py`)
- [ ] Integrate PyTorch `scaled_dot_product_attention` (SDPA) with causal masking (`model/attention.py`)
- [ ] Assemble full architecture (`model/gpt.py`) and pass unit tests (`tests/test_tokenizer.py`, `tests/test_rope.py`, `tests/test_rmsnorm.py`)

---

## Phase 3: Training Engineering
- [ ] Implement AdamW optimizer configuration with 2D weight decay splitting (`training/optimizer.py`)
- [ ] Implement learning rate schedule with linear warmup and cosine decay (`training/scheduler.py`)
- [ ] Implement gradient accumulation and gradient clipping (`training/train.py`)
- [ ] Implement checkpoint saving and state recovery (`training/checkpoint.py`)
- [ ] Implement validation metric evaluation and perplexity calculation (`evaluation/evaluate.py`, `evaluation/perplexity.py`)
- [ ] Execute full training runs for `configs/small.yaml` and `configs/medium.yaml` and record metric CSVs

---

## Phase 4: Inference
- [ ] Implement autoregressive generation loop (`inference/generate.py`)
- [ ] Add sampling controls: temperature scaling, top-k filtering, and top-p (nucleus) filtering
- [ ] Implement Key-Value (KV) cache to eliminate redundant computation during decoding (`inference/kv_cache.py`)
- [ ] Pass KV cache test suite (`tests/test_kv_cache.py`)
- [ ] Benchmark prefill latency (TTFT) and decode tokens-per-second with and without KV cache (`evaluation/benchmark.py`)

---

## Phase 5: Deployment
- [ ] Implement FastAPI server exposing health, metadata, and generation endpoints (`api/server.py`)
- [ ] Add Server-Sent Events (SSE) token-by-token streaming endpoint
- [ ] Build interactive lightweight chat UI in `app/` connected to the FastAPI streaming server
- [ ] Support dynamic config-driven model and checkpoint loading

---

## Phase 6: Documentation and Benchmarking
- [ ] Generate training loss and validation perplexity convergence curves
- [ ] Complete `docs/BENCHMARK_REPORT.md` comparing baseline vs modern architecture
- [ ] Execute ablation experiments and document findings in `docs/ABLATIONS.md`
- [ ] Document project limitations, educational reflections, and future improvements

