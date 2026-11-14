# Architecture and Engineering Design Decisions

This document records the empirical evidence, engineering tradeoffs, and rationale behind non-obvious design choices in MiniGPT-X.

---

## 1. Mixed Precision on Apple Silicon MPS: Float32 vs Bfloat16 / Float16

### Alternatives Considered
- `float32` full precision
- `torch.autocast(device_type="mps", dtype=torch.float16)`
- `torch.autocast(device_type="mps", dtype=torch.bfloat16)`

### Empirical Findings on Apple M5 Mac (16 GB)
1. **Float16 Autocast**:
   - Microbenchmark (500 matmuls): 946.47 ms for `float16` vs 603.79 ms for `float32`.
   - **Result**: `float16` autocast is ~56% *slower* than `float32` due to software casting overhead and memory alignment friction on the MPS backend.
2. **Bfloat16 Autocast**:
   - Full 4-layer Transformer step time (50 iterations): 4.11 s (`bfloat16`) vs 4.37 s (`float32`), an ~6% throughput improvement.
   - 20-step loss comparison: Final loss 2.5537 (`float32`) vs 2.5538 (`bfloat16`), maximum divergence < 0.01 nats.
3. **Decision**:
   - `float32` is selected as the default primary precision for maximum numerical stability and zero autocast casting anomalies across MPS kernels. `bfloat16` is supported via a `--precision` flag.
   - `torch.compile` is strictly avoided on Apple Silicon MPS due to active instability and missing inductor backend support on macOS.

---

## 2. Model Dimension Choices (Medium Config ~12M parameters)

- **Layers**: 6
- **Heads**: 6 (head dimension = 384 / 6 = 64)
- **Embedding Dimension (`n_embd`)**: 384
- **Vocabulary Size**: 4096 (trained custom byte-level BPE)
- **Parameter Breakdown**:
  - Embedding matrix: 4,096 * 384 = 1,572,864
  - Non-embedding parameters: 10,621,824
  - Total parameters (with tied weights): **12,194,688** (~12.2M)
  - This perfectly fits within the project's target envelope of 10M–15M parameters.

---

## 3. Byte-Pair Encoding (BPE) Pre-tokenization & Merges

- Uses standard Python `re` pattern `'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+` to prevent merge contamination across punctuation/whitespace.
- Byte-level base (256 raw bytes) guarantees 100% lossless roundtrip coverage for any unicode, emoji, or unseen character sequence without `<unk>`.
- Final token ID 4095 is reserved as `<|endoftext|>`.

