"""
Script to train the Byte-Pair Encoding (BPE) tokenizer on TinyStories sample
and encode the full train and validation splits to uint16 binary files.
"""

import multiprocessing as mp
import os
import time
import numpy as np
from tokenizer.encode import BPETokenizer

TRAIN_TXT = "data/raw/train.txt"
VALID_TXT = "data/raw/valid.txt"
TOKENIZER_JSON = "tokenizer/bpe_4096.json"
TRAIN_BIN = "data/processed/train.bin"
VALID_BIN = "data/processed/val.bin"

VOCAB_SIZE = 4096
SAMPLE_BYTES = 15 * 1024 * 1024  # 15 MB for fast BPE merge learning


def _encode_chunk(args):
    text_chunk, tokenizer_path = args
    tok = BPETokenizer.load(tokenizer_path)
    return tok.encode(text_chunk)


def encode_file_multiprocess(file_path: str, out_bin: str, tokenizer_path: str, chunk_size_mb: int = 5):
    print(f"\nProcessing {file_path} -> {out_bin} using multiprocessing...")
    start_time = time.time()
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    total_chars = len(text)
    # Split by stories or approximate chunks
    chunk_char_size = chunk_size_mb * 1024 * 1024
    chunks = []
    curr = 0
    while curr < total_chars:
        next_cut = min(curr + chunk_char_size, total_chars)
        # Try to slice at a boundary
        if next_cut < total_chars:
            boundary = text.rfind("<|endoftext|>", curr, next_cut + 1024)
            if boundary != -1 and boundary > curr:
                next_cut = boundary + len("<|endoftext|>")
        chunks.append((text[curr:next_cut], tokenizer_path))
        curr = next_cut

    print(f"Splitted into {len(chunks)} chunks across CPU cores...")
    num_procs = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=num_procs) as pool:
        results = pool.map(_encode_chunk, chunks)

    # Flatten and convert to uint16
    total_tokens = sum(len(r) for r in results)
    arr = np.empty(total_tokens, dtype=np.uint16)
    offset = 0
    for r in results:
        arr[offset : offset + len(r)] = r
        offset += len(r)

    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(arr.tobytes())

    elapsed = time.time() - start_time
    compression = total_chars / total_tokens if total_tokens else 0
    file_mb = os.path.getsize(out_bin) / (1024 * 1024)
    print(f"Completed {out_bin}: {total_tokens:,} tokens ({file_mb:.2f} MB), {compression:.2f} chars/token, in {elapsed:.1f}s.")
    return total_tokens, compression


def main():
    os.makedirs("tokenizer", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Step 1: Train BPE Tokenizer on sample
    print(f"Reading training text for tokenizer training ({SAMPLE_BYTES / (1024*1024):.0f} MB sample)...")
    with open(TRAIN_TXT, "r", encoding="utf-8", errors="replace") as f:
        sample_text = f.read(SAMPLE_BYTES)

    # Align to end of story
    last_sep = sample_text.rfind("<|endoftext|>")
    if last_sep != -1:
        sample_text = sample_text[:last_sep]

    tok = BPETokenizer(vocab_size=VOCAB_SIZE, special_token="<|endoftext|>")
    t0 = time.time()
    tok.train_from_text(sample_text, verbose=True)
    train_time = time.time() - t0
    print(f"Tokenizer trained in {train_time:.1f}s.")
    tok.save(TOKENIZER_JSON)

    # Step 2: Show example tokenizations
    print("\n--- Example Tokenizations ---")
    test_sentences = [
        "Once upon a time there was a little boy named Ben.",
        "The quick brown fox jumps over the lazy dog! 12345",
        "Hello world! <|endoftext|> A new story begins.",
        "Emoji test: 🌟🚀🎉",
    ]
    for s in test_sentences:
        encoded = tok.encode(s)
        decoded = tok.decode(encoded)
        print(f"Text:    {repr(s)}")
        print(f"Tokens:  {encoded} (len={len(encoded)})")
        print(f"Decoded: {repr(decoded)}")
        assert decoded == s, f"Mismatch in roundtrip: {decoded} != {s}"
        print()

    # Step 3: Encode train and valid splits to uint16 memmap-able files
    train_tokens, train_comp = encode_file_multiprocess(TRAIN_TXT, TRAIN_BIN, TOKENIZER_JSON)
    val_tokens, val_comp = encode_file_multiprocess(VALID_TXT, VALID_BIN, TOKENIZER_JSON)

    print("\n" + "=" * 60)
    print("TOKENIZER & DATASET TOKENIZATION SUMMARY:")
    print("=" * 60)
    print(f"Vocabulary Size:          {tok.get_vocab_size()}")
    print(f"Special Token:            {tok.special_token} (ID {tok.special_token_id})")
    print(f"Train Tokens:             {train_tokens:,} ({train_comp:.2f} chars/token)")
    print(f"Validation Tokens:        {val_tokens:,} ({val_comp:.2f} chars/token)")
    print(f"Train Binary:             {TRAIN_BIN}")
    print(f"Val Binary:               {VALID_BIN}")
    print("=" * 60)


if __name__ == "__main__":
    main()
