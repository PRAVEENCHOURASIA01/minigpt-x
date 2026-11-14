"""
Byte-Pair Encoding (BPE) Tokenizer implementation from scratch.

Features:
- Byte-level base vocabulary (256 raw bytes, mapping 0..255).
- GPT-2 style regex pre-tokenization splitting text into word-like chunks.
- Fast training using pre-token frequency counting and incremental pair statistics.
- Special token handling (<|endoftext|>).
- Serialization to/from JSON.
- Multiprocess dataset tokenization saving uint16 binary files.
"""

import json
import os
import re
from typing import Dict, List, Tuple

# Pre-tokenization regex pattern compatible with Python's standard `re` module
PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    def __init__(self, vocab_size: int = 4096, special_token: str = "<|endoftext|>"):
        self.vocab_size = vocab_size
        self.special_token = special_token
        # Base byte vocabulary: 0..255 mapped to bytes
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # Inverse mapping: bytes -> id
        self.inverse_vocab: Dict[bytes, int] = {bytes([i]): i for i in range(256)}
        # Merges: tuple of (token_id_1, token_id_2) -> merged_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        # Special token ID: initially 256 or set at completion
        self.special_token_id: int = 256
        # Memoization cache for pretoken encoding
        self._encode_cache: Dict[str, List[int]] = {}

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def train_from_text(self, text: str, verbose: bool = True):
        """
        Trains BPE merges on input text up to target self.vocab_size.
        Uses fast dictionary-based pre-token counting.
        """
        if verbose:
            print(f"Pre-tokenizing sample of {len(text):,} characters...")
        raw_words = PRETOKEN_PATTERN.findall(text)
        
        # Count frequency of each unique word chunk
        word_counts: Dict[Tuple[int, ...], int] = {}
        for w in raw_words:
            byte_seq = tuple(w.encode("utf-8"))
            word_counts[byte_seq] = word_counts.get(byte_seq, 0) + 1

        if verbose:
            print(f"Extracted {len(raw_words):,} word chunks ({len(word_counts):,} unique).")
            print(f"Iteratively merging pairs up to vocab size {self.vocab_size}...")

        # We reserve slot for special token
        target_merges = self.vocab_size - 256 - 1  # -256 base bytes, -1 for special token

        for step in range(target_merges):
            pair_counts: Dict[Tuple[int, int], int] = {}
            for seq, count in word_counts.items():
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + count

            if not pair_counts:
                if verbose:
                    print(f"No further pairs found to merge at step {step}.")
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            best_freq = pair_counts[best_pair]
            if best_freq < 2:
                if verbose:
                    print(f"Pair frequency dropped below 2 at step {step}. Halting merges.")
                break

            new_token_id = 256 + step
            self.merges[best_pair] = new_token_id
            merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_token_id] = merged_bytes
            self.inverse_vocab[merged_bytes] = new_token_id

            # Apply merge to word_counts dictionary
            new_word_counts: Dict[Tuple[int, ...], int] = {}
            p0, p1 = best_pair
            for seq, count in word_counts.items():
                if p0 not in seq or p1 not in seq:
                    new_word_counts[seq] = count
                    continue
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == p0 and seq[i + 1] == p1:
                        new_seq.append(new_token_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_word_counts[tuple(new_seq)] = count
            word_counts = new_word_counts

            if verbose and (step + 1) % 500 == 0:
                print(f"Step {step + 1}/{target_merges}: merged {best_pair} -> id {new_token_id} (occurred {best_freq} times)")

        # Register special token at the next available ID
        self.special_token_id = len(self.vocab)
        special_bytes = self.special_token.encode("utf-8")
        self.vocab[self.special_token_id] = special_bytes
        self.inverse_vocab[special_bytes] = self.special_token_id

        if verbose:
            print(f"BPE training complete! Final vocab size: {len(self.vocab)} (Special token ID: {self.special_token_id})")

    def _encode_pretoken(self, piece: str) -> List[int]:
        """Encodes an individual pre-tokenized piece using learned BPE merges."""
        if piece in self._encode_cache:
            return self._encode_cache[piece]

        tokens = list(piece.encode("utf-8"))
        while len(tokens) >= 2:
            best_pair = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    rank = self.merges[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            p0, p1 = best_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    new_tokens.append(self.merges[best_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        self._encode_cache[piece] = tokens
        return tokens

    def encode(self, text: str, allowed_special: bool = True) -> List[int]:
        """
        Encodes a string into a list of token IDs.
        If allowed_special is True, handles special_token cleanly.
        """
        if not text:
            return []

        if allowed_special and self.special_token in text:
            parts = text.split(self.special_token)
            result = []
            for i, part in enumerate(parts):
                if part:
                    result.extend(self.encode(part, allowed_special=False))
                if i < len(parts) - 1:
                    result.append(self.special_token_id)
            return result

        pieces = PRETOKEN_PATTERN.findall(text)
        # Check if findall missed any non-matching characters (like certain unicode or newlines)
        if pieces:
            joined = "".join(pieces)
            if len(joined) != len(text):
                # Fall back to byte-level encoding if regex split missed characters
                return [b for b in text.encode("utf-8")]
        else:
            return [b for b in text.encode("utf-8")]

        token_ids: List[int] = []
        for piece in pieces:
            token_ids.extend(self._encode_pretoken(piece))
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        Recovers raw bytes and handles unicode decode gracefully.
        """
        if not token_ids:
            return ""
        byte_chunks = []
        for tid in token_ids:
            if tid in self.vocab:
                byte_chunks.append(self.vocab[tid])
            else:
                raise ValueError(f"Token ID {tid} not found in vocabulary.")
        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, filepath: str):
        """Serializes tokenizer merges and vocab to a clean JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "special_token": self.special_token,
            "special_token_id": self.special_token_id,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved BPETokenizer ({len(self.vocab)} tokens) to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Loads a BPETokenizer from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"], special_token=data["special_token"])
        tok.special_token_id = data["special_token_id"]
        tok.merges = {
            (int(k.split(",")[0]), int(k.split(",")[1])): v
            for k, v in data["merges"].items()
        }
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        return tok
