"""
Tests for BPE Tokenizer encoding and decoding.
"""

import pytest


def test_tokenizer_roundtrip_reconstruction():
    """Verify that decoding encoded tokens perfectly reconstructs the original text."""
    # TODO: Check that decode(encode(x)) == x across various sample strings, including whitespace and unicode
    pytest.skip("TODO: implement after the module exists")


def test_tokenizer_vocab_size_adherence():
    """Verify tokenizer vocabulary size matches configured target vocabulary length."""
    # TODO: Verify len(tokenizer.get_vocab()) == configured_vocab_size and token IDs are contiguous from 0 to vocab_size-1
    pytest.skip("TODO: implement after the module exists")


def test_tokenizer_special_tokens():
    """Verify tokenizer correctly encodes and decodes special tokens (<|endoftext|>)."""
    # TODO: Verify <|endoftext|> is assigned a unique reserved ID and retains its identity across encode/decode
    pytest.skip("TODO: implement after the module exists")
