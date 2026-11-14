"""
Unit tests for the custom Byte-Pair Encoding (BPE) Tokenizer.
"""

import os
import pytest
from tokenizer.encode import BPETokenizer

TOKENIZER_JSON = "tokenizer/bpe_4096.json"


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.exists(TOKENIZER_JSON):
        # Fall back to training a minimal instance if the large file isn't found
        tok = BPETokenizer(vocab_size=300)
        tok.train_from_text("Once upon a time there was a little boy named Ben.")
        return tok
    return BPETokenizer.load(TOKENIZER_JSON)


def test_tokenizer_roundtrip_ascii(tokenizer):
    """Verify standard ASCII strings reconstruct identically."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world! 1234567890",
        "Short.",
        "A",
    ]
    for text in texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Failed roundtrip on {text!r}: got {decoded!r}"


def test_tokenizer_roundtrip_unicode_and_emoji(tokenizer):
    """Verify unicode characters, accents, and emoji reconstruct losslessly."""
    texts = [
        "Café, résumé, naïve, jalapeño.",
        "Emoji: 🌟🚀🎉🔥💡",
        "CJK characters: 你好世界，日本語テスト",
        "Symbols & Math: ∑(x_i^2) ∈ ℝ, π ≈ 3.14159",
    ]
    for text in texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Failed roundtrip on {text!r}: got {decoded!r}"


def test_tokenizer_whitespace_and_empty_edge_cases(tokenizer):
    """Verify empty strings, spaces, tabs, and newlines are preserved exactly."""
    texts = [
        "",
        " ",
        "   ",
        "\n",
        "\n\n\n",
        "\t\t  \r\n",
        "  leading and trailing whitespace  ",
        "line 1\nline 2\r\nline 3\tindented",
    ]
    for text in texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Failed roundtrip on whitespace {text!r}: got {decoded!r}"


def test_tokenizer_special_token_roundtrip(tokenizer):
    """Verify special token <|endoftext|> round-trips as a single token ID."""
    special = "<|endoftext|>"
    encoded = tokenizer.encode(special)
    assert len(encoded) == 1, f"Expected 1 token for special token, got {encoded}"
    assert encoded[0] == tokenizer.special_token_id
    decoded = tokenizer.decode(encoded)
    assert decoded == special

    # Mixed with text
    mixed = f"Story 1 ends here.{special}Story 2 begins here."
    enc_mixed = tokenizer.encode(mixed)
    assert tokenizer.special_token_id in enc_mixed
    dec_mixed = tokenizer.decode(enc_mixed)
    assert dec_mixed == mixed


def test_tokenizer_vocab_size_adherence(tokenizer):
    """Verify tokenizer vocabulary size matches 4096 and IDs are valid."""
    assert tokenizer.get_vocab_size() == 4096
    assert tokenizer.special_token_id in tokenizer.vocab
    assert 0 in tokenizer.vocab
    assert 255 in tokenizer.vocab
