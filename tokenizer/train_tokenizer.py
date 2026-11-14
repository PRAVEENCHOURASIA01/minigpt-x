"""
Byte-Pair Encoding (BPE) Tokenizer Training.

This module trains a custom BPE tokenizer from raw text files (e.g. TinyStories).
It reads text data, extracts byte or character sequences, iteratively merges frequent
pairs up to a target vocabulary size, and saves the resulting merge rules and vocab.
"""

# TODO: Define a BPETokenizerTrainer class or training function
# TODO: Read input training text from data/raw/
# TODO: Initialize base byte/character vocabulary
# TODO: Count adjacent token pair frequencies across the corpus
# TODO: Iteratively merge the most frequent pair and record merge rank
# TODO: Stop once the desired target vocabulary size is reached
# TODO: Save the learned vocabulary and merge dictionary to data/processed/
