"""
Tests for cross-entropy loss computation.
"""

import pytest


def test_initial_loss_near_uniform_entropy():
    """Verify untrained model loss is close to theoretical uniform cross-entropy: -ln(1/vocab_size)."""
    # TODO: Verify that at initialization, cross_entropy(logits, targets) is approximately equal to ln(vocab_size) within a small tolerance
    pytest.skip("TODO: implement after the module exists")


def test_loss_target_ignore_index():
    """Verify loss calculation properly ignores masked padding or special ignore index (-1)."""
    # TODO: Pass target sequence containing -1 ignore index and verify gradients and loss ignore those positions
    pytest.skip("TODO: implement after the module exists")


def test_loss_decreases_over_single_batch():
    """Verify standard gradient descent step on a single batch strictly reduces cross-entropy loss."""
    # TODO: Perform one forward-backward-optimizer step on an overfitted batch and verify loss drops
    pytest.skip("TODO: implement after the module exists")
