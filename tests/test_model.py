"""
==============================================================================
Module: tests/test_model.py
Role: Automated Unit Test Suite for Bi-LSTM & GRU Classifier Architectures
Reference: implementation_plan.md -> Section 4.6.1, 4.6.4, 4.6.5, Day 9
==============================================================================
"""

import pytest
import torch
from src.model.lstm_model import ISLRecognizer
from src.model.gru_model import ISLRecognizerGRU


def test_bilstm_forward_pass_shape():
    """
    Test 1: Verifies Bi-LSTM input (batch, 30, 225) -> output (batch, 65) tensor flow.
    """
    model = ISLRecognizer(num_classes=65)
    x = torch.randn(4, 30, 225)
    logits = model(x)

    assert logits.shape == (4, 65)
    assert not torch.isnan(logits).any()


def test_bilstm_parameter_count():
    """
    Test 2: Verifies Bi-LSTM parameter count matches ~2.85M params estimate.
    """
    model = ISLRecognizer(num_classes=65)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert 2_700_000 <= total_params <= 3_100_000, f"Unexpected parameter count: {total_params}"


def test_gru_forward_pass_shape():
    """
    Test 3: Verifies GRU fallback model forward shape (batch, 65) and ~400K param count.
    """
    model = ISLRecognizerGRU(num_classes=65)
    x = torch.randn(2, 30, 225)
    logits = model(x)

    assert logits.shape == (2, 65)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 250_000 <= total_params <= 350_000, f"Unexpected GRU parameter count: {total_params}"


if __name__ == "__main__":
    test_bilstm_forward_pass_shape()
    test_bilstm_parameter_count()
    test_gru_forward_pass_shape()
    print("[SUCCESS] All test_model tests passed.")
