"""Tests for text generation."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import Transformer, ModelConfig
from generation import generate


@pytest.fixture
def tiny_model() -> Transformer:
    cfg = ModelConfig(
        vocab_size=100,
        context_length=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
    )
    return Transformer(cfg)


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.decode.return_value = " world"
    tok.eos_token_id = 99
    return tok


class TestGenerate:
    def test_returns_string(self, tiny_model, mock_tokenizer):
        result = generate(tiny_model, mock_tokenizer, "hello", max_new_tokens=5)
        assert isinstance(result, str)

    def test_output_starts_with_prompt(self, tiny_model, mock_tokenizer):
        result = generate(tiny_model, mock_tokenizer, "hello", max_new_tokens=5)
        assert result.startswith("hello")
