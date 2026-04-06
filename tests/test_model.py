"""Tests for the Transformer model."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import Transformer, ModelConfig


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=100,
        context_length=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
    )


class TestTransformer:
    def test_forward_shape(self, tiny_cfg):
        model = Transformer(tiny_cfg)
        input_ids = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
        output = model(input_ids)
        assert output["logits"].shape == (2, 16, tiny_cfg.vocab_size)

    def test_loss_computed_with_labels(self, tiny_cfg):
        model = Transformer(tiny_cfg)
        input_ids = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
        labels = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
        output = model(input_ids, labels=labels)
        assert "loss" in output
        assert output["loss"].item() > 0

    def test_num_parameters_positive(self, tiny_cfg):
        model = Transformer(tiny_cfg)
        assert model.num_parameters() > 0
