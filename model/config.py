"""Model hyperparameters dataclass."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 100277
    context_length: int = 2048
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4    # GQA: 4 KV heads shared across 16 Q heads → 4× smaller KV cache
    n_layers: int = 24
    d_ff: int = 2816
    dropout: float = 0.0
    tie_weights: bool = True
    rope_theta: float = 500000.0  # RoPE base frequency

    @classmethod
    def from_dict(cls, cfg: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})
