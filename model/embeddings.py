"""Token embeddings. Positional information is handled by RoPE in the attention layers."""
from __future__ import annotations

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.token_emb(input_ids))
