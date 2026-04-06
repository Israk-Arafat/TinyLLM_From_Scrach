"""Token and positional embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        context_length: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        return self.dropout(self.token_emb(input_ids) + self.pos_emb(positions))
