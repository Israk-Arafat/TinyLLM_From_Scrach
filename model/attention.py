"""Multi-head causal self-attention with Rotary Position Embeddings (RoPE)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Return complex frequency tensor of shape (max_seq_len, head_dim // 2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x of shape (B, H, T, head_dim)."""
    T = x.shape[2]
    # View last dim as complex pairs, rotate, flatten back
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_r)
    out = torch.view_as_real(x_complex * freqs_cis[:T]).flatten(-2)
    return out.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Apply RoPE to queries and keys (freqs_cis already sliced to current positions)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=past_kv is None,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out)), (k, v)
