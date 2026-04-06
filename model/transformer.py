"""Decoder-only transformer with RoPE, RMSNorm, and SwiGLU (LLaMA-style)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention, precompute_rope_freqs
from .embeddings import Embeddings
from .config import ModelConfig


class RMSNorm(nn.Module):
    """RMS normalisation — faster than LayerNorm (no mean centering, no bias)."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: SiLU(xW1) ⊙ (xW3), projected back by W2.
    Uses 3 matrices at 2/3 the width of a standard 2-matrix FFN for equal parameter count."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.dropout)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ff = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        attn_out, new_kv = self.attn(self.ln1(x), freqs_cis, past_kv=past_kv)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_kv


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embeddings = Embeddings(cfg.vocab_size, cfg.d_model, cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.embeddings.token_emb.weight

        # Precompute RoPE freqs once; register as non-persistent buffer so it
        # moves with the model to the correct device automatically.
        freqs_cis = precompute_rope_freqs(
            cfg.d_model // cfg.n_heads, cfg.context_length, cfg.rope_theta
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        # Scale down residual output projections to stabilise the residual stream
        # at initialisation (GPT-2 paper: 1/sqrt(2*n_layers)).
        residual_std = std / math.sqrt(2 * self.cfg.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                init_std = residual_std if name.endswith(("out_proj", "w2")) else std
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_kv: list | None = None,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        T = input_ids.shape[1]
        past_len = past_kv[0][0].shape[2] if past_kv is not None else 0

        x = self.embeddings(input_ids)
        freqs_slice = self.freqs_cis[past_len : past_len + T]

        new_past_kv: list = []
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            x, layer_kv = block(x, freqs_slice, past_kv=layer_past)
            if use_cache:
                new_past_kv.append(layer_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss
        if use_cache:
            result["past_kv"] = new_past_kv
        return result

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
