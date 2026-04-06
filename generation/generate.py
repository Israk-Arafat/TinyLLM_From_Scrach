"""Autoregressive text generation with KV-cache and nucleus (top-p) sampling."""
from __future__ import annotations

import torch

from data.tokenizer import TiktokenTokenizer
from model import Transformer


def _apply_sampling(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        # Greedy: return a one-hot distribution at the argmax position
        next_token = logits.argmax(dim=-1, keepdim=True)
        return torch.full_like(logits, float("-inf")).scatter_(-1, next_token, 0.0)
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[:, -1:], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs_sorted = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs_sorted, dim=-1)
        # Remove tokens where the cumulative mass *before* this token exceeds top_p,
        remove = (cum_probs - probs_sorted) > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate(
    model: Transformer,
    tokenizer: TiktokenTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device | None = None,
) -> str:
    """Autoregressively generate text continuing from `prompt`."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    context_length = model.cfg.context_length
    prompt_ids = tokenizer.encode(prompt)[-context_length:]
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Prefill: process entire prompt at once and build KV cache
    outputs = model(input_ids, use_cache=True)
    past_kv = outputs["past_kv"]
    next_logits = outputs["logits"][:, -1, :]

    generated_ids: list[int] = []
    for _ in range(max_new_tokens):
        if len(prompt_ids) + len(generated_ids) >= context_length:
            break  # context window full

        next_logits = _apply_sampling(next_logits, temperature, top_k, top_p)
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(token_id)

        # Single-token forward pass — O(n) per step instead of O(n²)
        outputs = model(next_token, past_kv=past_kv, use_cache=True)
        past_kv = outputs["past_kv"]
        next_logits = outputs["logits"][:, -1, :]

    return prompt + tokenizer.decode(generated_ids)
