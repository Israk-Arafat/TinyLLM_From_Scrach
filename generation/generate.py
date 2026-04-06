"""Autoregressive text generation."""
from __future__ import annotations

import torch

from data.tokenizer import TiktokenTokenizer
from model import Transformer


@torch.no_grad()
def generate(
    model: Transformer,
    tokenizer: TiktokenTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
    device: torch.device | None = None,
) -> str:
    """Autoregressively generate text continuing from `prompt`."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    context_length = model.cfg.context_length

    for _ in range(max_new_tokens):
        # Crop to context window
        input_ids_cropped = input_ids[:, -context_length:]
        logits = model(input_ids_cropped)["logits"]
        next_logits = logits[:, -1, :]  # last position

        if temperature != 1.0:
            next_logits = next_logits / temperature

        if top_k > 0:
            values, _ = torch.topk(next_logits, top_k)
            threshold = values[:, -1].unsqueeze(-1)
            next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_ids = input_ids[0, len(prompt_ids) :].tolist()
    return prompt + tokenizer.decode(generated_ids)
