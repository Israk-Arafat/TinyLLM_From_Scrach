"""Validation metrics."""
from __future__ import annotations

import math
from typing import List

import torch

from model import Transformer


@torch.no_grad()
def evaluate(
    model: Transformer,
    val_batches: List[dict],
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> float:
    """Return mean validation loss over pre-materialised validation batches."""
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in val_batches:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(input_ids, labels=labels)
        total_loss += outputs["loss"].item()
        count += 1
    return total_loss / max(1, count)


def perplexity(loss: float) -> float:
    return math.exp(loss)
