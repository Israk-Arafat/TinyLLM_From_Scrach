"""Validation metrics."""
from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from model import Transformer


@torch.no_grad()
def evaluate(
    model: Transformer,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
) -> float:
    """Return mean validation loss over up to `max_batches` batches."""
    model.eval()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        labels = batch["labels"].unsqueeze(0).to(device)
        outputs = model(input_ids, labels=labels)
        total_loss += outputs["loss"].item()
        count += 1
    return total_loss / max(1, count)


def perplexity(loss: float) -> float:
    return math.exp(loss)
