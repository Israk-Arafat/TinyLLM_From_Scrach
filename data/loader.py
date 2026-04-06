"""Stream SlimPajama and build packed token dataloaders."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, IterableDataset

from .cleaner import clean_sample
from .tokenizer import load_tokenizer
from .packing import TokenPacker


def build_dataloaders(data_cfg: dict, batch_size: int = 1) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) from a data config dict."""
    from datasets import load_dataset

    tokenizer = load_tokenizer(data_cfg["tokenizer_name"])

    train_ds = load_dataset(
        data_cfg["dataset_name"],
        split=data_cfg["split"],
        streaming=data_cfg.get("streaming", True),
    )
    val_ds = load_dataset(
        data_cfg["dataset_name"],
        split=data_cfg["val_split"],
        streaming=data_cfg.get("streaming", True),
    )

    def process(dataset):
        packer = TokenPacker(chunk_size=data_cfg["chunk_size"])
        for sample in dataset:
            text = clean_sample(sample.get("text", ""), min_len=data_cfg["min_text_length"])
            if text is None:
                continue
            token_ids = tokenizer.encode(text)
            # Append EOS so the model learns document boundaries inside packed chunks
            token_ids.append(tokenizer.eos_token_id)
            chunks = packer.add(token_ids)
            yield from chunks
        yield from packer.flush()

    class _ChunkDataset(IterableDataset):
        def __init__(self, ds):
            self._ds = ds

        def __iter__(self):
            for chunk in process(self._ds):
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}

    # num_workers=0: streaming IterableDataset does not support multi-process loading
    num_workers = data_cfg.get("num_workers", 0)
    train_loader = DataLoader(
        _ChunkDataset(train_ds),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        _ChunkDataset(val_ds),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, val_loader
