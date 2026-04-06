"""Stream SlimPajama and build packed token dataloaders."""
from __future__ import annotations

from torch.utils.data import DataLoader

from .cleaner import clean_sample
from .tokenizer import load_tokenizer
from .packing import TokenPacker


def build_dataloaders(data_cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) from a data config dict."""
    from datasets import load_dataset

    tokenizer = load_tokenizer(data_cfg["tokenizer_name"])

    train_ds = load_dataset(
        data_cfg["dataset_name"],
        split=data_cfg["split"],
        streaming=data_cfg.get("streaming", True),
        trust_remote_code=True,
    )
    val_ds = load_dataset(
        data_cfg["dataset_name"],
        split=data_cfg["val_split"],
        streaming=data_cfg.get("streaming", True),
        trust_remote_code=True,
    )

    def process(dataset):
        packer = TokenPacker(chunk_size=data_cfg["chunk_size"])
        for sample in dataset:
            text = clean_sample(sample.get("text", ""), min_len=data_cfg["min_text_length"])
            if text is None:
                continue
            token_ids = tokenizer.encode(text)
            chunks = packer.add(token_ids)
            yield from chunks
        yield from packer.flush()

    # Wrap generators as IterableDataset
    from torch.utils.data import IterableDataset
    import torch

    class _ChunkDataset(IterableDataset):
        def __init__(self, ds):
            self._ds = ds

        def __iter__(self):
            for chunk in process(self._ds):
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}

    train_loader = DataLoader(
        _ChunkDataset(train_ds),
        batch_size=None,  # batching handled upstream via gradient accumulation
    )
    val_loader = DataLoader(
        _ChunkDataset(val_ds),
        batch_size=None,
    )
    return train_loader, val_loader
