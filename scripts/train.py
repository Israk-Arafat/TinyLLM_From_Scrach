"""Entry point for training."""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

# Reduce CUDA allocator fragmentation — helps when reserved-but-unallocated
# memory cannot satisfy new allocations due to non-contiguous free blocks.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Too many HTTP request logs from the HuggingFace streaming dataloader
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import build_dataloaders
from model import Transformer, ModelConfig
from training import Trainer
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyLLM")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint_dir from train config")
    parser.add_argument("--resume", default=None, help="Path to a .pt checkpoint to resume training from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.model_config) as f:
        model_cfg_dict = yaml.safe_load(f)["model"]
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)["training"]
    if args.checkpoint_dir is not None:
        train_cfg["checkpoint_dir"] = args.checkpoint_dir
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)["data"]

    torch.manual_seed(train_cfg.get("seed", 42))
    random.seed(train_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    model_cfg = ModelConfig.from_dict(model_cfg_dict)

    chunk_size = data_cfg.get("chunk_size")
    if chunk_size != model_cfg.context_length:
        raise ValueError(
            f"data_config chunk_size ({chunk_size}) must equal "
            f"model_config context_length ({model_cfg.context_length}). "
            "Update configs/data_config.yaml to match."
        )

    model = Transformer(model_cfg)
    logging.info("Model parameters: %s", f"{model.num_parameters():,}")

    # torch.compile gives a significant throughput boost on PyTorch ≥ 2.0
    if train_cfg.get("compile", False):
        logging.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing = True
        logging.info("Gradient checkpointing enabled — activations recomputed during backward")

    batch_size = train_cfg.get("batch_size", 1)
    train_loader, val_loader = build_dataloaders(data_cfg, batch_size=batch_size)

    optimizer = build_optimizer(model, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, warmup_steps=train_cfg["warmup_steps"], max_steps=train_cfg["max_steps"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=train_cfg,
        device=device,
    )
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
