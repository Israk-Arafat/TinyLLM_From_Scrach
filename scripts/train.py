"""Entry point for training."""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import torch
import yaml

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.model_config) as f:
        model_cfg_dict = yaml.safe_load(f)["model"]
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)["training"]
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)["data"]

    torch.manual_seed(train_cfg.get("seed", 42))
    random.seed(train_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    model_cfg = ModelConfig.from_dict(model_cfg_dict)
    model = Transformer(model_cfg)
    logging.info("Model parameters: %s", f"{model.num_parameters():,}")

    train_loader, val_loader = build_dataloaders(data_cfg)

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
    trainer.train()


if __name__ == "__main__":
    main()
