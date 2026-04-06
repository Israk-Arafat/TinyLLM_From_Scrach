"""Entry point for standalone evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import build_dataloaders
from model import Transformer, ModelConfig
from evaluation import evaluate
from evaluation.metrics import perplexity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TinyLLM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--max-batches", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.model_config) as f:
        model_cfg_dict = yaml.safe_load(f)["model"]
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ModelConfig.from_dict(model_cfg_dict)
    model = Transformer(model_cfg)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    _, val_loader = build_dataloaders(data_cfg)
    val_loss = evaluate(model, val_loader, device, max_batches=args.max_batches)
    print(f"val_loss={val_loss:.4f}  perplexity={perplexity(val_loss):.2f}")


if __name__ == "__main__":
    main()
