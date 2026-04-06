"""Entry point for text generation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.tokenizer import load_tokenizer
from model import Transformer, ModelConfig
from generation import generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with TinyLLM")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
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

    tokenizer = load_tokenizer(data_cfg["tokenizer_name"])

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(output)


if __name__ == "__main__":
    main()
