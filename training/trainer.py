"""Main training loop."""
from __future__ import annotations

import os
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import Transformer
from evaluation.metrics import evaluate

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: Transformer,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self._step = 0

    def train(self) -> None:
        grad_accum = self.cfg.get("gradient_accumulation_steps", 1)
        max_steps = self.cfg["max_steps"]
        max_grad_norm = self.cfg.get("max_grad_norm", 1.0)
        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.train()
        self.optimizer.zero_grad()

        for batch in self.train_loader:
            if self._step >= max_steps:
                break

            input_ids = batch["input_ids"].unsqueeze(0).to(self.device)
            labels = batch["labels"].unsqueeze(0).to(self.device)

            outputs = self.model(input_ids, labels=labels)
            loss = outputs["loss"] / grad_accum
            loss.backward()

            if (self._step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if self._step % self.cfg.get("eval_interval", 500) == 0:
                val_loss = evaluate(self.model, self.val_loader, self.device)
                logger.info("step=%d  train_loss=%.4f  val_loss=%.4f", self._step, loss.item() * grad_accum, val_loss)
                self.model.train()

            if self._step % self.cfg.get("save_interval", 2000) == 0 and self._step > 0:
                self._save_checkpoint(checkpoint_dir)

            self._step += 1

        self._save_checkpoint(checkpoint_dir, name="final")

    def _save_checkpoint(self, checkpoint_dir: Path, name: str | None = None) -> None:
        fname = f"step_{self._step}.pt" if name is None else f"{name}.pt"
        path = checkpoint_dir / fname
        torch.save(
            {
                "step": self._step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)
