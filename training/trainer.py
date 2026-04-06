"""Main training loop."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from model import Transformer
from evaluation.metrics import evaluate

logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


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
        self._step = 0       # micro-batch counter
        self._opt_step = 0   # optimizer-update counter (used for eval/save intervals)

        self._use_amp = cfg.get("use_amp", True) and device.type == "cuda"
        # bfloat16 is preferred on A100 (no overflow risk, no loss scaling needed)
        self._amp_dtype = torch.bfloat16

        # Validation batches are pre-materialised once to avoid re-streaming from HF
        self._val_batches: List[dict] | None = None

        # W&B
        self._use_wandb = cfg.get("use_wandb", False) and _WANDB_AVAILABLE
        if self._use_wandb:
            wandb.init(
                project=cfg.get("wandb_project", "tinyllm"),
                config=cfg,
                resume="allow",
            )
            wandb.watch(self.model, log_freq=500)

    def resume_from_checkpoint(self, path: str) -> None:
        """Load model, optimizer, scheduler and step counter from a saved checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self._step = ckpt.get("step", 0)
        self._opt_step = ckpt.get("opt_step", 0)
        logger.info("Resumed training from %s at step %d (opt_step %d)", path, self._step, self._opt_step)

    def _prefetch_val_batches(self) -> None:
        max_val_batches = self.cfg.get("max_val_batches", 100)
        logger.info("Pre-materialising %d validation batches...", max_val_batches)
        self._val_batches = []
        for i, batch in enumerate(self.val_loader):
            if i >= max_val_batches:
                break
            # Keep on CPU to save VRAM; move to device per-batch during eval
            self._val_batches.append({k: v for k, v in batch.items()})
        logger.info("Cached %d validation batches", len(self._val_batches))

    def train(self) -> None:
        grad_accum = self.cfg.get("gradient_accumulation_steps", 1)
        max_steps = self.cfg["max_steps"]
        max_grad_norm = self.cfg.get("max_grad_norm", 1.0)
        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Pre-materialise validation set before training starts
        self._prefetch_val_batches()

        self.model.train()
        self.optimizer.zero_grad()

        for batch in self.train_loader:
            if self._opt_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
                outputs = self.model(input_ids, labels=labels)
                loss = outputs["loss"] / grad_accum

            loss.backward()

            if (self._step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self._opt_step += 1

                if self._opt_step % self.cfg.get("eval_interval", 500) == 0:
                    val_loss = evaluate(self.model, self._val_batches, self.device,
                                        use_amp=self._use_amp, amp_dtype=self._amp_dtype)
                    train_loss_val = loss.item() * grad_accum
                    logger.info("opt_step=%d  train_loss=%.4f  val_loss=%.4f",
                                self._opt_step, train_loss_val, val_loss)
                    if self._use_wandb:
                        wandb.log({"train_loss": train_loss_val, "val_loss": val_loss,
                                   "lr": self.scheduler.get_last_lr()[0]}, step=self._opt_step)
                    self.model.train()

                if self._opt_step % self.cfg.get("save_interval", 2000) == 0:
                    self._save_checkpoint(checkpoint_dir)

            self._step += 1

        # Flush any gradients accumulated in a partial cycle at the end of training
        pending = self._step % grad_accum
        if pending != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self._opt_step += 1

        self._save_checkpoint(checkpoint_dir, name="final")

    def _save_checkpoint(self, checkpoint_dir: Path, name: str | None = None) -> None:
        fname = f"step_{self._opt_step}.pt" if name is None else f"{name}.pt"
        path = checkpoint_dir / fname
        torch.save(
            {
                "step": self._step,
                "opt_step": self._opt_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)
