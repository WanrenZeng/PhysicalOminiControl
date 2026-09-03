from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L


class PeriodicCheckpoint(L.Callback):
    def __init__(self, directory: str | Path, every_n_train_steps: int) -> None:
        super().__init__()
        if every_n_train_steps <= 0:
            raise ValueError("every_n_train_steps must be positive.")
        self.directory = Path(directory)
        self.every_n_train_steps = every_n_train_steps

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if step == 0 or step % self.every_n_train_steps:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(self.directory / f"step_{step}.ckpt")
        trainer.save_checkpoint(self.directory / "last.ckpt")

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(self.directory / "last.ckpt")


__all__ = ["PeriodicCheckpoint"]
