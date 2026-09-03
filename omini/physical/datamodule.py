from __future__ import annotations

import os
from typing import Any, Mapping

import lightning as L
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .data import PairedVisibleInfraredDataset, build_dataset
from .validation import build_validation_dataloader


def _world_size_and_rank() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))


class PhysicalDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_config: Mapping[str, Any],
        *,
        train_batch_size: int,
        validation_batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        if train_batch_size <= 0 or validation_batch_size <= 0 or num_workers < 0:
            raise ValueError("Batch sizes must be positive and num_workers non-negative.")
        self.dataset_config = dict(dataset_config)
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.enable_validation = bool(self.dataset_config.get("enable_validation", True))
        self.train_dataset: PairedVisibleInfraredDataset | None = None
        self.validation_dataset: PairedVisibleInfraredDataset | None = None

    @property
    def dataset_name(self) -> str:
        return str(self.dataset_config["name"]).lower()

    def setup(self, stage: str | None = None) -> None:
        normalized_stage = "" if stage is None else str(stage).lower()
        if stage is None or normalized_stage in {"fit", "trainerfn.fitting"}:
            self.train_dataset = build_dataset(
                self.dataset_name,
                self.dataset_config["train_annotations"],
                self.dataset_config["root"],
                horizontal_flip_prob=float(self.dataset_config.get("horizontal_flip_prob", 0.5)),
            )
            if self.enable_validation:
                self.validation_dataset = build_dataset(
                    self.dataset_name,
                    self.dataset_config["validation_annotations"],
                    self.dataset_config["root"],
                    horizontal_flip_prob=0.0,
                )
            train_limit = self.dataset_config.get("train_num_samples")
            validation_limit = self.dataset_config.get("validation_num_samples")
            if train_limit is not None:
                self.train_dataset.records = self.train_dataset.records[: int(train_limit)]
            if validation_limit is not None and self.validation_dataset is not None:
                self.validation_dataset.records = self.validation_dataset.records[: int(validation_limit)]

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting the train dataloader.")
        world_size, rank = _world_size_and_rank()
        sampler = (
            DistributedSampler(
                self.train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )
            if world_size > 1
            else None
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Any]] | None:
        if not self.enable_validation:
            return None
        if self.validation_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting the validation dataloader.")
        world_size, rank = _world_size_and_rank()
        return build_validation_dataloader(
            self.validation_dataset,
            batch_size=self.validation_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            num_replicas=world_size if world_size > 1 else None,
            rank=rank if world_size > 1 else None,
        )


__all__ = ["PhysicalDataModule"]
