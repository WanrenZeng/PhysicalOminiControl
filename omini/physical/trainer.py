from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Sequence
from typing import Mapping

import lightning as L
import torch
from torch import nn

from .ema import ExponentialMovingAverage
from .training import PhysicalTrainingModel, build_optimizer
from .validation import ValidationMetrics


class PhysicalLightningModule(L.LightningModule):
    def __init__(
        self,
        model: PhysicalTrainingModel,
        *,
        stage: str,
        optimizer_config: Mapping[str, Any],
        ema_config: Mapping[str, Any] | None = None,
        hfrm_loss_config: Mapping[str, float] | None = None,
        joint_loss_config: Mapping[str, Any] | None = None,
        validation_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.stage = stage.lower()
        self.optimizer_config = dict(optimizer_config)
        self.ema_config = dict(ema_config or {})
        self.hfrm_loss_config = dict(hfrm_loss_config or {})
        self.joint_loss_config = dict(joint_loss_config or {})
        self.validation_config = dict(validation_config or {})
        self.model.set_train_stage(self.stage)
        self._ema: ExponentialMovingAverage | None = None
        self._validation_metrics: list[ValidationMetrics] = []
        self._validation_epoch_started = False

        if self.stage not in {"dit", "hfrm", "joint"}:
            raise ValueError("stage must be 'dit', 'hfrm', or 'joint'.")

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        parameters = self.trainable_parameters
        if not parameters:
            raise RuntimeError("No trainable parameters are enabled for the selected stage.")
        return build_optimizer(parameters, self.optimizer_config)

    def setup(self, stage: str | None = None) -> None:
        self.model.set_vae_device(self.device)
        if self.stage in {"dit", "joint"} and self.ema_config.get("enabled", True) and self._ema is None:
            if self.model.dit is None:
                raise RuntimeError("The DiT stage requires a model with include_dit=True.")
            self._ema = ExponentialMovingAverage(
                self.model.dit.parameters(),
                decay=float(self.ema_config.get("decay", 0.9999)),
                update_after_step=int(self.ema_config.get("update_after_step", 0)),
                use_warmup=bool(self.ema_config.get("use_warmup", True)),
                inv_gamma=float(self.ema_config.get("inv_gamma", 1.0)),
                power=float(self.ema_config.get("power", 0.75)),
            )

    def _current_teacher_forcing_prob(self) -> float:
        """Teacher-forcing probability for the current optimizer step.

        Supports a static value (``teacher_forcing_prob``) or an annealing
        schedule that starts high and decays low, so the HFRM leans on the
        ground-truth LL while the DiT is still inaccurate and gradually switches
        to the DiT's own prediction. Annealing keys, when present, take
        precedence:

        * ``teacher_forcing_prob_start`` / ``teacher_forcing_prob_end``
        * ``teacher_forcing_anneal_warmup``: steps to hold the start value
        * ``teacher_forcing_anneal_steps``: steps over which to decay start->end
        * ``teacher_forcing_schedule``: ``linear`` (default) or ``cosine``
        """
        cfg = self.joint_loss_config
        start = cfg.get("teacher_forcing_prob_start")
        end = cfg.get("teacher_forcing_prob_end")
        if start is None or end is None:
            return float(cfg.get("teacher_forcing_prob", 0.5))
        start = float(start)
        end = float(end)
        warmup = int(cfg.get("teacher_forcing_anneal_warmup", 0))
        anneal_steps = int(cfg.get("teacher_forcing_anneal_steps", 0))
        step = int(self.global_step)
        if step <= warmup:
            return start
        if anneal_steps <= 0:
            return end
        progress = (step - warmup) / anneal_steps
        progress = min(max(progress, 0.0), 1.0)
        if str(cfg.get("teacher_forcing_schedule", "linear")).lower() == "cosine":
            fraction = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            fraction = 1.0 - progress
        return end + (start - end) * fraction

    def training_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.stage == "dit":
            loss, logs = self.model.flow_matching_loss(batch)
        elif self.stage == "joint":
            teacher_forcing_prob = self._current_teacher_forcing_prob()
            loss, logs = self.model.end_to_end_loss(
                batch,
                high_frequency_weight=float(self.joint_loss_config.get("high_frequency_weight", 0.5)),
                pixel_weight=float(self.joint_loss_config.get("pixel_weight", 1.0)),
                ms_ssim_weight=float(self.joint_loss_config.get("ms_ssim_weight", 0.0)),
                flow_weight=float(self.joint_loss_config.get("flow_weight", 1.0)),
                teacher_forcing_prob=teacher_forcing_prob,
                teacher_noise_std=float(self.joint_loss_config.get("teacher_noise_std", 0.05)),
            )
            self.log(
                "train/teacher_forcing_prob",
                teacher_forcing_prob,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
                batch_size=batch["visible"].shape[0],
            )
        else:
            loss, logs = self.model.hfrm_loss(
                batch,
                high_frequency_weight=float(self.hfrm_loss_config.get("high_frequency_weight", 0.5)),
                pixel_weight=float(self.hfrm_loss_config.get("pixel_weight", 1.0)),
                ms_ssim_weight=float(self.hfrm_loss_config.get("ms_ssim_weight", 0.0)),
            )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["visible"].shape[0],
        )
        for name, value in logs.items():
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                batch_size=batch["visible"].shape[0],
            )
        return loss

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        if self._ema is not None:
            if self.model.dit is None:
                raise RuntimeError("EMA exists without an enabled DiT.")
            self._ema.step(self.model.dit.parameters())

    def set_validation_metrics(self, metrics: Sequence[ValidationMetrics]) -> None:
        self._validation_metrics = list(metrics)

    def on_validation_epoch_start(self) -> None:
        if not self._validation_metrics:
            raise RuntimeError("Validation ran without configured ValidationMetrics.")
        for index, metrics in enumerate(self._validation_metrics):
            metrics.set_device(self.device)
            metrics.begin_epoch(f"{self.current_epoch}_{index}")
        self._validation_epoch_started = True

    @contextlib.contextmanager
    def _ema_weights(self) -> Iterator[None]:
        if self._ema is None or not self.validation_config.get("use_ema", True):
            yield
            return
        if self.model.dit is None:
            raise RuntimeError("EMA validation requires an enabled DiT.")
        stored_parameters = self._ema.store(self.model.dit.parameters())
        self._ema.copy_to(self.model.dit.parameters())
        try:
            yield
        finally:
            self._ema.restore(self.model.dit.parameters(), stored_parameters)

    def validation_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if not self._validation_epoch_started:
            raise RuntimeError("Validation metrics were not initialized before validation.")
        if dataloader_idx >= len(self._validation_metrics):
            raise IndexError(f"No ValidationMetrics instance configured for dataloader {dataloader_idx}.")

        if self.stage == "hfrm":
            components = self.model._wavelet_components(batch)
            reconstructed, _ = self.model.reconstruct_with_ll(
                components["infrared_ll"], components["visible_high"]
            )
        else:
            with self._ema_weights():
                reconstructed, _, _ = self.model.sample_infrared(
                    batch["visible"],
                    num_inference_steps=int(self.validation_config.get("num_inference_steps", 28)),
                )

        self._validation_metrics[dataloader_idx].update(
            reconstructed,
            batch["infrared"],
            batch["sample_id"],
            target_source_paths=batch.get("infrared_path"),
        )
        self._log_validation_images(dataloader_idx, batch, reconstructed)

    def _log_validation_images(
        self, dataloader_idx: int, batch: Mapping[str, torch.Tensor], reconstructed: torch.Tensor
    ) -> None:
        """Write a few visible|predicted|target panels to TensorBoard on rank 0.

        Gated by ``validation.tensorboard_images`` (default true) and limited to
        ``validation.tensorboard_image_count`` samples (default 4). All tensors
        are RGB in ``[0, 1]`` and share the same spatial size, so they are
        concatenated along width into a single comparison strip.
        """
        if not bool(self.validation_config.get("tensorboard_images", True)):
            return
        if self.logger is None or self.global_rank != 0:
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_image"):
            return
        count = int(self.validation_config.get("tensorboard_image_count", 4))
        count = max(0, min(count, int(reconstructed.shape[0])))
        target = batch["infrared"]
        visible = batch.get("visible")
        for index in range(count):
            panels = []
            if visible is not None:
                panels.append(visible[index])
            panels.append(reconstructed[index])
            panels.append(target[index])
            grid = torch.cat(
                [panel.detach().float().clamp(0, 1).cpu() for panel in panels], dim=2
            )
            experiment.add_image(
                f"val/{dataloader_idx}/sample_{index}",
                grid,
                global_step=self.global_step,
                dataformats="CHW",
            )

    def on_validation_epoch_end(self) -> None:
        for index, metrics in enumerate(self._validation_metrics):
            result = metrics.compute()
            for name, value in result.items():
                self.log(
                    f"val/{index}/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=name in {"psnr", "fid_clean"},
                    sync_dist=False,
                    batch_size=1,
                )
        self._validation_epoch_started = False

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["physical_stage"] = self.stage
        if self.model.dit is not None:
            checkpoint["model_config"] = self.model.dit.config.__dict__
        if self._ema is not None:
            checkpoint["dit_ema"] = self._ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        saved_stage = checkpoint.get("physical_stage")
        if saved_stage is not None and saved_stage != self.stage:
            raise ValueError(
                f"Checkpoint stage {saved_stage!r} does not match requested stage {self.stage!r}."
            )
        ema_state = checkpoint.get("dit_ema")
        if ema_state is not None:
            if self.model.dit is None:
                raise RuntimeError("Cannot load DiT EMA into a model without a DiT.")
            if self._ema is None:
                self._ema = ExponentialMovingAverage(self.model.dit.parameters())
            self._ema.load_state_dict(ema_state)


__all__ = ["PhysicalLightningModule"]
