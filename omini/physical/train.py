from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from .datamodule import PhysicalDataModule
from .flux_dit import PhysicalFluxDiTConfig
from .training import PhysicalTrainingModel
from .trainer import PhysicalLightningModule
from .validation import ValidationMetrics


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping.")
    return config


def _run_directory(config: dict[str, Any]) -> Path:
    output_root = Path(config["train"]["output_dir"])
    run_name = config["train"].get("run_name") or time.strftime("%Y%m%d-%H%M%S")
    return output_root / run_name


def _build_model(config: dict[str, Any]) -> PhysicalTrainingModel:
    stage = str(config["train"]["stage"]).lower()
    model_config = config["model"]
    hfrm_config = config.get("hfrm", {})
    include_dit = stage == "dit"
    dit_config = PhysicalFluxDiTConfig(**model_config) if include_dit else None
    model = PhysicalTrainingModel(
        vae_path=config["vae"]["path"],
        dit_config=dit_config,
        hfrm_base_dim=int(hfrm_config.get("base_dim", 96)),
        vae_sample_posterior=bool(config["vae"].get("sample_posterior", True)),
        include_dit=include_dit,
    )
    pretrained_hfrm = hfrm_config.get("pretrained_path")
    if pretrained_hfrm:
        model.load_hfrm_checkpoint(pretrained_hfrm)
    if config["train"].get("gradient_checkpointing", False) and include_dit:
        model.enable_gradient_checkpointing()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    run_directory = _run_directory(config)
    train_config = config["train"]
    stage = str(train_config["stage"]).lower()

    dataset_config = dict(config["dataset"])
    dataset_config["enable_validation"] = bool(config.get("validation", {}).get("enabled", True))
    datamodule = PhysicalDataModule(
        dataset_config,
        train_batch_size=int(train_config["batch_size"]),
        validation_batch_size=int(train_config.get("validation_batch_size", 1)),
        num_workers=int(train_config.get("num_workers", 4)),
        pin_memory=bool(train_config.get("pin_memory", True)),
    )
    model = _build_model(config)
    lightning_module = PhysicalLightningModule(
        model,
        stage=stage,
        optimizer_config=train_config["optimizer"],
        ema_config=config.get("ema"),
        hfrm_loss_config=config.get("hfrm_loss"),
        validation_config=config.get("validation"),
    )

    validation_config = config.get("validation", {})
    validation_enabled = bool(validation_config.get("enabled", True))
    if validation_enabled:
        validation_directory = run_directory / "validation"
        validation_metrics = [
            ValidationMetrics(
                validation_directory / datamodule.dataset_name,
                "cuda" if torch.cuda.is_available() else "cpu",
                save_images=bool(validation_config.get("save_images", True)),
                compute_fid=bool(validation_config.get("compute_fid", True)),
                fid_reference=str(validation_config.get("fid_reference", "source")),
                fid_num_workers=int(validation_config.get("fid_num_workers", 4)),
                fid_batch_size=int(validation_config.get("fid_batch_size", 8)),
                keep_fid_images=bool(validation_config.get("keep_fid_images", False)),
            )
        ]
        lightning_module.set_validation_metrics(validation_metrics)

    callbacks = [
        ModelCheckpoint(
            dirpath=run_directory / "checkpoints",
            filename="{step}",
            every_n_train_steps=int(train_config.get("checkpoint_every_n_steps", 1000)),
            save_top_k=-1,
            save_last=True,
            save_on_train_epoch_end=False,
        )
    ]
    strategy = "ddp" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "auto"

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=int(train_config.get("devices", 1)),
        strategy=strategy,
        precision=train_config.get("precision", "bf16-mixed"),
        max_steps=int(train_config["max_steps"]),
        accumulate_grad_batches=int(train_config.get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(train_config.get("gradient_clip_val", 1.0)),
        check_val_every_n_epoch=None if validation_enabled else 1,
        val_check_interval=int(train_config.get("val_check_interval", 1000)) if validation_enabled else None,
        limit_val_batches=1.0 if validation_enabled else 0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=True,
        callbacks=callbacks,
        use_distributed_sampler=False,
        log_every_n_steps=int(train_config.get("log_every_n_steps", 10)),
    )

    if trainer.is_global_zero:
        run_directory.mkdir(parents=True, exist_ok=True)
        with (run_directory / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
