"""Standalone HFRM inference / evaluation for PhysicalOminiControl.

This is the current-repo counterpart of ``evaluation_hfrm_ll.py`` (which was
written for the old ``OminiControl-main`` layout and imports ``src.*`` modules
that no longer exist here). It reuses the public ``omini.physical`` API.

Pipeline (HFRM only, ground-truth infrared LL as input, matching the
``stage=hfrm`` training/validation path)::

    visible[0,1] --data_transform--> [-1,1] --DMT--> visible_high (9ch)
    infrared[0,1] --data_transform--> [-1,1] --DMT--> infrared_ll (3ch), infrared_high
    predicted_high = HFRMNet(infrared_ll, visible_high)
    reconstructed  = IDMT([infrared_ll, predicted_high])  -> [-1,1]
    reconstructed01 = inverse_data_transform(reconstructed) -> [0,1]

Metrics (PSNR / SSIM / LPIPS, optional CleanFID) are computed with the repo's
``ValidationMetrics`` so numbers match training-time validation. A
``visible | predicted | GT`` grid PNG and a ``metrics.txt`` are written per run.

Run with the project conda env (has lightning/torchmetrics/lpips/cleanfid)::

    /share401/zengpeiyi/miniconda3/envs/phyo/bin/python tests/evaluation_hfrm.py \
        --config train/config/physical/flir_hfrm.yaml \
        --checkpoint runs/physical/flir_hfrm/<run>/checkpoints/last.ckpt \
        --output_dir runs/physical/flir_hfrm/<run>/evaluation

The checkpoint may be a Lightning ``.ckpt`` (HFRM weights are read from the
``model.reconstructor.hfrm.*`` keys) or a raw ``hfrm.pth`` state dict.

if needs fid:
    cd /share401/zengpeiyi/work_base/PhysicalOminiControl
    /share401/zengpeiyi/miniconda3/envs/phyo/bin/python tests/evaluation_hfrm.py \
    --config train/config/physical/flir_hfrm.yaml \
    --checkpoint runs/physical/flir_hfrm/<run>/checkpoints/last.ckpt \
    --output_dir runs/physical/flir_hfrm/<run>/evaluation \
    --compute_fid          # optional
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

# Allow running this file directly from tests/ (repo root holds the omini pkg).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omini.physical import (
    DMT,
    IDMT,
    HFRMNet,
    ValidationMetrics,
    build_dataset,
)

to_pil = T.ToPILImage()


# --- normalization helpers (identical to omini.physical.training) -------------
def data_transform(images: torch.Tensor) -> torch.Tensor:
    """[0, 1] -> [-1, 1]."""
    return images.mul(2).sub(1)


def inverse_data_transform(images: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] (clamped)."""
    return images.add(1).div(2).clamp(0, 1)


# --- checkpoint loading -------------------------------------------------------
_HFRM_PREFIXES = (
    "model.reconstructor.hfrm.",
    "reconstructor.hfrm.",
    "hfrm.",
)


def load_hfrm_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Extract an ``HFRMNet`` state dict from a Lightning ckpt or a raw .pth."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a state dictionary.")

    for prefix in _HFRM_PREFIXES:
        filtered = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if filtered:
            return filtered

    # Already a bare HFRMNet state dict (e.g. legacy hfrm.pth).
    return dict(state_dict)


def build_hfrm(checkpoint_path: str | Path, base_dim: int, device: torch.device, dtype: torch.dtype) -> HFRMNet:
    hfrm = HFRMNet(in_channels_ll=3, in_channels_hf=9, base_dim=base_dim)
    state_dict = load_hfrm_state_dict(checkpoint_path)
    missing, unexpected = hfrm.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"HFRM checkpoint mismatch for {checkpoint_path}.\n"
            f"  missing keys  ({len(missing)}): {sorted(missing)[:5]}...\n"
            f"  unexpected keys ({len(unexpected)}): {sorted(unexpected)[:5]}..."
        )
    return hfrm.to(device=device, dtype=dtype).eval()


# --- dataset / config ---------------------------------------------------------
def resolve_dataset(config: Mapping[str, Any] | None, args: argparse.Namespace) -> Any:
    dataset_config = dict((config or {}).get("dataset", {}))
    name = args.dataset_name or dataset_config.get("name")
    root = args.dataset_root or dataset_config.get("root")
    if not name or not root:
        raise ValueError("Dataset name/root must come from --config or the CLI overrides.")

    split_key = "train_annotations" if args.split == "train" else "validation_annotations"
    annotations = args.annotations or dataset_config.get(split_key)
    if not annotations:
        raise ValueError(f"No annotations found for split={args.split!r} (key {split_key!r}).")

    root_path = Path(root)
    if not root_path.is_absolute():
        # Relative roots in the shipped configs are relative to the repo root's
        # parent (e.g. "../datasets/FLIR-align"); resolve against CWD as-is.
        root_path = Path(root).expanduser()
    annotations_path = Path(annotations).expanduser()

    return build_dataset(
        str(name),
        annotations_path,
        root_path,
        horizontal_flip_prob=0.0,  # deterministic evaluation
        return_paths=True,
    )


def resolve_num_samples(dataset: Any, config: Mapping[str, Any] | None, args: argparse.Namespace) -> int:
    if args.num_samples is not None:
        return min(int(args.num_samples), len(dataset))
    dataset_config = (config or {}).get("dataset", {})
    key = "train_num_samples" if args.split == "train" else "validation_num_samples"
    configured = dataset_config.get(key)
    if configured:
        return min(int(configured), len(dataset))
    return len(dataset)


# --- visualization ------------------------------------------------------------
def image_grid(images: list[Image.Image], rows: int, cols: int) -> Image.Image:
    assert len(images) == rows * cols
    width, height = images[0].size
    grid = Image.new("RGB", size=(cols * width, rows * height))
    for index, image in enumerate(images):
        grid.paste(image.convert("RGB"), box=((index % cols) * width, (index // cols) * height))
    return grid


def _psnr_uint8(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


# --- main evaluation ----------------------------------------------------------
@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    config: Mapping[str, Any] | None = None
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

    checkpoint_path = args.checkpoint or ((config or {}).get("hfrm", {}) or {}).get("pretrained_path")
    if not checkpoint_path:
        raise ValueError("Provide --checkpoint or an hfrm.pretrained_path in --config.")
    base_dim = int(args.base_dim or (config or {}).get("hfrm", {}).get("base_dim", 96))

    hfrm = build_hfrm(checkpoint_path, base_dim, device, dtype)
    dmt, idmt = DMT(), IDMT()

    dataset = resolve_dataset(config, args)
    num_samples = resolve_num_samples(dataset, config, args)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    validation_config = (config or {}).get("validation", {}) if config else {}
    output_dir = Path(args.output_dir)
    grid_dir = output_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    metrics = ValidationMetrics(
        output_dir,
        device,
        save_images=bool(args.compute_fid),
        compute_fid=bool(args.compute_fid),
        fid_reference=str(args.fid_reference),
        keep_fid_images=True,
    )
    metrics.set_device(device)
    metrics.begin_epoch("eval")

    metrics_path = output_dir / "metrics.txt"
    processed = 0
    with metrics_path.open("w", encoding="utf-8") as sink:
        sink.write("sample\tPSNR\n")
        for batch in loader:
            visible = batch["visible"].to(device)
            infrared = batch["infrared"].to(device)

            visible_coeff = dmt(data_transform(visible))
            infrared_coeff = dmt(data_transform(infrared))
            visible_high = visible_coeff[:, 3:].to(device=device, dtype=dtype)
            infrared_ll = infrared_coeff[:, :3].to(device=device, dtype=dtype)

            predicted_high = hfrm(infrared_ll, visible_high)
            reconstructed = idmt(torch.cat((infrared_ll, predicted_high), dim=1))
            reconstructed01 = inverse_data_transform(reconstructed).to(torch.float32)

            target01 = infrared.to(torch.float32)
            sample_ids = list(batch["sample_id"])
            metrics.update(
                reconstructed01,
                target01,
                sample_ids,
                target_source_paths=batch.get("infrared_path"),
            )

            # visible | predicted | GT grids + per-sample PSNR log
            pred_cpu = reconstructed01.clamp(0, 1).cpu()
            vis_cpu = visible.to(torch.float32).clamp(0, 1).cpu()
            tgt_cpu = target01.clamp(0, 1).cpu()
            for index, sample_id in enumerate(sample_ids):
                pred_np = (pred_cpu[index].mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy())
                tgt_np = (tgt_cpu[index].mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy())
                psnr = _psnr_uint8(pred_np, tgt_np)
                sink.write(f"{sample_id}\t{psnr:.2f}\n")
                if args.save_images:
                    grid = image_grid(
                        [to_pil(vis_cpu[index]), to_pil(pred_cpu[index]), to_pil(tgt_cpu[index])],
                        rows=1,
                        cols=3,
                    )
                    grid.save(grid_dir / f"{sample_id}_grid.png")
                print(f"[{processed + index + 1}/{num_samples}] {sample_id}  PSNR={psnr:.2f}")

            processed += int(visible.shape[0])
            if processed >= num_samples:
                break

    result = metrics.compute()
    summary = (
        f"Average PSNR: {result['psnr']:.2f}  "
        f"SSIM: {result['ssim']:.4f}  "
        f"LPIPS: {result['lpips']:.4f}  "
        f"FID: {result['fid_clean']:.3f}"
    )
    print(summary)
    with metrics_path.open("a", encoding="utf-8") as sink:
        sink.write(
            f"AVERAGE\tPSNR={result['psnr']:.4f}\tSSIM={result['ssim']:.4f}\t"
            f"LPIPS={result['lpips']:.4f}\tFID={result['fid_clean']:.4f}\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone HFRM inference/evaluation.")
    parser.add_argument("--config", default=None, help="Physical HFRM yaml for dataset/hfrm settings.")
    parser.add_argument("--checkpoint", default=None, help="Lightning .ckpt or raw hfrm.pth.")
    parser.add_argument("--output_dir", required=True, help="Directory for grids and metrics.txt.")
    parser.add_argument("--split", choices=("validation", "train"), default="validation")
    parser.add_argument("--dataset_name", default=None, help="Override config dataset name.")
    parser.add_argument("--dataset_root", default=None, help="Override config dataset root.")
    parser.add_argument("--annotations", default=None, help="Override annotation JSON path.")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_dim", type=int, default=None, help="HFRMNet base_dim (default from config/96).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--save_images", action="store_true", default=True, help="Save grid PNGs (default on).")
    parser.add_argument("--no_save_images", dest="save_images", action="store_false")
    parser.add_argument("--compute_fid", action="store_true", help="Also compute CleanFID (saves pred/target files).")
    parser.add_argument("--fid_reference", choices=("source", "source_resized", "processed"), default="source")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
