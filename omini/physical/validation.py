"""DDP-safe validation metrics for visible-to-infrared reconstruction.

Metric settings intentionally match OminiControl's offline evaluator
``evaluation_pretrain_ll/evaluation_lpips_correct.py``:

* PSNR: per-image PSNR over RGB tensors in ``[0, 1]``;
* SSIM: legacy ``torchmetrics.image.ssim.SSIM(data_range=1.0)`` settings
  (11x11 Gaussian window, sigma=1.5);
* LPIPS: ``lpips.LPIPS(net="alex")`` on RGB inputs mapped to ``[-1, 1]``;
* FID: CleanFID ``mode="clean"``, ``model_name="inception_v3"``, with
  ``batch_size=8`` and ``num_workers=4``.

PSNR, SSIM, and LPIPS are accumulated as scalar sums/counts and reduced with
``torch.distributed.all_reduce``. FID is deliberately computed only on rank
zero after every rank writes its assigned generated/reference images to a
shared validation directory. This avoids averaging invalid rank-local FID
values and never gathers image tensors between GPUs.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler


METRIC_NAMES = ("psnr", "ssim", "lpips", "fid_clean")


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _barrier() -> None:
    if _is_distributed():
        dist.barrier()


def _safe_sample_id(sample_id: str) -> str:
    """Create a flat, filename-safe ID while preserving readable sample names."""
    safe_id = str(sample_id).replace("/", "_").replace("\\", "_")
    if not safe_id or safe_id in {".", ".."}:
        raise ValueError(f"Invalid validation sample ID: {sample_id!r}")
    return safe_id


def _to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Convert RGB ``[0, 1]`` batches to CPU ``uint8`` images."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            "Expected image tensor with shape (batch, 3, height, width), "
            f"got {tuple(images.shape)}."
        )
    return images.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu()


def _save_tensor_batch(images: torch.Tensor, directory: Path, disk_ids: Sequence[str]) -> None:
    """Write an RGB batch as PNGs, refusing accidental overwrites."""
    if images.shape[0] != len(disk_ids):
        raise ValueError("The number of images must equal the number of disk IDs.")

    directory.mkdir(parents=True, exist_ok=True)
    for image, disk_id in zip(_to_uint8(images), disk_ids):
        path = directory / f"{disk_id}.png"
        if path.exists():
            raise FileExistsError(
                f"Duplicate validation sample ID would overwrite {path}. "
                "Use unique annotation filenames and the distributed validation sampler."
            )
        Image.fromarray(image.permute(1, 2, 0).numpy(), mode="RGB").save(path)


class DistributedEvaluationSampler(Sampler[int]):
    """Shard evaluation data without padding or duplicate samples.

    PyTorch's default ``DistributedSampler(drop_last=False)`` pads shards so
    every rank has the same length. That repeats some validation examples and
    biases PSNR/SSIM/LPIPS as well as FID. This sampler assigns indices by
    stride instead: rank ``r`` receives ``r, r + world_size, ...``.
    """

    def __init__(
        self,
        dataset: Dataset[object],
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if _is_distributed() else 1
        if rank is None:
            rank = _rank()
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid distributed evaluation sampler rank/world size.")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        return max(0, (len(self.dataset) - self.rank + self.num_replicas - 1) // self.num_replicas)


def build_validation_dataloader(
    dataset: Dataset[object],
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    num_replicas: int | None = None,
    rank: int | None = None,
) -> DataLoader[object]:
    """Create a deterministic validation loader without DDP sample padding.

    Each dataset needs its own loader because FLIR/KAIST and M3FD use different
    spatial sizes. Lightning should receive the returned loaders as a list.
    """
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0.")

    use_distributed_sampler = _is_distributed() or num_replicas is not None or rank is not None
    sampler: Sampler[int] | None = (
        DistributedEvaluationSampler(dataset, num_replicas=num_replicas, rank=rank)
        if use_distributed_sampler
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def _copy_source_batch(
    source_paths: Sequence[str | Path], directory: Path, disk_ids: Sequence[str]
) -> None:
    """Copy raw source IR files exactly as OminiControl's FID evaluator does."""
    if len(source_paths) != len(disk_ids):
        raise ValueError("The number of infrared source paths must equal the batch size.")

    directory.mkdir(parents=True, exist_ok=True)
    for source_path, disk_id in zip(source_paths, disk_ids):
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Infrared reference image does not exist: {source}")
        suffix = source.suffix.lower() or ".png"
        destination = directory / f"{disk_id}{suffix}"
        if destination.exists():
            raise FileExistsError(
                f"Duplicate validation sample ID would overwrite {destination}. "
                "Use unique annotation filenames and the distributed validation sampler."
            )
        shutil.copy2(source, destination)


def _save_source_batch_resized(
    source_paths: Sequence[str | Path], directory: Path, disk_ids: Sequence[str], image_size: tuple[int, int]
) -> None:
    """Write raw IR references after the same RGB/resize preprocessing as the model."""
    if len(source_paths) != len(disk_ids):
        raise ValueError("The number of infrared source paths must equal the batch size.")

    directory.mkdir(parents=True, exist_ok=True)
    for source_path, disk_id in zip(source_paths, disk_ids):
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Infrared reference image does not exist: {source}")
        destination = directory / f"{disk_id}.png"
        if destination.exists():
            raise FileExistsError(
                f"Duplicate validation sample ID would overwrite {destination}. "
                "Use unique annotation filenames and the distributed validation sampler."
            )
        with Image.open(source) as image:
            processed = image.convert("RGB").resize(image_size, resample=Image.Resampling.BICUBIC)
            processed.save(destination)


class ValidationMetrics:
    """Streaming paired metrics and CleanFID that work in single-GPU and DDP.

    Args:
        output_dir: Root directory for per-epoch validation files. With DDP it
            must be on storage visible to all ranks.
        device: Device used for LPIPS, SSIM, and distributed scalar reductions.
        save_images: Persist generated/reference files. It must be true for FID.
        compute_fid: Calculate CleanFID at epoch end on global rank zero.
        fid_reference: ``"source"`` copies raw infrared files from
            ``target_source_paths`` and exactly matches OminiControl's offline
            FID reference preparation. ``"source_resized"`` applies the dataset's
            RGB + bicubic resize preprocessing to those source files before FID;
            this is the recommended protocol for M3FD, whose native IR files are
            resized to 512x384 for model input. ``"processed"`` writes the model
            target tensors instead.
        fid_num_workers: CleanFID worker count; default matches OminiControl.
        fid_batch_size: CleanFID batch size; default matches OminiControl.
        keep_fid_images: Keep generated/reference files after FID. Set false to
            clean them after metrics are computed.

    A separate instance should be created for each validation dataset (FLIR,
    KAIST, M3FD), because each has a different resolution and FID distribution.
    """

    def __init__(
        self,
        output_dir: str | Path,
        device: torch.device | str,
        *,
        save_images: bool = True,
        compute_fid: bool = True,
        fid_reference: str = "source",
        fid_num_workers: int = 4,
        fid_batch_size: int = 8,
        keep_fid_images: bool = False,
    ) -> None:
        if compute_fid and not save_images:
            raise ValueError("compute_fid=True requires save_images=True.")
        if fid_reference not in {"source", "source_resized", "processed"}:
            raise ValueError(
                "fid_reference must be 'source', 'source_resized', or 'processed'."
            )
        if fid_num_workers < 0 or fid_batch_size <= 0:
            raise ValueError("fid_num_workers must be non-negative and fid_batch_size positive.")

        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self.save_images = save_images
        self.compute_fid = compute_fid
        self.fid_reference = fid_reference
        self.fid_num_workers = fid_num_workers
        self.fid_batch_size = fid_batch_size
        self.keep_fid_images = keep_fid_images
        self._lpips_model: torch.nn.Module | None = None
        self._ssim_metric: torch.nn.Module | None = None
        self._epoch_dir: Path | None = None
        self.reset()

    @property
    def is_global_zero(self) -> bool:
        return _rank() == 0

    @property
    def epoch_dir(self) -> Path:
        if self._epoch_dir is None:
            raise RuntimeError("Call begin_epoch() before updating validation metrics.")
        return self._epoch_dir

    @property
    def prediction_dir(self) -> Path:
        return self.epoch_dir / "predictions"

    @property
    def target_dir(self) -> Path:
        return self.epoch_dir / "targets"

    def _lpips(self) -> torch.nn.Module:
        if self._lpips_model is None:
            try:
                import lpips
            except ImportError as error:
                raise ImportError(
                    "LPIPS validation requires the `lpips` package. "
                    "Use the cogvideo environment or install it explicitly."
                ) from error
            self._lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()
            self._lpips_model.requires_grad_(False)
        return self._lpips_model

    def _ssim(self) -> torch.nn.Module:
        if self._ssim_metric is None:
            try:
                # This exact legacy class and default parameters are used by the
                # existing OminiControl offline evaluator.
                from torchmetrics.image.ssim import SSIM
            except ImportError as error:
                raise ImportError(
                    "SSIM validation requires `torchmetrics.image.ssim.SSIM`."
                ) from error
            self._ssim_metric = SSIM(data_range=1.0).to(self.device).eval()
        return self._ssim_metric

    def set_device(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        if self._lpips_model is not None:
            self._lpips_model.to(self.device)
        if self._ssim_metric is not None:
            self._ssim_metric.to(self.device)
        self.reset()

    def reset(self) -> None:
        self._sums = torch.zeros(3, dtype=torch.float64, device=self.device)
        self._count = torch.zeros(1, dtype=torch.float64, device=self.device)
        if self._ssim_metric is not None:
            self._ssim_metric.reset()

    def begin_epoch(self, epoch: int | str) -> None:
        """Reset state and prepare an epoch directory without rank races."""
        self.reset()
        self._epoch_dir = self.output_dir / f"epoch_{epoch}"

        if self.is_global_zero:
            shutil.rmtree(self._epoch_dir, ignore_errors=True)
        _barrier()

        if self.save_images:
            self.prediction_dir.mkdir(parents=True, exist_ok=True)
            self.target_dir.mkdir(parents=True, exist_ok=True)
        _barrier()

    @torch.inference_mode()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_ids: Sequence[str],
        *,
        target_source_paths: Sequence[str | Path] | None = None,
    ) -> None:
        """Accumulate metrics for an aligned generated infrared batch.

        Inputs must be RGB tensors in ``[0, 1]``. For ``fid_reference`` values
        ``'source'`` and ``'source_resized'``, provide raw paths from
        ``batch['infrared_path']``. The latter resizes each raw reference to the
        model target tensor's spatial size before it is written for CleanFID.
        """
        if self._epoch_dir is None:
            raise RuntimeError("Call begin_epoch() before update().")
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have equal shapes, got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError("prediction and target must have shape (batch, 3, H, W).")
        if prediction.shape[0] != len(sample_ids):
            raise ValueError("The number of sample IDs must equal the batch size.")

        prediction = prediction.detach().to(self.device, dtype=torch.float32).clamp(0, 1)
        target = target.detach().to(self.device, dtype=torch.float32).clamp(0, 1)
        batch_size = prediction.shape[0]

        # OminiControl computes PSNR independently for every batch_size==1
        # sample, then averages. This batched expression is mathematically
        # identical while avoiding Python loops.
        mse = (prediction - target).square().flatten(1).mean(dim=1)
        psnr = torch.where(
            mse == 0,
            torch.full_like(mse, float("inf")),
            20 * torch.log10(torch.reciprocal(torch.sqrt(mse))),
        )

        # Legacy TorchMetrics SSIM retains full inputs in its internal state.
        # Resetting after each forward keeps memory O(batch), while retaining
        # the exact OminiControl class and default kernel/data_range settings.
        ssim_metric = self._ssim()
        ssim = ssim_metric(prediction, target)
        ssim_metric.reset()

        # This matches OminiControl's `real * 2 - 1`, `generated * 2 - 1`
        # invocation of LPIPS(AlexNet).
        lpips_value = self._lpips()(prediction * 2 - 1, target * 2 - 1)
        lpips_sum = lpips_value.reshape(batch_size, -1).mean(dim=1).sum()

        self._sums += torch.stack((psnr.sum(), ssim * batch_size, lpips_sum)).to(torch.float64)
        self._count += batch_size

        if self.save_images:
            disk_ids = [f"rank{_rank():04d}_{_safe_sample_id(sample_id)}" for sample_id in sample_ids]
            _save_tensor_batch(prediction, self.prediction_dir, disk_ids)
            if self.fid_reference in {"source", "source_resized"}:
                if target_source_paths is None:
                    raise ValueError(
                        "target_source_paths is required when fid_reference uses source files."
                    )
                if self.fid_reference == "source":
                    _copy_source_batch(target_source_paths, self.target_dir, disk_ids)
                else:
                    image_size = (target.shape[-1], target.shape[-2])
                    _save_source_batch_resized(
                        target_source_paths, self.target_dir, disk_ids, image_size
                    )
            else:
                _save_tensor_batch(target, self.target_dir, disk_ids)

    def _reduce_paired_metrics(self) -> dict[str, float]:
        sums_and_count = torch.cat((self._sums, self._count))
        if _is_distributed():
            dist.all_reduce(sums_and_count, op=dist.ReduceOp.SUM)

        count = sums_and_count[-1].item()
        if count == 0:
            raise RuntimeError("Validation finished without any samples.")
        psnr_sum = sums_and_count[0].item()
        return {
            "psnr": float("inf") if math.isinf(psnr_sum) else (sums_and_count[0] / count).item(),
            "ssim": (sums_and_count[1] / count).item(),
            "lpips": (sums_and_count[2] / count).item(),
        }

    def _compute_fid_on_global_zero(self) -> float:
        try:
            from cleanfid import fid
        except ImportError as error:
            raise ImportError(
                "FID validation requires `clean-fid`. Use the cogvideo environment "
                "or install it explicitly."
            ) from error

        return float(
            fid.compute_fid(
                str(self.prediction_dir),
                str(self.target_dir),
                mode="clean",
                model_name="inception_v3",
                num_workers=self.fid_num_workers,
                batch_size=self.fid_batch_size,
                device=self.device,
                verbose=False,
            )
        )

    def _compute_and_broadcast_fid(self) -> float:
        """Run FID once while preventing rank-zero failures from deadlocking DDP."""
        fid_value = torch.full((1,), float("nan"), dtype=torch.float64, device=self.device)
        success = torch.ones(1, dtype=torch.int32, device=self.device)
        error_message = ""

        if self.is_global_zero:
            try:
                fid_value.fill_(self._compute_fid_on_global_zero())
            except Exception as error:
                success.zero_()
                error_message = f"CleanFID failed: {error}"

        if _is_distributed():
            dist.broadcast(success, src=0)
            dist.broadcast(fid_value, src=0)

        if success.item() == 0:
            raise RuntimeError(error_message or "CleanFID failed on global rank zero.")
        return fid_value.item()

    def compute(self) -> dict[str, float]:
        """Synchronize ranks and return global PSNR, SSIM, LPIPS, and FID.

        Call this exactly once on every rank in ``on_validation_epoch_end``.
        FID requires a shared filesystem because rank zero reads image files
        emitted by every rank.
        """
        result = self._reduce_paired_metrics()

        if self.compute_fid:
            _barrier()
            try:
                result["fid_clean"] = self._compute_and_broadcast_fid()
            finally:
                # All ranks wait before one rank removes shared image files.
                _barrier()
                if not self.keep_fid_images and self.is_global_zero:
                    shutil.rmtree(self.epoch_dir, ignore_errors=True)
                _barrier()
        else:
            result["fid_clean"] = float("nan")

        return result


__all__ = [
    "METRIC_NAMES",
    "DistributedEvaluationSampler",
    "build_validation_dataloader",
    "ValidationMetrics",
]
