"""Standalone paired visible-to-infrared datasets.

All datasets use the same JSON record schema inherited from the local data
preparation pipeline::

    {
        "vision_path": "visible/example.png",
        "infrared_path": "lwir/example.png"
    }

Optional caption fields in the source annotations are deliberately ignored:
PhysicalOminiControl is a text-free visible-to-infrared model.  Samples return
``visible`` and ``infrared`` RGB tensors in ``[0, 1]`` with shape ``(3, H, W)``.
Convert them to ``[-1, 1]`` in the training step before applying DMT.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


ImageSize: TypeAlias = tuple[int, int]  # (width, height)
AnnotationSource: TypeAlias = str | Path | Sequence[Mapping[str, Any]]


DATASET_SIZES: dict[str, ImageSize] = {
    "flir": (640, 512),
    "kaist": (640, 512),
    "m3fd": (512, 384),
}
"""Native training resolutions in ``(width, height)`` order."""


def _load_records(annotations: AnnotationSource) -> list[dict[str, Any]]:
    """Load records from a JSON annotation file or an in-memory sequence."""
    if isinstance(annotations, (str, Path)):
        annotation_path = Path(annotations)
        with annotation_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    else:
        loaded = list(annotations)

    if not isinstance(loaded, list):
        raise ValueError(
            "Annotations must be a JSON list or a sequence of record mappings."
        )
    if not all(isinstance(record, Mapping) for record in loaded):
        raise ValueError("Every annotation record must be a mapping.")

    return [dict(record) for record in loaded]


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to a contiguous ``float32`` tensor in ``[0, 1]``."""
    array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).to(torch.float32).div_(255)


class PairedVisibleInfraredDataset(Dataset[dict[str, Any]]):
    """Paired visible/infrared dataset with spatially consistent preprocessing.

    Args:
        annotations: A path to a JSON list, or records containing ``vision_path``
            and ``infrared_path``. Relative paths are resolved against
            ``dataset_root``.
        dataset_root: Directory containing the paths in the annotations.
        image_size: Output resolution in ``(width, height)`` order. Both images
            are independently resized to this common size without cropping, so
            their pixel alignment is preserved.
        dataset_name: Metadata label returned per sample.
        horizontal_flip_prob: Probability of applying the same horizontal flip
            to both modalities. Keep it at zero for deterministic evaluation.
        return_paths: Return absolute source paths and a sample identifier for
            validation and result saving.

    The configured size must be divisible by 16. This guarantees compatibility
    with the DMT, the two-downsampling HFRMNet, and a VAE with 16x image
    downsampling.
    """

    required_fields = ("vision_path", "infrared_path")

    def __init__(
        self,
        annotations: AnnotationSource,
        dataset_root: str | Path,
        image_size: ImageSize,
        dataset_name: str,
        horizontal_flip_prob: float = 0.0,
        return_paths: bool = True,
    ) -> None:
        self.records = _load_records(annotations)
        self.dataset_root = Path(dataset_root)
        self.image_size = self._validate_image_size(image_size)
        self.dataset_name = dataset_name
        self.horizontal_flip_prob = self._validate_probability(horizontal_flip_prob)
        self.return_paths = return_paths

        for index, record in enumerate(self.records):
            missing = [field for field in self.required_fields if not record.get(field)]
            if missing:
                raise ValueError(
                    f"Annotation record {index} is missing required fields: {missing}."
                )

    @staticmethod
    def _validate_image_size(image_size: ImageSize) -> ImageSize:
        if len(image_size) != 2:
            raise ValueError("image_size must be a (width, height) pair.")
        width, height = int(image_size[0]), int(image_size[1])
        if width <= 0 or height <= 0:
            raise ValueError("image_size values must be positive.")
        if width % 16 or height % 16:
            raise ValueError(
                "image_size must be divisible by 16 for the DMT/VAE/HFRM pipeline, "
                f"but got {(width, height)}."
            )
        return width, height

    @staticmethod
    def _validate_probability(value: float) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("horizontal_flip_prob must be in [0, 1].")
        return value

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.dataset_root / path

    @staticmethod
    def _read_rgb(path: Path) -> Image.Image:
        if not path.is_file():
            raise FileNotFoundError(f"Paired image file does not exist: {path}")
        with Image.open(path) as image:
            return image.convert("RGB")

    def _transform_pair(
        self, visible: Image.Image, infrared: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        visible = visible.resize(self.image_size, resample=Image.Resampling.BICUBIC)
        infrared = infrared.resize(self.image_size, resample=Image.Resampling.BICUBIC)

        if random.random() < self.horizontal_flip_prob:
            visible = visible.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            infrared = infrared.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return visible, infrared

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        vision_path = self._resolve_path(record["vision_path"])
        infrared_path = self._resolve_path(record["infrared_path"])

        visible, infrared = self._transform_pair(
            self._read_rgb(vision_path), self._read_rgb(infrared_path)
        )
        sample: dict[str, Any] = {
            "visible": _to_tensor(visible),
            "infrared": _to_tensor(infrared),
            "dataset_name": self.dataset_name,
            "sample_id": str(record.get("filename", infrared_path.stem)),
        }
        if self.return_paths:
            sample["vision_path"] = str(vision_path)
            sample["infrared_path"] = str(infrared_path)
        return sample


class FLIRDataset(PairedVisibleInfraredDataset):
    """FLIR aligned visible-to-thermal pairs at 640x512 (width x height)."""

    def __init__(
        self,
        annotations: AnnotationSource,
        dataset_root: str | Path,
        image_size: ImageSize = DATASET_SIZES["flir"],
        horizontal_flip_prob: float = 0.0,
        return_paths: bool = True,
    ) -> None:
        super().__init__(
            annotations=annotations,
            dataset_root=dataset_root,
            image_size=image_size,
            dataset_name="flir",
            horizontal_flip_prob=horizontal_flip_prob,
            return_paths=return_paths,
        )


class KAISTDataset(PairedVisibleInfraredDataset):
    """KAIST visible/LWIR pairs at 640x512 (width x height)."""

    def __init__(
        self,
        annotations: AnnotationSource,
        dataset_root: str | Path,
        image_size: ImageSize = DATASET_SIZES["kaist"],
        horizontal_flip_prob: float = 0.0,
        return_paths: bool = True,
    ) -> None:
        super().__init__(
            annotations=annotations,
            dataset_root=dataset_root,
            image_size=image_size,
            dataset_name="kaist",
            horizontal_flip_prob=horizontal_flip_prob,
            return_paths=return_paths,
        )


class M3FDDataset(PairedVisibleInfraredDataset):
    """M3FD visible/infrared pairs at 512x384 (width x height)."""

    def __init__(
        self,
        annotations: AnnotationSource,
        dataset_root: str | Path,
        image_size: ImageSize = DATASET_SIZES["m3fd"],
        horizontal_flip_prob: float = 0.0,
        return_paths: bool = True,
    ) -> None:
        super().__init__(
            annotations=annotations,
            dataset_root=dataset_root,
            image_size=image_size,
            dataset_name="m3fd",
            horizontal_flip_prob=horizontal_flip_prob,
            return_paths=return_paths,
        )


def build_dataset(
    dataset_name: str,
    annotations: AnnotationSource,
    dataset_root: str | Path,
    **kwargs: Any,
) -> PairedVisibleInfraredDataset:
    """Create one of the three standard datasets by name.

    Valid names are ``"flir"``, ``"kaist"``, and ``"m3fd"``.
    """
    dataset_classes: dict[str, type[PairedVisibleInfraredDataset]] = {
        "flir": FLIRDataset,
        "kaist": KAISTDataset,
        "m3fd": M3FDDataset,
    }
    normalized_name = dataset_name.lower()
    try:
        dataset_class = dataset_classes[normalized_name]
    except KeyError as error:
        valid_names = ", ".join(dataset_classes)
        raise ValueError(f"Unknown dataset {dataset_name!r}; choose one of: {valid_names}.") from error
    return dataset_class(annotations, dataset_root, **kwargs)


__all__ = [
    "DATASET_SIZES",
    "PairedVisibleInfraredDataset",
    "FLIRDataset",
    "KAISTDataset",
    "M3FDDataset",
    "build_dataset",
]
