from __future__ import annotations

from pathlib import Path

import torch


class FluxVAE(torch.nn.Module):
    def __init__(
        self,
        model_path: str | Path,
        *,
        subfolder: str | None = "vae",
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as error:
            raise ImportError("FluxVAE requires diffusers.AutoencoderKL.") from error

        source = str(model_path)
        source_path = Path(source)
        load_kwargs = {"local_files_only": local_files_only}
        if subfolder is not None and (not source_path.exists() or (source_path / subfolder).is_dir()):
            load_kwargs["subfolder"] = subfolder
        self.model = AutoencoderKL.from_pretrained(source, **load_kwargs)
        self.model.requires_grad_(False).eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def latent_channels(self) -> int:
        return int(self.model.config.latent_channels)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, *, sample_posterior: bool) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected VAE images with shape (B, 3, H, W), got {tuple(images.shape)}."
            )
        images = images.to(self.device, dtype=torch.float32).clamp(0, 1).mul(2).sub(1)
        latent_dist = self.model.encode(images).latent_dist
        latents = latent_dist.sample() if sample_posterior else latent_dist.mode()
        return (latents - self.model.config.shift_factor) * self.model.config.scaling_factor

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4 or latents.shape[1] != self.latent_channels:
            raise ValueError(
                "Expected VAE latents with shape "
                f"(B, {self.latent_channels}, H, W), got {tuple(latents.shape)}."
            )
        latents = latents.to(self.device, dtype=torch.float32)
        latents = latents / self.model.config.scaling_factor + self.model.config.shift_factor
        images = self.model.decode(latents, return_dict=False)[0]
        return images.add(1).div(2).clamp(0, 1)


__all__ = ["FluxVAE"]
