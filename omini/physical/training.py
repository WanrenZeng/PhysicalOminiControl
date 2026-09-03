from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .flux_dit import PhysicalFluxDiT, PhysicalFluxDiTConfig
from .latent import pack_flux_latents, prepare_flux_ids, unpack_flux_latents
from .reconstruction import HFRMReconstructor
from .vae import FluxVAE
from .wavelet import DMT


def data_transform(images: torch.Tensor) -> torch.Tensor:
    return images.mul(2).sub(1)


def inverse_data_transform(images: torch.Tensor) -> torch.Tensor:
    return images.add(1).div(2).clamp(0, 1)


def ll_preprocess(coefficients: torch.Tensor) -> torch.Tensor:
    return coefficients.add(2).div(4)


def ll_postprocess(images: torch.Tensor) -> torch.Tensor:
    return images.mul(4).sub(2)


def logistic_normal_timesteps(
    batch_size: int, *, device: torch.device, generator: torch.Generator | None = None
) -> torch.Tensor:
    return torch.sigmoid(torch.randn(batch_size, device=device, generator=generator))


def build_optimizer(parameters: Any, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    optimizer_type = str(config["type"]).lower()
    params = dict(config.get("params", {}))
    if optimizer_type == "adamw":
        return torch.optim.AdamW(parameters, **params)
    if optimizer_type == "adam":
        return torch.optim.Adam(parameters, **params)
    if optimizer_type == "prodigy":
        try:
            import prodigyopt
        except ImportError as error:
            raise ImportError("Prodigy optimizer requires the `prodigyopt` package.") from error
        return prodigyopt.Prodigy(parameters, **params)
    raise ValueError(f"Unsupported optimizer type: {config['type']!r}.")


class PhysicalTrainingModel(nn.Module):
    """Full text-free visible-to-infrared training model.

    The frozen FLUX VAE encodes wavelet-domain LL RGB images. ``PhysicalFluxDiT``
    learns flow matching for IR-LL VAE tokens. ``HFRMReconstructor`` can either
    train independently with ground-truth IR LL, or reconstruct full IR from
    predicted IR LL for end-to-end validation and fine tuning.
    """

    def __init__(
        self,
        *,
        vae_path: str | Path,
        dit_config: PhysicalFluxDiTConfig | None = None,
        hfrm_base_dim: int = 96,
        vae_sample_posterior: bool = True,
        include_dit: bool = True,
    ) -> None:
        super().__init__()
        self.dmt = DMT()
        self.dit = PhysicalFluxDiT(dit_config) if include_dit else None
        self.reconstructor = HFRMReconstructor(base_dim=hfrm_base_dim)
        self.vae = FluxVAE(vae_path) if include_dit else None
        self.vae_sample_posterior = vae_sample_posterior

        if self.dit is not None and self.vae is not None and self.vae.latent_channels * 4 != self.dit.config.in_channels:
            raise ValueError(
                "VAE packed channels do not match DiT input channels: "
                f"{self.vae.latent_channels * 4} != {self.dit.config.in_channels}."
            )

    @property
    def hfrm(self) -> nn.Module:
        return self.reconstructor.hfrm

    def set_vae_device(self, device: torch.device | str) -> None:
        if self.vae is not None:
            self.vae.to(device)
            self.vae.eval()

    def enable_gradient_checkpointing(self) -> None:
        if self.dit is None:
            raise RuntimeError("Gradient checkpointing is only available when the DiT is enabled.")
        self.dit.enable_gradient_checkpointing()

    def load_hfrm_checkpoint(self, checkpoint_path: str | Path, *, strict: bool = True) -> None:
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(state_dict, dict):
            raise ValueError("HFRM checkpoint must contain a state dictionary.")
        self.hfrm.load_state_dict(state_dict, strict=strict)

    def set_train_stage(self, stage: str) -> None:
        stage = stage.lower()
        if stage not in {"dit", "hfrm", "joint"}:
            raise ValueError("stage must be 'dit', 'hfrm', or 'joint'.")
        if self.dit is None and stage in {"dit", "joint"}:
            raise RuntimeError("The DiT stage requires include_dit=True.")
        if self.dit is not None:
            self.dit.requires_grad_(stage in {"dit", "joint"})
        self.hfrm.requires_grad_(stage in {"hfrm", "joint"})

    def _wavelet_components(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visible = data_transform(batch["visible"])
        infrared = data_transform(batch["infrared"])
        visible_coefficients = self.dmt(visible)
        infrared_coefficients = self.dmt(infrared)
        return {
            "visible_ll": visible_coefficients[:, :3],
            "visible_high": visible_coefficients[:, 3:],
            "infrared_ll": infrared_coefficients[:, :3],
            "infrared_high": infrared_coefficients[:, 3:],
            "infrared": infrared,
        }

    @torch.no_grad()
    def encode_ll_pair(
        self, visible_ll: torch.Tensor, infrared_ll: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        if self.vae is None:
            raise RuntimeError("LL VAE encoding requires include_dit=True.")
        visible_ll_images = ll_preprocess(visible_ll)
        infrared_ll_images = ll_preprocess(infrared_ll)
        visible_latents = self.vae.encode(
            visible_ll_images, sample_posterior=self.vae_sample_posterior
        )
        infrared_latents = self.vae.encode(
            infrared_ll_images, sample_posterior=self.vae_sample_posterior
        )
        if visible_latents.shape != infrared_latents.shape:
            raise ValueError("Visible and infrared LL VAE latent shapes must match.")
        return (
            pack_flux_latents(visible_latents),
            pack_flux_latents(infrared_latents),
            tuple(infrared_latents.shape[-2:]),
        )

    def flow_matching_loss(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = self._wavelet_components(batch)
        visible_tokens, infrared_tokens, latent_shape = self.encode_ll_pair(
            components["visible_ll"], components["infrared_ll"]
        )
        if self.dit is None:
            raise RuntimeError("Flow matching requires include_dit=True.")
        batch_size = infrared_tokens.shape[0]
        times = logistic_normal_timesteps(batch_size, device=infrared_tokens.device)
        noise = torch.randn_like(infrared_tokens)
        noisy_tokens = (1 - times[:, None, None]) * infrared_tokens + times[:, None, None] * noise
        target_velocity = noise - infrared_tokens
        image_ids = prepare_flux_ids(*latent_shape, device=infrared_tokens.device)
        predicted_velocity = self.dit(
            noisy_tokens,
            visible_tokens,
            times,
            image_ids=image_ids,
        )
        loss = F.mse_loss(predicted_velocity.float(), target_velocity.float())
        return loss, {
            "flow_loss": loss.detach(),
            "time_mean": times.detach().mean(),
        }

    def hfrm_loss(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        high_frequency_weight: float = 0.5,
        pixel_weight: float = 1.0,
        ms_ssim_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = self._wavelet_components(batch)
        predicted_high = self.hfrm(components["infrared_ll"], components["visible_high"])
        reconstructed = self.reconstructor.idmt(
            torch.cat((components["infrared_ll"], predicted_high), dim=1)
        )
        high_loss = F.l1_loss(predicted_high.float(), components["infrared_high"].float())
        pixel_loss = F.l1_loss(reconstructed.float(), components["infrared"].float())
        ms_ssim_loss = reconstructed.new_zeros(())
        if ms_ssim_weight:
            try:
                from pytorch_msssim import ms_ssim
            except ImportError as error:
                raise ImportError(
                    "MS-SSIM loss requires `pytorch-msssim`; install it before enabling ms_ssim_weight."
                ) from error
            ms_ssim_loss = 1 - ms_ssim(
                inverse_data_transform(reconstructed).float(),
                inverse_data_transform(components["infrared"]).float(),
                data_range=1.0,
                size_average=True,
            )
        loss = (
            high_frequency_weight * high_loss
            + pixel_weight * pixel_loss
            + ms_ssim_weight * ms_ssim_loss
        )
        return loss, {
            "hfrm_loss": loss.detach(),
            "hfrm_high_loss": high_loss.detach(),
            "hfrm_pixel_loss": pixel_loss.detach(),
            "hfrm_ms_ssim_loss": ms_ssim_loss.detach(),
        }

    @torch.no_grad()
    def encode_visible_ll(self, visible: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        if self.dit is None:
            raise RuntimeError("Visible LL encoding requires include_dit=True.")
        visible_coefficients = self.dmt(data_transform(visible))
        visible_ll = visible_coefficients[:, :3]
        visible_high = visible_coefficients[:, 3:]
        if self.vae is None:
            raise RuntimeError("Visible LL encoding requires include_dit=True.")
        visible_latents = self.vae.encode(
            ll_preprocess(visible_ll), sample_posterior=False
        )
        dtype = next(self.dit.parameters()).dtype
        return (
            pack_flux_latents(visible_latents).to(dtype=dtype),
            visible_high,
            tuple(visible_latents.shape[-2:]),
        )

    @torch.no_grad()
    def sample_infrared(
        self,
        visible: torch.Tensor,
        *,
        num_inference_steps: int = 28,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dit is None:
            raise RuntimeError("Infrared sampling requires include_dit=True.")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler
        except ImportError as error:
            raise ImportError("Infrared sampling requires diffusers.") from error

        visible_tokens, visible_high, latent_shape = self.encode_visible_ll(visible)
        sample = torch.randn(
            visible_tokens.shape,
            device=visible_tokens.device,
            dtype=visible_tokens.dtype,
            generator=generator,
        )
        scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
        scheduler.set_timesteps(num_inference_steps, device=visible_tokens.device)
        for timestep in scheduler.timesteps:
            normalized_timestep = torch.full(
                (sample.shape[0],),
                timestep / scheduler.config.num_train_timesteps,
                device=sample.device,
                dtype=sample.dtype,
            )
            velocity = self.predict_velocity(
                sample, visible_tokens, normalized_timestep, latent_shape
            )
            sample = scheduler.step(velocity, timestep, sample, return_dict=False)[0]

        predicted_ll = self.decode_ir_ll_tokens(sample, latent_shape)
        reconstructed, predicted_high = self.reconstruct_with_ll(predicted_ll, visible_high)
        return reconstructed, predicted_ll, predicted_high

    @torch.no_grad()
    def decode_ir_ll_tokens(
        self, tokens: torch.Tensor, latent_shape: tuple[int, int]
    ) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("LL VAE decoding requires include_dit=True.")
        latents = unpack_flux_latents(tokens, *latent_shape)
        ll_images = self.vae.decode(latents)
        return ll_postprocess(ll_images)

    @torch.no_grad()
    def predict_velocity(
        self,
        noisy_ir_tokens: torch.Tensor,
        visible_ll_tokens: torch.Tensor,
        timestep: float | torch.Tensor,
        latent_shape: tuple[int, int],
    ) -> torch.Tensor:
        if self.dit is None:
            raise RuntimeError("Velocity prediction requires include_dit=True.")
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.full(
                (noisy_ir_tokens.shape[0],), timestep, device=noisy_ir_tokens.device
            )
        image_ids = prepare_flux_ids(*latent_shape, device=noisy_ir_tokens.device)
        return self.dit(noisy_ir_tokens, visible_ll_tokens, timestep, image_ids=image_ids)

    @torch.no_grad()
    def reconstruct_with_ll(
        self, predicted_infrared_ll: torch.Tensor, visible_high: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reconstructed, predicted_high = self.reconstructor(predicted_infrared_ll, visible_high)
        return inverse_data_transform(reconstructed), predicted_high


__all__ = [
    "data_transform",
    "inverse_data_transform",
    "ll_preprocess",
    "ll_postprocess",
    "logistic_normal_timesteps",
    "build_optimizer",
    "PhysicalTrainingModel",
]
