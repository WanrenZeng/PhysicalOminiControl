"""Compact text-free DiT derived directly from FLUX single-stream blocks.

The model keeps FLUX packed latents, AdaLN-Zero modulation, RoPE, and
``FluxSingleTransformerBlock``. Text, text projections, condition-specific
QKV projections, and ControlNet residuals are intentionally absent. A visible
LL token sequence is concatenated as an immutable spatial condition at every
block; all branches use the block's one shared Q/K/V projection set.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from .latent import prepare_flux_ids


@dataclass(frozen=True)
class PhysicalFluxDiTConfig:
    """Default configuration for the approximately 300M text-free FLUX DiT."""

    in_channels: int = 64
    out_channels: int = 64
    hidden_size: int = 768
    num_attention_heads: int = 12
    attention_head_dim: int = 64
    num_layers: int = 33
    mlp_ratio: float = 4.0
    axes_dims_rope: tuple[int, int, int] = (16, 24, 24)
    theta: int = 10000

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if self.hidden_size <= 0 or self.num_attention_heads <= 0:
            raise ValueError("hidden_size and num_attention_heads must be positive.")
        if self.hidden_size != self.num_attention_heads * self.attention_head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * attention_head_dim."
            )
        if sum(self.axes_dims_rope) != self.attention_head_dim:
            raise ValueError(
                "The sum of axes_dims_rope must equal attention_head_dim."
            )
        if self.num_layers <= 0 or self.mlp_ratio <= 0:
            raise ValueError("num_layers and mlp_ratio must be positive.")


class TimestepEmbedding(nn.Module):
    """FLUX-style sinusoidal timestep projection without text conditioning."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.linear_1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(f"Expected timesteps with shape (B,), got {tuple(timesteps.shape)}.")

        half_dimension = self.frequency_embedding_size // 2
        exponent = -torch.log(torch.tensor(10000.0, device=timesteps.device))
        exponent = exponent * torch.arange(
            half_dimension, device=timesteps.device, dtype=torch.float32
        ) / half_dimension
        frequencies = torch.exp(exponent)
        arguments = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = F.pad(embedding, (0, 1))
        embedding = embedding.to(dtype=self.linear_1.weight.dtype)
        return self.linear_2(F.silu(self.linear_1(embedding)))


class PhysicalFluxDiT(nn.Module):
    """A compact FLUX-derived visual conditional flow-matching transformer.

    Args:
        config: FLUX-compatible compact architecture. The default has roughly
            294M learnable parameters and uses 33 original FLUX single-stream
            blocks at a hidden width of 768.

    Inputs:
        noisy_ir_tokens: Packed noisy IR-LL VAE latents, ``(B, N, 64)``.
        visible_ll_tokens: Packed visible-LL VAE latents, ``(B, N, 64)``.
        timesteps: Flow-matching times in ``[0, 1]``, ``(B,)``.
        image_ids: Optional FLUX spatial IDs, ``(N, 3)``. When omitted,
            ``latent_height`` and ``latent_width`` are required to construct
            them.

    Returns:
        Velocity prediction with the same shape as ``noisy_ir_tokens``.

    The visible branch is intentionally reset to its input embedding after
    every block. It provides static spatial keys/values to each denoising
    block, so no dedicated condition QKV matrices or external ControlNet are
    introduced.
    """

    def __init__(self, config: PhysicalFluxDiTConfig | None = None):
        super().__init__()
        self.config = config or PhysicalFluxDiTConfig()

        try:
            from diffusers.models.normalization import AdaLayerNormContinuous
            from diffusers.models.transformers.transformer_flux import (
                FluxPosEmbed,
                FluxSingleTransformerBlock,
            )
        except ImportError as error:
            raise ImportError(
                "PhysicalFluxDiT requires diffusers with FLUX transformer blocks."
            ) from error

        self.x_embedder = nn.Linear(self.config.in_channels, self.config.hidden_size)
        self.condition_embedder = nn.Linear(
            self.config.in_channels, self.config.hidden_size
        )
        self.token_type_embedding = nn.Parameter(torch.zeros(2, self.config.hidden_size))
        self.time_embed = TimestepEmbedding(self.config.hidden_size)
        self.pos_embed = FluxPosEmbed(
            theta=self.config.theta, axes_dim=list(self.config.axes_dims_rope)
        )
        self.transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.config.hidden_size,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    mlp_ratio=self.config.mlp_ratio,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(
            self.config.hidden_size,
            self.config.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.proj_out = nn.Linear(self.config.hidden_size, self.config.out_channels)
        self.gradient_checkpointing = False

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def _validate_inputs(
        self,
        noisy_ir_tokens: torch.Tensor,
        visible_ll_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        image_ids: torch.Tensor,
    ) -> None:
        if noisy_ir_tokens.ndim != 3 or visible_ll_tokens.ndim != 3:
            raise ValueError("Both token inputs must have shape (B, N, C).")
        if noisy_ir_tokens.shape != visible_ll_tokens.shape:
            raise ValueError(
                "noisy_ir_tokens and visible_ll_tokens must have equal shapes, got "
                f"{tuple(noisy_ir_tokens.shape)} and {tuple(visible_ll_tokens.shape)}."
            )
        if noisy_ir_tokens.shape[-1] != self.config.in_channels:
            raise ValueError(
                f"Expected packed tokens with {self.config.in_channels} channels, "
                f"got {noisy_ir_tokens.shape[-1]}."
            )
        if timesteps.shape != (noisy_ir_tokens.shape[0],):
            raise ValueError(
                "timesteps must have shape (batch,), matching the input batch size."
            )
        if image_ids.shape != (noisy_ir_tokens.shape[1], 3):
            raise ValueError(
                "image_ids must have shape (token_count, 3), matching the input tokens."
            )

    def forward(
        self,
        noisy_ir_tokens: torch.Tensor,
        visible_ll_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        image_ids: torch.Tensor | None = None,
        latent_height: int | None = None,
        latent_width: int | None = None,
    ) -> torch.Tensor:
        if image_ids is None:
            if latent_height is None or latent_width is None:
                raise ValueError(
                    "Provide image_ids, or both latent_height and latent_width."
                )
            image_ids = prepare_flux_ids(
                latent_height, latent_width, device=noisy_ir_tokens.device
            )
        else:
            image_ids = image_ids.to(device=noisy_ir_tokens.device)

        self._validate_inputs(noisy_ir_tokens, visible_ll_tokens, timesteps, image_ids)

        temb = self.time_embed(timesteps.to(device=noisy_ir_tokens.device) * 1000)
        temb = temb.to(dtype=noisy_ir_tokens.dtype)
        image_rotary_emb = self.pos_embed(torch.cat((image_ids, image_ids), dim=0))

        hidden_states = self.x_embedder(noisy_ir_tokens)
        hidden_states = hidden_states + self.token_type_embedding[0].to(hidden_states.dtype)
        condition_states = self.condition_embedder(visible_ll_tokens)
        condition_states = condition_states + self.token_type_embedding[1].to(
            condition_states.dtype
        )

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:
                condition_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    condition_states,
                    temb,
                    image_rotary_emb,
                    use_reentrant=False,
                )
            else:
                condition_states, hidden_states = block(
                    hidden_states,
                    condition_states,
                    temb,
                    image_rotary_emb=image_rotary_emb,
                )

            condition_states = self.condition_embedder(visible_ll_tokens)
            condition_states = condition_states + self.token_type_embedding[1].to(
                condition_states.dtype
            )

        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


__all__ = ["PhysicalFluxDiTConfig", "TimestepEmbedding", "PhysicalFluxDiT"]
