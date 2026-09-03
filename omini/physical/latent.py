import torch


def pack_flux_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(f"Expected latents with shape (B, C, H, W), got {tuple(latents.shape)}.")

    batch_size, channels, height, width = latents.shape
    if height % 2 or width % 2:
        raise ValueError(
            "FLUX latent packing requires even spatial dimensions, "
            f"but got height={height}, width={width}."
        )

    latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)


def unpack_flux_latents(
    tokens: torch.Tensor, latent_height: int, latent_width: int
) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape (B, N, C), got {tuple(tokens.shape)}.")
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError("latent_height and latent_width must both be positive.")
    if latent_height % 2 or latent_width % 2:
        raise ValueError("FLUX latent unpacking requires even spatial dimensions.")

    batch_size, token_count, packed_channels = tokens.shape
    expected_tokens = (latent_height // 2) * (latent_width // 2)
    if token_count != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} tokens for latent shape "
            f"({latent_height}, {latent_width}), got {token_count}."
        )
    if packed_channels % 4:
        raise ValueError(
            f"Packed token channel count must be divisible by four, got {packed_channels}."
        )

    channels = packed_channels // 4
    latents = tokens.view(
        batch_size, latent_height // 2, latent_width // 2, channels, 2, 2
    )
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels, latent_height, latent_width)


def prepare_flux_ids(
    latent_height: int, latent_width: int, *, device: torch.device | str | None = None
) -> torch.Tensor:
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError("latent_height and latent_width must both be positive.")
    if latent_height % 2 or latent_width % 2:
        raise ValueError("FLUX token IDs require even latent spatial dimensions.")

    token_height = latent_height // 2
    token_width = latent_width // 2
    ids = torch.zeros(token_height, token_width, 3, device=device)
    ids[..., 1] = torch.arange(token_height, device=device)[:, None]
    ids[..., 2] = torch.arange(token_width, device=device)[None, :]
    return ids.reshape(token_height * token_width, 3)


__all__ = ["pack_flux_latents", "unpack_flux_latents", "prepare_flux_ids"]
