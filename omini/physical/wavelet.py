"""Standalone one-level Haar wavelet decomposition and reconstruction.

The transform operates on image channels: ``(B, C, H, W)`` becomes
``(B, 4C, H/2, W/2)`` in ``[LL, HL, LH, HH]`` channel order.  The inverse
expects the same ordering.  Both modules are parameter-free and fully
differentiable.
"""

import torch
from torch import nn


def dmt_init_channel(x: torch.Tensor) -> torch.Tensor:
    """Apply a one-level channel-wise Haar DMT.

    Args:
        x: Image tensor with shape ``(batch, channels, height, width)``.
            Height and width must both be even.

    Returns:
        Haar coefficients with shape ``(batch, 4 * channels, height / 2,
        width / 2)`` ordered as ``LL, HL, LH, HH``.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor (B, C, H, W), got {tuple(x.shape)}.")

    _, _, height, width = x.shape
    if height % 2 or width % 2:
        raise ValueError(
            "DMT requires even spatial dimensions, "
            f"but received height={height}, width={width}."
        )

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    x_ll = x1 + x2 + x3 + x4
    x_hl = -x1 - x2 + x3 + x4
    x_lh = -x1 + x2 - x3 + x4
    x_hh = x1 - x2 - x3 + x4

    return torch.cat((x_ll, x_hl, x_lh, x_hh), dim=1)


def idmt_init_channel(x: torch.Tensor) -> torch.Tensor:
    """Apply the inverse of :func:`dmt_init_channel`.

    Args:
        x: Haar coefficients with shape ``(batch, 4 * channels, height,
            width)`` in ``LL, HL, LH, HH`` channel order.

    Returns:
        Reconstructed image tensor with shape
        ``(batch, channels, 2 * height, 2 * width)``.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor (B, 4C, H, W), got {tuple(x.shape)}.")

    batch, channels_four, height, width = x.shape
    if channels_four % 4:
        raise ValueError(
            "IDMT requires a channel count divisible by four, "
            f"but received {channels_four}."
        )

    channels = channels_four // 4
    x_ll = x[:, 0:channels, :, :] / 2
    x_hl = x[:, channels : 2 * channels, :, :] / 2
    x_lh = x[:, 2 * channels : 3 * channels, :, :] / 2
    x_hh = x[:, 3 * channels : 4 * channels, :, :] / 2

    reconstructed = x.new_zeros((batch, channels, height * 2, width * 2))
    reconstructed[:, :, 0::2, 0::2] = x_ll - x_hl - x_lh + x_hh
    reconstructed[:, :, 1::2, 0::2] = x_ll - x_hl + x_lh - x_hh
    reconstructed[:, :, 0::2, 1::2] = x_ll + x_hl - x_lh - x_hh
    reconstructed[:, :, 1::2, 1::2] = x_ll + x_hl + x_lh + x_hh

    return reconstructed


class DMT(nn.Module):
    """Parameter-free one-level channel-wise Haar decomposition."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dmt_init_channel(x)


class IDMT(nn.Module):
    """Parameter-free inverse of :class:`DMT`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return idmt_init_channel(x)


__all__ = ["DMT", "IDMT", "dmt_init_channel", "idmt_init_channel"]
