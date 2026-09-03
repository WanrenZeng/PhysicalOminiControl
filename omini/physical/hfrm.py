"""High-frequency reconstruction module for visible-to-infrared translation.

``HFRMNet`` predicts target-domain Haar high-frequency coefficients from a
predicted target LL band and source-domain high-frequency coefficients.  It is
standalone PyTorch code and has no FLUX, condition-attention, or ControlNet
dependency.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels, norm=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.conv2(self.conv1(x)))


class Encoder(nn.Module):
    """Three-scale feature encoder used for LL and high-frequency inputs."""

    def __init__(self, in_channels: int, base_dim: int = 64):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, base_dim),
            ResidualBlock(base_dim),
            ResidualBlock(base_dim),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ResidualBlock(base_dim * 2),
            ResidualBlock(base_dim * 2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(
                base_dim * 2, base_dim * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.GELU(),
            ResidualBlock(base_dim * 4),
            ResidualBlock(base_dim * 4),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f1, f2, f3


class FusionBlock(nn.Module):
    """Fuses deepest LL and source high-frequency features."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.res = ResidualBlock(dim)

    def forward(self, high_feature: torch.Tensor, ll_feature: torch.Tensor) -> torch.Tensor:
        return self.res(self.conv(torch.cat([high_feature, ll_feature], dim=1)))


class Decoder(nn.Module):
    """High-frequency decoder with same-scale skip connections."""

    def __init__(self, out_channels: int, base_dim: int = 64):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                base_dim * 4, base_dim * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.GELU(),
            ResidualBlock(base_dim * 2),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                base_dim * 2, base_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.GELU(),
            ResidualBlock(base_dim),
        )
        self.out_conv = nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1)

    def forward(
        self, feature_1: torch.Tensor, feature_2: torch.Tensor, feature_3: torch.Tensor
    ) -> torch.Tensor:
        x = self.up1(feature_3) + feature_2
        x = self.up2(x) + feature_1
        return self.out_conv(x)


class HFRMNet(nn.Module):
    """Predict target high-frequency bands from target LL and source HF bands.

    For RGB-to-infrared reconstruction, the default inputs are:

    * ``infrared_ll``: predicted infrared LL coefficients, ``(B, 3, H, W)``;
    * ``visible_high``: visible ``[HL, LH, HH]`` coefficients, ``(B, 9, H, W)``.

    The output is predicted infrared ``[HL, LH, HH]`` coefficients with shape
    ``(B, 9, H, W)``.  ``H`` and ``W`` must be divisible by four because the
    network contains two downsampling stages.
    """

    def __init__(
        self, in_channels_ll: int = 3, in_channels_hf: int = 9, base_dim: int = 96
    ):
        super().__init__()
        self.in_channels_ll = in_channels_ll
        self.in_channels_hf = in_channels_hf
        self.ll_encoder = Encoder(in_channels_ll, base_dim)
        self.hf_encoder = Encoder(in_channels_hf, base_dim)
        self.fusion = FusionBlock(base_dim * 4)
        self.decoder = Decoder(out_channels=in_channels_hf, base_dim=base_dim)

    def forward(self, infrared_ll: torch.Tensor, visible_high: torch.Tensor) -> torch.Tensor:
        if infrared_ll.ndim != 4 or visible_high.ndim != 4:
            raise ValueError("HFRMNet inputs must both be 4D tensors (B, C, H, W).")
        if infrared_ll.shape[0] != visible_high.shape[0] or infrared_ll.shape[-2:] != visible_high.shape[-2:]:
            raise ValueError(
                "infrared_ll and visible_high must have matching batch and spatial dimensions."
            )
        if infrared_ll.shape[1] != self.in_channels_ll:
            raise ValueError(
                f"Expected infrared_ll to have {self.in_channels_ll} channels, "
                f"got {infrared_ll.shape[1]}."
            )
        if visible_high.shape[1] != self.in_channels_hf:
            raise ValueError(
                f"Expected visible_high to have {self.in_channels_hf} channels, "
                f"got {visible_high.shape[1]}."
            )
        height, width = infrared_ll.shape[-2:]
        if height % 4 or width % 4:
            raise ValueError(
                "HFRMNet requires spatial dimensions divisible by four, "
                f"but received height={height}, width={width}."
            )

        _, _, ll_f3 = self.ll_encoder(infrared_ll)
        high_f1, high_f2, high_f3 = self.hf_encoder(visible_high)
        fused = self.fusion(high_f3, ll_f3)
        return self.decoder(high_f1, high_f2, fused)


__all__ = [
    "ConvBlock",
    "ResidualBlock",
    "Encoder",
    "FusionBlock",
    "Decoder",
    "HFRMNet",
]
