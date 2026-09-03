"""Learned inverse-DMT reconstruction path.

The target LL coefficients come from the first-stage generator.  HFRMNet uses
them together with the source image's high-frequency coefficients to predict
target high frequencies, after which IDMT reconstructs the full target image.
"""

import torch
from torch import nn

from .hfrm import HFRMNet
from .wavelet import IDMT


class HFRMReconstructor(nn.Module):
    """Combine high-frequency prediction with parameter-free IDMT.

    Args:
        in_channels_ll: Number of LL channels (three for RGB/infrared images).
        in_channels_hf: Number of concatenated high-frequency channels (nine
            for RGB ``HL, LH, HH`` bands).
        base_dim: Base width of HFRMNet.
    """

    def __init__(
        self, in_channels_ll: int = 3, in_channels_hf: int = 9, base_dim: int = 96
    ):
        super().__init__()
        self.hfrm = HFRMNet(
            in_channels_ll=in_channels_ll,
            in_channels_hf=in_channels_hf,
            base_dim=base_dim,
        )
        self.idmt = IDMT()

    def predict_high(
        self, infrared_ll: torch.Tensor, visible_high: torch.Tensor
    ) -> torch.Tensor:
        """Predict target-domain ``[HL, LH, HH]`` coefficients."""
        return self.hfrm(infrared_ll, visible_high)

    def forward(
        self, infrared_ll: torch.Tensor, visible_high: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(reconstructed_infrared, predicted_infrared_high)``."""
        infrared_high = self.predict_high(infrared_ll, visible_high)
        reconstructed_infrared = self.idmt(torch.cat([infrared_ll, infrared_high], dim=1))
        return reconstructed_infrared, infrared_high


__all__ = ["HFRMReconstructor"]
