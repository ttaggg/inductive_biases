"""Common units for different INR."""

import torch
from torch import nn


class PosEncoding(nn.Module):
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""

    def __init__(self, in_features: int, num_frequencies: int) -> None:
        super().__init__()

        self.out_features = in_features * (1 + 2 * num_frequencies)

        frequencies = torch.arange(num_frequencies, dtype=torch.float32)
        frequencies = (2**frequencies * torch.pi).unsqueeze(0)
        self.register_buffer("freqs", frequencies)

        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_expanded = x.unsqueeze(-1)
        x_sin = torch.sin(x_expanded * self.freqs)
        x_cos = torch.cos(x_expanded * self.freqs)

        pos_enc = torch.cat([x_expanded, x_sin, x_cos], dim=-1)
        pos_enc = self.flatten(pos_enc)

        return pos_enc
