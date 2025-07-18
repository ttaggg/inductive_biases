"""ReLU + positional encoding architecture."""

import numpy as np
import torch
from torch import nn

from ib.models.inrs.common import PosEncoding


class LinearBlock(nn.Module):
    """Linear layer followed by a ReLU activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_magnitude: float = 1.0,
        is_last: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        self.is_last = is_last
        self.init_weights(weight_magnitude)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.is_last:
            return x
        x = self.activation(x)
        return x

    def init_weights(self, weight_magnitude: float) -> None:
        with torch.no_grad():
            self.linear.weight.uniform_(
                -np.sqrt(weight_magnitude / (self.in_features + self.out_features)),
                np.sqrt(weight_magnitude / (self.in_features + self.out_features)),
            )


class ReluPe(nn.Module):
    """ReLU + Positional Encoding architecture."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        num_frequencies: int = 10,
        weight_magnitude: float = 1.0,
    ):
        super().__init__()

        self.pos_encoding = PosEncoding(in_features, num_frequencies)

        layers = [
            LinearBlock(
                self.pos_encoding.out_features, hidden_features, weight_magnitude
            )
        ]
        for _ in range(hidden_layers):
            layers.append(
                LinearBlock(hidden_features, hidden_features, weight_magnitude)
            )
        layers.append(
            LinearBlock(hidden_features, out_features, weight_magnitude, is_last=True)
        )

        self.net = nn.Sequential(*layers)

    def forward(
        self, inputs: torch.Tensor, return_meta: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        x = self.pos_encoding(inputs)
        outputs = self.net(x)
        if return_meta:
            return outputs, {}
        return outputs
