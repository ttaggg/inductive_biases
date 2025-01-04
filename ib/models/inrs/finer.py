"""FINER architecture.

Adapted from:
https://github.com/liuzhen0212/FINERplusplus
"""

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def generate_alpha(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x) + 1


class FinerLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        is_last: bool = False,
        omega: int = 30,
    ) -> None:
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.is_last = is_last
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega,
                    np.sqrt(6 / self.in_features) / self.omega,
                )

    def forward(self, input):
        x = self.linear(input)
        if self.is_last:
            return x
        return torch.sin(self.omega * generate_alpha(x) * x)


class Finer(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        first_omega: float = 30.0,
        hidden_omega: float = 30.0,
    ) -> None:
        super().__init__()

        layers = []
        layers.append(
            FinerLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega=first_omega,
            )
        )
        for _ in range(hidden_layers):
            layers.append(
                FinerLayer(
                    hidden_features,
                    hidden_features,
                    omega=hidden_omega,
                )
            )
        layers.append(
            FinerLayer(
                hidden_features,
                out_features,
                is_last=True,
                omega=hidden_omega,
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.net(inputs)
        return outputs
