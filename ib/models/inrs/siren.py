"""Siren architecture.

Copied from:
https://github.com/vsitzmann/siren
"""

import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    """# See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        is_last: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
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
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.linear(inputs)
        if self.is_last:
            return x
        return torch.sin(self.omega_0 * x)


class Siren(nn.Module):
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
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega,
            )
        )
        for _ in range(hidden_layers):
            layers.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    omega_0=hidden_omega,
                )
            )
        layers.append(
            SineLayer(
                hidden_features,
                out_features,
                is_last=True,
                omega_0=hidden_omega,
            )
        )

        self.net = nn.Sequential(*layers)

    def forward(
        self, inputs: torch.Tensor, return_meta: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        outputs = self.net(inputs)
        if return_meta:
            return outputs, {}
        return outputs
