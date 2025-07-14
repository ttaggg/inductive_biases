import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ib.models.inrs.common import PosEncoding
from ib.utils.logging_module import logging


class Modulator(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        n_layers: int,
        num_frequencies: int,
    ) -> None:
        super().__init__()

        self.pos_encoding = PosEncoding(in_features, num_frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(self.pos_encoding.out_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_layers * 3),
        )
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(x)
        x = self.mlp(x)
        x = torch.unsqueeze(x.T, dim=-1)
        x = torch.reshape(x, (self.n_layers, 3, -1, 1))

        return x  # [3, n_layers, batch_size, 1]


class SirenFmLayer(nn.Module):
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

    def forward(self, inputs: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        x = self.linear(inputs)
        if self.is_last:
            return x

        a = F.softplus(beta[0])
        b = F.softplus(beta[1])
        c = beta[2]

        return a * torch.sin(b * self.omega_0 * x + c)


class SirenFm(nn.Module):
    def __init__(
        self,
        in_features: int,
        mod_hidden_size: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        num_frequencies: int = 10,
        first_omega: float = 30.0,
        hidden_omega: float = 30.0,
        modulator_path: str = None,
    ):
        super().__init__()
        self.modulator_path = modulator_path
        self.modulator = Modulator(
            in_features=in_features,
            hidden_size=mod_hidden_size,
            n_layers=hidden_layers + 1,
            num_frequencies=num_frequencies,
        )

        if modulator_path is not None:
            modulator_sd = torch.load(modulator_path)
            self.modulator.load_state_dict(modulator_sd)
            logging.info(f"Loaded modulator from {modulator_path}")

        layers = []
        layers.append(
            SirenFmLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega,
            )
        )
        for _ in range(hidden_layers):
            layers.append(
                SirenFmLayer(
                    hidden_features,
                    hidden_features,
                    omega_0=hidden_omega,
                )
            )
        layers.append(
            SirenFmLayer(
                hidden_features,
                out_features,
                is_last=True,
                omega_0=hidden_omega,
            )
        )
        self.layers = nn.ModuleList(layers)

    def forward(
        self, inputs: torch.Tensor, return_meta: bool = False
    ) -> tuple[torch.Tensor, dict]:
        x = inputs

        betas = self.modulator(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, betas[i])

        # Handle last layer.
        x = self.layers[-1](x, None)

        metadata = {}
        for i, beta in enumerate(betas):
            beta = beta.detach()

            a = F.softplus(beta[0])
            b = F.softplus(beta[1])
            c = beta[2]

            metadata[f"a_mean/{i}"] = a.mean().detach()
            metadata[f"a_max/{i}"] = a.max().detach()
            metadata[f"a_min/{i}"] = a.min().detach()
            metadata[f"b_mean/{i}"] = b.mean().detach()
            metadata[f"b_max/{i}"] = b.max().detach()
            metadata[f"b_min/{i}"] = b.min().detach()
            metadata[f"c_mean/{i}"] = c.mean().detach()
            metadata[f"c_max/{i}"] = c.max().detach()
            metadata[f"c_min/{i}"] = c.min().detach()
            # metadata[f"b/{i}"] = b.detach()
            # metadata[f"a/{i}"] = a.detach()
            # metadata[f"c/{i}"] = c.detach()

        if return_meta:
            return x, metadata
        return x
