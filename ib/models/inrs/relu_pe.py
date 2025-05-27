"""ReLU + positional encoding architecture."""

import torch
from torch import nn

from ib.models.inrs.common import PosEncoding


class LinearBlock(nn.Module):
    """Linear layer followed by a ReLU activation."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return x

    def init_weights(self) -> None:
        nn.init.kaiming_normal_(
            self.linear.weight, a=0.0, nonlinearity="relu", mode="fan_in"
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
    ):
        super().__init__()

        self.pos_encoding = PosEncoding(in_features, num_frequencies)

        layers = [LinearBlock(self.pos_encoding.out_features, hidden_features)]
        for _ in range(hidden_layers):
            layers.append(LinearBlock(hidden_features, hidden_features))
        layers.append(nn.Linear(hidden_features, out_features))

        self.net = nn.Sequential(*layers)

    def forward(
        self, inputs: torch.Tensor, return_meta: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        x = self.pos_encoding(inputs)
        outputs = self.net(x)
        if return_meta:
            return outputs, {}
        return outputs
