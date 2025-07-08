"""STAF.

Copied and adapted from https://github.com/AlirezaMorsali/STAF
"""

import torch
from torch import nn
import math


class StafLayer(nn.Module):
    """
    StafLayer applies a sinusoidal modulation to the output of a linear transformation.

    The layer first performs a linear mapping on the input, then modulates the result by a weighted sum
    of sinusoidal functions. Each sinusoid has its own frequency (ws), phase (phis), and scale factor (bs).

    The modulation is defined as:
        output = Σ_{i=1}^{tau} bs[i] * sin(ws[i] * linout + phis[i])

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, adds a learnable bias to the linear layer. Default is True.
        is_first (bool, optional): If True, indicates that this is the first layer in the network.
                                   This may affect initialization (see paper Sec. 3.2 and supplement Sec. 1.5).
                                   Default is False.
        omega_0 (float, optional): Frequency factor that scales the activations before applying the sinusoidal function.
                                   It is also used to scale weights in subsequent layers. Default is 30.
        scale (float, optional): A scaling factor (unused directly in this code but reserved for potential extensions). Default is 10.0.
        init_weights (bool, optional): Flag indicating whether to initialize weights. Default is True.
    """

    def __init__(
        self,
        in_features,
        out_features,
        tau,
        bias=True,
        is_first=False,
        omega_0=30,
    ):
        super().__init__()

        # Set number of sinusoidal functions to combine (tau = number of frequencies)
        self.tau = tau
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        # Define a linear transformation layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Initialize the sinusoidal parameters: frequencies (ws), phases (phis), and scale factors (bs)
        self.init_params()

    def init_params(self):
        """
        Initialize parameters for the sinusoidal activations.

        - ws: Frequency parameters, scaled by omega_0 and randomly initialized.
        - phis: Phase offsets, uniformly sampled from [-π, π].
        - bs: Scale factors for each sinusoid, initialized based on a Laplace distribution.
        """
        # Frequencies: scale random values by omega_0
        ws = self.omega_0 * torch.rand(self.tau)
        self.ws = nn.Parameter(ws, requires_grad=True)

        # Phases: initialize uniformly in the range [-π, π]
        self.phis = nn.Parameter(
            -math.pi + 2 * math.pi * torch.rand(self.tau), requires_grad=True
        )

        # Scale factors: initialize based on a Laplace distribution to provide diversity in activations
        diversity_y = 1 / (2 * self.tau)
        laplace_samples = torch.distributions.Laplace(0, diversity_y).sample(
            (self.tau,)
        )
        self.bs = nn.Parameter(
            torch.sign(laplace_samples) * torch.sqrt(torch.abs(laplace_samples)),
            requires_grad=True,
        )

    def forward(self, input):
        """
        Forward pass through the STAF layer.

        Args:
            input (Tensor): Input tensor to the layer.

        Returns:
            Tensor: Output tensor after applying the linear transformation and sinusoidal modulation.
        """
        # Apply linear transformation and then the sinusoidal modulation defined in param_act()
        return self.param_act(self.linear(input))

    def param_act(self, linout):
        """
        Apply the sinusoidal activation function to the linear output.

        This function first unsqueezes the linear output to add a dimension, then applies a sine function
        after scaling by each frequency and shifting by its corresponding phase. The result is weighted by bs.

        Args:
            linout (Tensor): Output from the linear layer.

        Returns:
            Tensor: Sum of modulated sinusoidal activations along the frequency dimension.
        """
        # Compute sinusoidal modulation for each frequency and sum them
        sinusoidal_modulation = self.bs * torch.sin(
            self.ws * linout.unsqueeze(-1) + self.phis
        )
        return sinusoidal_modulation.sum(dim=-1)


class Staf(nn.Module):
    """
    INR (Implicit Neural Representation) network using STAF layers.

    This network is composed of a sequence of STAF layers (nonlinear layers) followed by an optional final linear layer.
    It can be used for tasks like representing images, shapes, or other continuous signals.

    Args:
        in_features (int): Number of input features (or dimensions).
        hidden_features (int): Number of features in the hidden layers.
        hidden_layers (int): Number of hidden layers in the network.
        out_features (int): Number of output features.
        outermost_linear (bool, optional): If True, adds a final linear layer; otherwise, uses a STAF layer as the final layer.
                                           Default is True.
        first_omega_0 (float, optional): Frequency factor for the first layer. Default is 30.
        hidden_omega_0 (float, optional): Frequency factor for hidden layers. Default is 30.0.
        scale (float, optional): Scale factor (passed to each STAF layer). Default is 10.0.
        tau: (int, optional) number of frequencies. Default is 5.
        pos_encode (bool, optional): If True, applies positional encoding to the inputs (not used in this implementation).
        sidelength (int, optional): Reserved for potential image tasks (unused here). Default is 512.
        fn_samples (optional): Reserved for potential future use. Default is None.
        use_nyquist (bool, optional): Reserved flag for Nyquist-related processing. Default is True.
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=True,
        first_omega_0=30,
        hidden_omega_0=30.0,
        tau=5,
    ):
        super().__init__()
        self.nonlin = StafLayer

        self.net = []
        self.net.append(
            self.nonlin(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
                tau=tau,
            )
        )

        for _ in range(hidden_layers):
            self.net.append(
                self.nonlin(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    tau=tau,
                )
            )

        if outermost_linear:
            dtype = torch.float
            final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)

            self.net.append(final_linear)
        else:
            self.net.append(
                self.nonlin(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    tau=tau,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords, return_meta: bool = False):
        output = self.net(coords)

        if return_meta:
            return output, {}

        return output
