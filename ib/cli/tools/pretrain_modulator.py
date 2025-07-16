"""Pre-train Modulator for stable coefficients."""

import math
import sys
from pathlib import Path
from typing import Annotated, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import typer
from tqdm import tqdm

from ib.models.inrs.vainer import Modulator
from ib.utils.model import make_coordinates
from ib.utils.logging_module import logging
from ib.utils.pipeline import measure_time, resolve_and_expand_path

app = typer.Typer(add_completion=False)


def create_training_coordinates(
    grid_size: int = 64,
    coord_range: Tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    # Generate full grid of coordinates
    full_coords = make_coordinates(
        grid_size=(grid_size, grid_size, grid_size), coord_range=coord_range
    )
    return full_coords


def compute_target_coefficients(
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = F.softplus(beta[0])
    b = F.softplus(beta[1])
    c = beta[2]
    return a, b, c


def train(
    modulator: Modulator,
    dataloader: DataLoader,
    num_epochs: int,
    lr: float,
    target_coeff: float = 1.0,
    device: str = "cuda",
) -> None:
    """Train the modulator to predict default target coefficients."""
    modulator.to(device)
    modulator.train()

    target_beta_ab = math.log(math.exp(target_coeff) - 1.0)
    target_beta_c = 0.0

    criterion = nn.MSELoss()
    optimizer = optim.Adam(modulator.parameters(), lr=lr)

    logging.info(
        f"Training modulator with {sum(p.numel() for p in modulator.parameters())} parameters"
    )
    logging.info(f"Target beta values: a,b={target_beta_ab:.4f}, c={target_beta_c:.4f}")

    with tqdm(
        total=num_epochs,
        desc="Training modulator",
        unit=" epochs",
        dynamic_ncols=True,
        disable=not sys.stdout.isatty(),
    ) as pbar:

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_coeff_error = 0.0
            num_batches = 0

            for batch_coords in dataloader:
                batch_coords = batch_coords[0].to(device)  # DataLoader returns tuple

                optimizer.zero_grad()

                # Forward pass
                betas = modulator(batch_coords)  # [n_layers, 3, batch_size, 1]

                # Create target: specific values for each coefficient type
                target_betas = torch.zeros_like(betas)
                target_betas[:, 0, :, :] = target_beta_ab  # a coefficients
                target_betas[:, 1, :, :] = target_beta_ab  # b coefficients
                target_betas[:, 2, :, :] = target_beta_c  # c coefficients

                # Compute loss - we want beta to match target values
                loss = criterion(betas, target_betas)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Track metrics
                total_loss += loss.item()

                # Compute actual coefficient values for monitoring
                with torch.no_grad():
                    avg_coeff_error = 0.0
                    for layer_beta in betas:
                        a, b, c = compute_target_coefficients(layer_beta)
                        avg_coeff_error += F.mse_loss(
                            a, torch.full_like(a, target_coeff)
                        )
                        avg_coeff_error += F.mse_loss(
                            b, torch.full_like(b, target_coeff)
                        )
                        avg_coeff_error += F.mse_loss(c, torch.full_like(c, 0.0))
                    total_coeff_error += avg_coeff_error.item()

                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_coeff_error = total_coeff_error / num_batches

            print(epoch, avg_loss, avg_coeff_error)
            # Update progress bar
            pbar.set_postfix(
                {
                    "beta_loss": f"{avg_loss:.6f}",
                    "coeff_error": f"{avg_coeff_error:.6f}",
                }
            )
            pbar.update(1)


@app.command(no_args_is_help=True)
@measure_time
def pretrain_modulator(
    save_path: Annotated[
        Path,
        typer.Option(
            callback=resolve_and_expand_path, help="Path to save trained modulator"
        ),
    ],
    in_features: int = typer.Option(3, help="Input feature dimension"),
    mod_hidden_size: int = typer.Option(256, help="Modulator hidden size"),
    n_layers: int = typer.Option(4, help="Number of layers (hidden_layers + 1)"),
    num_frequencies: int = typer.Option(
        5, help="Number of positional encoding frequencies"
    ),
    grid_size: int = typer.Option(128, help="Grid size for coordinate generation"),
    batch_size: int = typer.Option(10000, help="Training batch size"),
    num_epochs: int = typer.Option(20, help="Number of training epochs"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    coord_range_min: float = typer.Option(-1.0, help="Minimum coordinate value"),
    coord_range_max: float = typer.Option(1.0, help="Maximum coordinate value"),
    device: str = typer.Option("cuda", help="Device to use (cpu/cuda)"),
) -> None:
    """Pre-train Modulator for stable coefficients."""

    logging.stage("Pre-training Modulator for stable coefficients")

    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, using CPU")
        device = "cpu"

    logging.info(f"Using device: {device}")

    # Create modulator
    modulator = Modulator(
        in_features=in_features,
        hidden_size=mod_hidden_size,
        n_layers=n_layers,
        num_frequencies=num_frequencies,
    )

    logging.info(
        f"Created Modulator with {sum(p.numel() for p in modulator.parameters())} parameters"
    )

    # Create training coordinates using make_coordinates
    coord_range = (coord_range_min, coord_range_max)
    train_coords = create_training_coordinates(
        grid_size=grid_size, coord_range=coord_range
    )

    train_dataset = TensorDataset(train_coords)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    logging.info(f"Created training dataset with {len(train_dataset)} samples")

    # Train the modulator
    logging.stage("Starting training")
    train(
        modulator=modulator,
        dataloader=train_dataloader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
    )

    torch.save(modulator.state_dict(), save_path)

    logging.panel(
        "Usage",
        f"Load this pre-trained modulator in your main training script:\nmodulator.load_state_dict(torch.load('{save_path}'))",
    )


if __name__ == "__main__":
    app()
