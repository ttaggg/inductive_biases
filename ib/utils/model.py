"""Utils for models."""

import sys
from pathlib import Path
from typing import Generator

import lightning as L
import numpy as np
import torch
from tqdm import tqdm

from ib.utils.logging_module import logging


def save_model(
    model: L.LightningModule,
    output_dir: Path,
    run_name: str,
    epoch: int,
) -> None:
    """Save the whole model."""
    model_path = output_dir / f"model_{run_name}_epoch_{epoch}.pt"
    torch.save(model, model_path)


def make_coordinates(
    grid_size: tuple[int, int, int],
    coord_range: tuple[int, int] = (-1.0, 1.0),
) -> torch.Tensor:

    x_coords = np.linspace(
        coord_range[0],
        coord_range[1],
        grid_size[0],
        dtype=np.float32,
    )
    y_coords = np.linspace(
        coord_range[0],
        coord_range[1],
        grid_size[1],
        dtype=np.float32,
    )
    z_coords = np.linspace(
        coord_range[0],
        coord_range[1],
        grid_size[2],
        dtype=np.float32,
    )

    x_coords, y_coords, z_coords = np.meshgrid(
        x_coords, y_coords, z_coords, indexing="ij"
    )
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    z_coords = z_coords.flatten()
    coords = np.stack([x_coords, y_coords, z_coords]).T
    return torch.from_numpy(coords)


def make_coordinates_generator(
    grid_size: tuple[int, int, int],
    batch_size: int,
    coord_range: tuple[float, float] = (-1.0, 1.0),
) -> Generator[torch.Tensor, None, None]:
    """Memory-efficient generator."""

    x_coords = torch.linspace(
        coord_range[0],
        coord_range[1],
        grid_size[0],
        dtype=torch.float32,
    )
    y_coords = torch.linspace(
        coord_range[0],
        coord_range[1],
        grid_size[1],
        dtype=torch.float32,
    )
    z_coords = torch.linspace(
        coord_range[0],
        coord_range[1],
        grid_size[2],
        dtype=torch.float32,
    )

    total_points = grid_size[0] * grid_size[1] * grid_size[2]

    # Generate coordinates in batches without pre-allocating huge tensors
    for start_idx in range(0, total_points, batch_size):
        end_idx = min(start_idx + batch_size, total_points)

        batch_indices = torch.arange(start_idx, end_idx, dtype=torch.long)
        # Convert to 3D indices (same as meshgrid with indexing="ij")
        k = batch_indices % grid_size[2]
        j = (batch_indices // grid_size[2]) % grid_size[1]
        i = batch_indices // (grid_size[1] * grid_size[2])

        batch_coords = torch.stack([x_coords[i], y_coords[j], z_coords[k]], dim=1)

        yield batch_coords


@torch.no_grad()
def query_model(
    model: L.LightningModule,
    resolution: int,
    batch_size: int,
    device: str,
) -> np.array:
    logging.stage("Querying the model.")

    grid_size = (resolution, resolution, resolution)

    if resolution <= 1024:
        coords = make_coordinates(grid_size=grid_size)
        coords = torch.split(coords, batch_size, dim=0)
        num_batches = len(coords)
    else:
        logging.info("Using generator to make coordinates.")
        coords = make_coordinates_generator(grid_size=grid_size, batch_size=batch_size)
        num_batches = (resolution**3 + batch_size - 1) // batch_size

    with tqdm(
        total=num_batches,
        desc="Query model",
        unit=" steps",
        dynamic_ncols=True,
        disable=not sys.stdout.isatty(),
    ) as pbar:

        combined_sdf = []
        for batch_coords in coords:
            batch_coords = batch_coords.to(device)
            batch_sdf = model(batch_coords)
            batch_sdf = batch_sdf.detach().cpu().numpy()
            combined_sdf.append(batch_sdf)

            pbar.update(1)

    sdf = np.concatenate(combined_sdf, axis=0)
    sdf = sdf.reshape(grid_size)
    return sdf
