"""Utils for models."""

import os
from typing import Tuple

import lightning as L
import numpy as np
import torch

from ib.utils.logging_module import logging


def save_model(model: L.LightningModule, output_dir: str, epoch: int) -> None:
    """Save the whole model."""
    model_path = os.path.join(
        output_dir,
        f"model_epoch_{epoch}.pt",
    )
    torch.save(model, model_path)


def make_coordinates(
    grid_size: Tuple[int, int, int],
    coord_range: Tuple[int, int] = (-1.0, 1.0),
) -> torch.Tensor:

    x_coords = np.linspace(coord_range[0], coord_range[1], grid_size[0])
    y_coords = np.linspace(coord_range[0], coord_range[1], grid_size[1])
    z_coords = np.linspace(coord_range[0], coord_range[1], grid_size[2])
    x_coords, y_coords, z_coords = np.meshgrid(x_coords, y_coords, z_coords)
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    z_coords = z_coords.flatten()
    coords = np.stack([x_coords, y_coords, z_coords]).T
    return torch.from_numpy(coords).type(torch.float32)


def query_model(
    model: L.LightningModule,
    resolution: int,
    batch_size: int,
    device: str,
) -> np.array:

    grid_size = (resolution, resolution, resolution)
    coords = make_coordinates(grid_size=grid_size)
    coords = torch.split(coords, batch_size, dim=0)

    combined_sdf = []
    for i, batch_coords in enumerate(coords):
        batch_coords = batch_coords.to(device)

        if i % (int(len(coords) * 0.1) + 1) == 0:
            logging.info(f"{i} / {len(coords)} batches are done.")

        batch_sdf = model(batch_coords)
        batch_sdf = batch_sdf.detach().cpu().numpy()
        combined_sdf.append(batch_sdf)

    sdf = np.concatenate(combined_sdf, axis=0)
    sdf = sdf.reshape(grid_size)
    return sdf
