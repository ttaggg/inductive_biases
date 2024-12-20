"""Utils for models."""

import os
from typing import Tuple

import lightning as L
import numpy as np
import torch


def save_model(model: L.LightningModule, output_dir: str, epoch: int) -> None:
    """Save the whole model."""

    model_dir = os.path.join(output_dir, "saved_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(
        model_dir,
        f"model_epoch_{epoch}.pt",
    )
    torch.save(model, model_path)


def make_coordinates(
    shape: Tuple[int, int, int],
    coord_range: Tuple[int, int] = (-1.0, 1.0),
) -> torch.Tensor:

    x_coords = np.linspace(coord_range[0], coord_range[1], shape[0])
    y_coords = np.linspace(coord_range[0], coord_range[1], shape[1])
    z_coords = np.linspace(coord_range[0], coord_range[1], shape[2])
    x_coords, y_coords, z_coords = np.meshgrid(x_coords, y_coords, z_coords)
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    z_coords = z_coords.flatten()
    coords = np.stack([x_coords, y_coords, z_coords]).T
    return torch.from_numpy(coords).type(torch.float32)


def query_model(model: L.LightningModule, resolution: int) -> np.array:
    coords = make_coordinates(shape=(resolution, resolution, resolution))
    sdf = model(coords)
    sdf = sdf.reshape((resolution, resolution, resolution))
    sdf = sdf.detach().cpu().numpy()
    return sdf
