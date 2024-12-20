"""Utils for models."""

import os

import lightning as L
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
