"""Base models for INR training."""

import os
from typing import Dict

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer

from ib.utils.model import save_model


class BaseModel(L.LightningModule):
    """Base model for INR training."""

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.inr = instantiate(model_cfg.inr)
        self.loss_fn = instantiate(model_cfg.loss)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.inr(inputs)
        return outputs

    def training_step(self, model_inputs: Dict[str, torch.Tensor], _) -> torch.Tensor:
        losses = self.loss_fn(self.inr, model_inputs)
        loss = torch.stack(list(losses.values())).mean()
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.inr.parameters(), lr=self.model_cfg.lr)
        return optimizer

    def on_train_epoch_end(self) -> None:
        """Actions to make in the end of epoch."""
        if self.current_epoch % self.model_cfg.save_model_every_n_epochs == 0:
            save_model(self, self.model_cfg.paths.saved_models, self.current_epoch)

    def on_train_end(self) -> None:
        """Actions to perform after the training is complete."""
        save_model(self, self.model_cfg.paths.saved_models, self.current_epoch)
