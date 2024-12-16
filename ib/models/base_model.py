"""Base models for INR training."""

from typing import Dict

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer


class BaseModel(L.LightningModule):
    """Base model for INR training."""

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.inr = instantiate(model_cfg.inr)
        self.loss_fn = instantiate(model_cfg.loss)

    def forward(self, model_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.inr(model_inputs["inputs"])

    def training_step(self, model_inputs: Dict[str, torch.Tensor], _) -> torch.Tensor:
        model_outputs = self.inr(model_inputs["inputs"])
        losses = self.loss_fn(model_inputs, model_outputs)
        loss = torch.stack(list(losses.values())).mean()
        self.log("loss", loss, on_step=True, prog_bar=True)
        self.log_dict(losses, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.inr.parameters(), lr=self.model_cfg.lr)
        return optimizer
