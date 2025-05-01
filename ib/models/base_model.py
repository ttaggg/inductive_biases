"""Base models for INR training."""

import lightning as L
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer


class BaseModel(L.LightningModule):
    """Base model for INR training."""

    def __init__(
        self,
        inr: nn.Module,
        loss_fn: nn.Module,
        model_cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.inr = inr
        self.loss_fn = loss_fn
        self.model_cfg = model_cfg

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.inr(inputs)
        return outputs

    def training_step(self, model_inputs: dict[str, torch.Tensor], _) -> torch.Tensor:
        losses = self.loss_fn(self.inr, model_inputs)
        loss = torch.stack(list(losses.values())).mean()
        self.log_weights()
        self.log_grads()
        self.log("losses/total", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def log_weights(self) -> None:
        for name, param in self.inr.named_parameters():
            if param.requires_grad:
                self.log(
                    f"weights_norm/{name}",
                    torch.linalg.norm(param),
                    on_step=False,
                    on_epoch=True,
                )

    def log_grads(self) -> None:
        for name, param in self.inr.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.log(
                    f"grads_norm/{name}",
                    torch.linalg.norm(param.grad),
                    on_step=False,
                    on_epoch=True,
                )

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(
            self.inr.parameters(),
            lr=self.model_cfg.lr,
            weight_decay=self.model_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.model_cfg.max_epochs,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Step at the end of each epoch
                "frequency": 1,  # Apply every epoch
            },
        }
