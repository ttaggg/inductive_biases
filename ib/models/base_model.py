"""Base models for INR training."""

import yaml
from pathlib import Path

import lightning as L
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer

from ib.utils.logging_module import logging
from ib.utils.model import save_model
from ib.metrics.evaluator import Evaluator


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
        self.eval_cfg = model_cfg.evaluator
        self.evaluator = Evaluator(
            file_path=Path(self.eval_cfg.file_path),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.inr(inputs)
        return outputs

    def training_step(self, model_inputs: dict[str, torch.Tensor], _) -> torch.Tensor:
        losses = self.loss_fn(self.inr, model_inputs)
        loss = torch.stack(list(losses.values())).mean()
        self.log("losses/total", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.inr.parameters(), lr=self.model_cfg.lr)
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

    def on_train_epoch_end(self) -> None:
        """Actions to make in the end of epoch."""
        if (
            self.current_epoch > 0
            and self.current_epoch % self.model_cfg.save_model_every_n_epochs == 0
        ):
            save_model(self, self.model_cfg.paths.saved_models, self.current_epoch)

        if (
            self.current_epoch > 0
            and self.current_epoch % self.model_cfg.eval_model_every_n_epochs == 0
        ):
            self.evaluate()

    def on_train_end(self) -> None:
        """Actions to perform after the training is complete."""
        save_model(self, self.model_cfg.paths.saved_models, self.current_epoch)
        self.evaluate()

    def evaluate(self) -> None:
        """Run the evaluator on the current model."""
        results = self.evaluator.run(
            model=self,
            metric_names=self.eval_cfg.metric_names,
            resolution=self.eval_cfg.resolution,
            batch_size=self.eval_cfg.batch_size,
            save_mesh=self.eval_cfg.save_mesh,
        )
        # Log results using self.logger.experiment, because direct use of log_dict()
        # is not possible in on_train_end.
        if self.logger:
            for key, value in results.items():
                self.logger.experiment.add_scalar(key, value, self.current_epoch)

        logging.panel(f"Results: Epoch {self.current_epoch}.", yaml.dump(results))
