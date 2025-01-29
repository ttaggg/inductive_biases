"""Base models for INR training."""

from typing import Dict

import lightning as L
import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer

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
            pointcloud_path=self.eval_cfg.file_path,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.inr(inputs)
        return outputs

    def training_step(self, model_inputs: Dict[str, torch.Tensor], _) -> torch.Tensor:
        losses = self.loss_fn(self.inr, model_inputs)
        loss = torch.stack(list(losses.values())).mean()
        self.log("losses/total", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.inr.parameters(), lr=self.model_cfg.lr)
        return optimizer

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
        )
        # Log results using self.logger.experiment, because direct use of log_dict()
        # is not possible in on_train_end.
        if self.logger:
            for key, value in results.items():
                self.logger.experiment.add_scalar(key, value, self.current_epoch)
