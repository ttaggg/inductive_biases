"""Various callbacks."""

import yaml
from pathlib import Path

import lightning as L
from omegaconf import DictConfig

from ib.metrics.evaluator import Evaluator
from ib.utils.logging_module import logging
from ib.utils.model import save_model


class EvaluatorCallback(L.Callback):
    """Callback to evaluate the model during the training."""

    def __init__(self, eval_cfg: DictConfig):
        self.eval_cfg = eval_cfg
        self.evaluator = Evaluator(
            file_path=Path(self.eval_cfg.file_path),
            metric=self.eval_cfg.metric_names,
        )

    def evaluate(self, pl_module: L.LightningModule) -> None:
        """Run the evaluator on the current model."""
        results = self.evaluator.run(
            model=pl_module,
            resolution=self.eval_cfg.resolution,
            batch_size=self.eval_cfg.batch_size,
            save_mesh=self.eval_cfg.save_mesh,
        )
        logging.panel(f"Results: Epoch {pl_module.current_epoch}.", yaml.dump(results))

        # Log results using self.logger.experiment, because direct use of log_dict()
        # is not possible in on_train_end.
        if pl_module.logger:
            for key, value in results.items():
                pl_module.logger.experiment.add_scalar(
                    key, value, pl_module.current_epoch
                )

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        cfg = pl_module.model_cfg
        epoch = trainer.current_epoch
        if epoch > 0 and epoch % cfg.eval_model_every_n_epochs == 0:
            self.evaluate(pl_module)

    def on_train_end(self, _: L.Trainer, pl_module: L.LightningModule) -> None:
        self.evaluate(pl_module)


class SaveModelCallback(L.Callback):
    """Callback to save the whole L.Lightning module during the training."""

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        cfg = pl_module.model_cfg
        epoch = trainer.current_epoch
        run_name = pl_module.model_cfg.run_name.replace("/", "_")
        if epoch > 0 and epoch % cfg.save_model_every_n_epochs == 0:
            save_model(pl_module, cfg.paths.saved_models, run_name, epoch)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        cfg = pl_module.model_cfg
        epoch = trainer.current_epoch
        run_name = pl_module.model_cfg.run_name.replace("/", "_")
        save_model(pl_module, cfg.paths.saved_models, run_name, epoch)
