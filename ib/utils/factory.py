"""Create various objects."""

import os

import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
import wandb

from ib.models.base_model import BaseModel
from ib.utils.callbacks import EvaluatorCallback, SaveModelCallback
from ib.utils.logging_module import logging
from ib.utils.tensorboard import CustomTensorBoardLogger


def create_loader(dataset_cfg: DictConfig, trainer_cfg: DictConfig) -> DataLoader:
    """Get an appropriate loader."""
    logging.stage("Loading the data.")

    # One of the datasets in ib.datasets
    dataset = instantiate(dataset_cfg)

    kwargs = {}
    if trainer_cfg.num_workers > 0:
        kwargs = {"pin_memory": True, "multiprocessing_context": None}

    # NOTE(oleg): currently we create the whole batch in the dataset,
    # because it is 10+ times faster for pointclouds than the normal way.
    data_loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=None,
        num_workers=trainer_cfg.num_workers,
        persistent_workers=trainer_cfg.persistent_workers,
        **kwargs,
    )
    return data_loader


def create_model(model_cfg: DictConfig) -> nn.Module:
    """Get an appropriate model."""
    inr = instantiate(model_cfg.inr)
    loss_fn = instantiate(model_cfg.loss)
    model = BaseModel(inr, loss_fn, model_cfg)
    return model


def create_trainer(trainer_cfg: DictConfig, eval_cfg: DictConfig) -> L.Trainer:
    """Configure trainer."""

    evaluator_callback = EvaluatorCallback(eval_cfg)
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    save_model_callback = SaveModelCallback()

    loggers = [
        CustomTensorBoardLogger(
            save_dir=trainer_cfg.paths.lightning_logs, name="", version=""
        )
    ]
    if "WANDB_API_KEY" in os.environ and os.environ["WANDB_API_KEY"]:
        wandb.login()
        loggers.append(
            WandbLogger(
                name=trainer_cfg.run_name,
                save_dir=trainer_cfg.paths.lightning_logs,
                version="",
                project=os.environ.get("WANDB_PROJECT", None),
                entity=os.environ.get("WANDB_ENTITY", None),
            )
        )

    trainer = L.Trainer(
        # Flags.
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        max_epochs=trainer_cfg.max_epochs,
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        # Callbacks and loggers.
        logger=loggers,
        callbacks=[
            lr_monitor_callback,
            save_model_callback,
            evaluator_callback,
        ],
        enable_checkpointing=False,
        # Debug.
        fast_dev_run=trainer_cfg.fast_dev_run,
        limit_train_batches=trainer_cfg.limit_train_batches,
    )

    return trainer
