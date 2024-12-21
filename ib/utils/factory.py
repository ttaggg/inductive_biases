"""Create various objects."""

import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from ib.models.base_model import BaseModel
from ib.utils.logging_module import logging


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
        batch_size=1,
        num_workers=trainer_cfg.num_workers,
        persistent_workers=trainer_cfg.persistent_workers,
        **kwargs,
    )
    return data_loader


def create_model(model_cfg: DictConfig) -> nn.Module:
    """Get an appropriate model."""

    if model_cfg.name == "siren_original":
        return BaseModel(model_cfg)

    raise ValueError(f"Unknown model was given: {model_cfg.name}.")


def create_trainer(trainer_cfg: DictConfig) -> L.Trainer:
    """Configure trainer."""

    trainer = L.Trainer(
        # Flags.
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        max_epochs=trainer_cfg.max_epochs,
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        # Callbacks and loggers.
        logger=TensorBoardLogger(
            save_dir=trainer_cfg.paths.lightning_logs, name="", version=""
        ),
        callbacks=[LearningRateMonitor(logging_interval="epoch")],
        enable_checkpointing=False,
        # Debug.
        fast_dev_run=trainer_cfg.fast_dev_run,
        limit_train_batches=trainer_cfg.limit_train_batches,
    )

    return trainer
