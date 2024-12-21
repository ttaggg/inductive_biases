"""General utilities."""

import os
from datetime import date
from types import SimpleNamespace

import torch
from typer import Context
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

from ib.utils.logging_module import logging


def initialize_directories(output_dir_root: str, run_name: str) -> SimpleNamespace:
    """
    Create a consistent output structure for a new run.

    Expected directory structure:
    outputs/YY-MM-DD_run_name/
            ├── latest -> version_1
            ├── version_0/
            |   ├── log_file.txt
            |   ├── lightning_logs/
            |   └── saved_models/
            └── version_1/
                ├── log_file.txt
                ├── lightning_logs/
                └── saved_models/
    """

    output_dir_base = f"{date.today().strftime('%y-%m-%d')}_{run_name}"
    output_dir = os.path.join(output_dir_root, output_dir_base)
    os.makedirs(output_dir, exist_ok=True)

    # Determine the next version
    version_dirs = [
        subdir
        for subdir in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, subdir))
        and subdir.startswith("version_")
    ]
    next_version = f"version_{len(version_dirs)}"

    # Create directories for the new version
    version_dir = os.path.join(output_dir, next_version)
    lightning_logs_dir = os.path.join(version_dir, "lightning_logs")
    saved_models_dir = os.path.join(version_dir, "saved_models")

    os.makedirs(lightning_logs_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)

    # Create or update the 'latest' symlink
    symlink_path = os.path.join(output_dir, "latest")
    if os.path.islink(symlink_path) or os.path.exists(symlink_path):
        os.remove(symlink_path)
    os.symlink(version_dir, symlink_path)

    return SimpleNamespace(
        version=version_dir,
        lightning_logs=lightning_logs_dir,
        saved_models=saved_models_dir,
    )


def initialize_run(ctx: Context) -> DictConfig:
    """Load Hydra configs, initialize output directory,
    set logging file, set torch float32 matmul precision."""

    # Parse config, initialize directories.
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config", overrides=ctx.args)

    paths = initialize_directories(cfg.output_dir_root, cfg.run_name)
    cfg.paths = OmegaConf.create(vars(paths))

    # Log to file.
    logging.set_log_file(cfg.paths.version)

    # Beautiful panel with configuration.
    logging.panel(OmegaConf.to_yaml(cfg), title="Configuration")
    logging.panel(
        f"Output directory: {cfg.paths.version} \nLogs: {logging.log_file_path}",
        title="Outputs",
    )
    OmegaConf.save(config=cfg, f=os.path.join(cfg.paths.version, "config.yaml"))

    # Also set float32_matmul_precision.
    # TODO(oleg): maybe remove this part, it does not belong here.
    torch.set_float32_matmul_precision(cfg.trainer.float32_matmul_precision)

    return cfg
