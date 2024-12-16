"""General utilities."""

import os
from datetime import date
from typing import Tuple

from typer import Context
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

from ib.utils.logging_module import logging


def init_dir(output_dir_root: str, run_name: str) -> str:
    """Initialize output directory."""

    output_dir_base = f"{run_name}_{date.today().strftime('%d_%b')}"
    output_dir = os.path.join(output_dir_root, output_dir_base)

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    return output_dir


def initialize_run(ctx: Context) -> Tuple[DictConfig, str]:
    """Load Hydra configs, initialize output directory, and logging file."""

    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config", overrides=ctx.args)

    output_dir = init_dir(cfg.output_dir_root, cfg.run_name)
    logging.set_log_file(output_dir)

    logging.panel(OmegaConf.to_yaml(cfg), title="Configuration")
    logging.panel(
        f"Output directory: {output_dir} \nLogs: {logging.log_file_path}",
        title="Outputs",
    )

    return cfg, output_dir
