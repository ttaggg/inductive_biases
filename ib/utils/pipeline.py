"""General utilities."""

import time
from datetime import date
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import torch
from typer import Context
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

from ib.utils.logging_module import logging


def resolve_and_expand_path(path: Path) -> Path:
    return path.expanduser().resolve()


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        timing = []
        if hours > 0:
            timing.append(f"{int(hours)} hours")
        if hours > 0 or minutes > 0:
            timing.append(f"{int(minutes)} minutes")
        timing.append(f"{seconds:.2f} seconds")

        logging.info(f"Execution time for {func.__name__}: {', '.join(timing)}.")
        return result

    return wrapper


def initialize_directories(output_dir_root: str, run_name: str) -> SimpleNamespace:
    """
    Create a consistent output structure for a new run.

    Expected directory structure:
    outputs/YY-MM-DD_run_name/
            ├── latest -> version_1
            ├── version_0/
            |   ├── config.yaml
            |   ├── log_file.txt
            |   ├── lightning_logs/
            |   └── saved_models/
            └── version_1/
                ├── config.yaml
                ├── log_file.txt
                ├── lightning_logs/
                └── saved_models/
    """

    output_dir_base = f"{date.today().strftime('%y-%m-%d')}_{run_name}"
    output_dir = Path(output_dir_root) / output_dir_base
    resolve_and_expand_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine the next version
    version_dirs = [
        subdir
        for subdir in output_dir.iterdir()
        if subdir.is_dir() and subdir.name.startswith("version_")
    ]
    next_version = f"version_{len(version_dirs)}"

    # Create directories for the new version
    version_dir = output_dir / next_version
    lightning_logs_dir = version_dir / "lightning_logs"
    saved_models_dir = version_dir / "saved_models"

    lightning_logs_dir.mkdir(parents=True, exist_ok=True)
    saved_models_dir.mkdir(parents=True, exist_ok=True)

    # Create or update the 'latest' symlink
    symlink_path = output_dir / "latest"
    if symlink_path.is_symlink() or symlink_path.exists():
        symlink_path.unlink()
    symlink_path.symlink_to(version_dir)

    return SimpleNamespace(
        version=version_dir,
        lightning_logs=lightning_logs_dir,
        saved_models=saved_models_dir,
    )


def initialize_run(ctx: Context) -> DictConfig:
    """Load Hydra configs, initialize output directory,
    set logging file, set torch float32 matmul precision."""

    # Parse config, initialize directories.
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="config", overrides=ctx.args)

    paths = initialize_directories(cfg.output_dir_root, cfg.run_name)
    cfg.paths = OmegaConf.create(vars(paths))

    # Log to file.
    logging.set_log_file(cfg.paths.version)

    # Beautiful panel with configuration.
    logging.panel("Configuration", OmegaConf.to_yaml(cfg))
    logging.panel(
        "Outputs",
        f"Output directory: {cfg.paths.version} \nLogs: {logging.log_file_path}",
    )
    OmegaConf.save(config=cfg, f=cfg.paths.version / "config.yaml")

    # Also set float32_matmul_precision.
    # TODO(oleg): maybe remove this part, it does not belong here.
    torch.set_float32_matmul_precision(cfg.trainer.float32_matmul_precision)

    return cfg
