"""General utilities."""

import os
import random
import re
import time
from datetime import date
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import dotenv
import numpy as np
import torch
from hydra import initialize, compose
from lightning.pytorch import seed_everything
from typer import Context
from omegaconf import OmegaConf, DictConfig

from ib.utils.logging_module import logging


def resolve_and_expand_path(path: Path) -> Path:
    return path.expanduser().resolve()


def generate_output_mesh_path(
    base_dir: Path,
    run_name: str,
    current_epoch: int,
    resolution: int,
) -> Path:
    meshes_dir = base_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    return (
        meshes_dir / f"mesh_model_{run_name}_epoch_{current_epoch}_res_{resolution}.ply"
    )


def generate_output_results_path(
    base_dir: Path,
    run_name: str,
    current_epoch: int,
    resolution: int,
) -> Path:
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return (
        results_dir / f"results_{run_name}_epoch_{current_epoch}_res_{resolution}.json"
    )


def decode_path(mesh_path: Path) -> tuple[str, int, int]:
    # Try pattern with resolution
    # Support accidental misnaming of the mesh file.
    m = re.search(
        r"(?:mesh_model_|mesh_|model_)(?P<run_name>.+?)_epoch_(?P<epoch>\d+)_res_(?P<res>\d+)",
        mesh_path.stem,
    )
    if m is not None:
        run_name = m.group("run_name")
        current_epoch = int(m.group("epoch"))
        resolution = int(m.group("res"))
        return run_name, current_epoch, resolution

    # Try pattern without resolution (legacy)
    m = re.search(
        r"(?:mesh_model_|mesh_|model_)(?P<run_name>.+?)_epoch_(?P<epoch>\d+)",
        mesh_path.stem,
    )
    if m is not None:
        run_name = m.group("run_name")
        current_epoch = int(m.group("epoch"))
        return run_name, current_epoch, None

    return None, None, None


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


def set_seed(seed: int = 42) -> None:
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Environment
    os.environ["PYTHONHASHSEED"] = str(seed)
    # PyTorch Lightning
    seed_everything(seed, workers=True)


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
            |   ├── results/
            |   └── saved_models/
            └── version_1/
                ├── config.yaml
                ├── log_file.txt
                ├── lightning_logs/
                ├── results/
                └── saved_models/
    """

    output_dir_base = f"{date.today().strftime('%y-%m-%d')}_{run_name}"
    output_dir = Path(output_dir_root) / output_dir_base
    output_dir = resolve_and_expand_path(output_dir)
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
    results_dir = version_dir / "results"

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
        results=results_dir,
    )


def initialize_run(ctx: Context) -> DictConfig:
    """Load Hydra configs, initialize output directory,
    set logging file, set torch float32 matmul precision, set seed."""

    # Load environment variables.
    dotenv.load_dotenv(".env")

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

    # Set seed.
    if cfg.trainer.seed is not None:
        set_seed(cfg.trainer.seed)

    return cfg
