"""Evaluation."""

import yaml
from pathlib import Path
from typing import List
from typing_extensions import Annotated

import typer

from ib.utils.logging_module import logging
from ib.utils.pipeline import measure_time, resolve_and_expand_path
from ib.metrics.evaluator import Evaluator, Metric

app = typer.Typer(add_completion=False)


@app.command(no_args_is_help=True)
@measure_time
def evaluation(
    model_path: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    metric: List[Metric] = typer.Option(...),
    pointcloud_path: Annotated[
        Path, typer.Option(callback=resolve_and_expand_path)
    ] = None,
    resolution: int = 512,
    batch_size: int = 256000,
    device: str = "cuda",
    save_mesh: bool = False,
) -> None:
    """Evaluate the model."""

    logging.stage("Running evaluation.")

    evaluator = Evaluator(pointcloud_path)
    results = evaluator.run_from_path(
        model_path,
        device,
        metric,
        resolution,
        batch_size,
        save_mesh,
    )

    logging.panel("Results", yaml.dump(results))


if __name__ == "__main__":
    app()
