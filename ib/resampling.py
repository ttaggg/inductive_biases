"""Resampling."""

import typer
from typing_extensions import Annotated

from ib.utils.logging_module import logging

app = typer.Typer(add_completion=False)


@app.command()
def resampling(mesh_path: Annotated[str, typer.Option(...)]) -> None:
    """Resample mesh vertices."""
    logging.stage("Running resampling.")
    logging.panel(mesh_path)


if __name__ == "__main__":
    app()
