"""Visualization."""

import typer
from typing_extensions import Annotated

from ib.utils.logging_module import logging

app = typer.Typer(add_completion=False)


@app.command()
def visualization(
    model_path: Annotated[str, typer.Option(...)],
) -> None:
    """Run the visualization."""
    logging.stage("Running visualization.")
    logging.panel(model_path)


if __name__ == "__main__":
    app()
