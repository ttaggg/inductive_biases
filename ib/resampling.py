"""Resampling."""

from pathlib import Path
from typing import Annotated

import typer

from ib.datasets.resamplers import Resampler
from ib.utils.logging_module import logging
from ib.utils.pipeline import measure_time, resolve_and_expand_path

app = typer.Typer(add_completion=False)


def generate_output_path(file_path: Path, num_samples: int) -> Path:
    return file_path.with_name(
        f"{file_path.stem}_resampled_{num_samples}{file_path.suffix}"
    )


@app.command(no_args_is_help=True)
@measure_time
def resampling(
    input_path: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    num_samples: Annotated[int, typer.Option(...)],
    visualize: bool = False,
) -> None:
    """Resample pointcloud vertices."""

    logging.stage("Running resampling.")

    resampler = Resampler.from_obj_file(input_path)
    resampler.run(num_samples)

    output_path = generate_output_path(input_path, num_samples)
    resampler.save(output_path)

    if visualize:
        resampler.show()


if __name__ == "__main__":
    app()
