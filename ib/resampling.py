"""Resampling."""

import os
from typing_extensions import Annotated

import typer

from ib.datasets.resamplers import ObjResampler
from ib.utils.logging_module import logging

app = typer.Typer(add_completion=False)


def generate_output_path(file_path, num_samples):
    base, ext = os.path.splitext(file_path)
    return f"{base}_resampled_{num_samples}{ext}"


@app.command(no_args_is_help=True)
def resampling(
    input_path: Annotated[str, typer.Option(...)],
    num_samples: Annotated[int, typer.Option(...)],
    visualize: bool = False,
) -> None:
    """Resample pointcloud vertices.

    uv run resampling --input-path=[INPUT_OBJ] --num-samples=[NUM_SAMPLES]
    """

    logging.stage("Running resampling.")

    resampler = ObjResampler(input_path)
    resampler.sample_vertices_and_normals(num_samples)

    output_path = generate_output_path(input_path, num_samples)
    resampler.save(output_path)

    if visualize:
        resampler.show_sampled()


if __name__ == "__main__":
    app()
