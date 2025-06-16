"""Export INR model to mesh."""

from pathlib import Path
from typing import Annotated

import typer

from ib.models.decoders import SdfDecoder
from ib.utils.logging_module import logging
from ib.utils.pipeline import (
    generate_output_mesh_path,
    decode_path,
    measure_time,
    resolve_and_expand_path,
)

app = typer.Typer(add_completion=False)


@app.command(no_args_is_help=True)
@measure_time
def decoding(
    model_path: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    resolution: int = 512,
    batch_size: int = 256000,
    device: str = "cuda",
    visualize: bool = False,
) -> None:
    """Export encoded in the INR shape to mesh."""

    logging.stage("Running export to a mesh.")

    decoder = SdfDecoder.from_model_path(model_path, device)
    decoder.run(resolution, batch_size)

    run_name, current_epoch, _ = decode_path(model_path)
    output_path = generate_output_mesh_path(
        model_path.parent.parent, run_name, current_epoch, resolution
    )
    decoder.save(output_path)

    if visualize:
        decoder.show()


if __name__ == "__main__":
    app()
