"""Export INR model to mesh."""

from pathlib import Path
from typing_extensions import Annotated

import typer

from ib.models.decoders import SdfDecoder
from ib.utils.logging_module import logging
from ib.utils.pipeline import resolve_and_expand_path

app = typer.Typer(add_completion=False)


def generate_output_path(model_path: Path, resolution: int) -> str:
    meshes_dir = model_path.parents[1] / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    return str(meshes_dir / f"mesh_{model_path.stem}_res_{resolution}.ply")


@app.command(no_args_is_help=True)
def decoding(
    model_path: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    resolution: int = 512,
    batch_size: int = 256000,
    device: str = "cuda",
    visualize: bool = False,
) -> None:
    """Export encoded in the INR shape to mesh.

    uv run decoding --model-path=[MODEL_PATH] --device=[DEVICE]
        --resolution=[SDF_RESOLUTION] --visualize=[OPEN3D_VIS]
    """
    logging.stage("Running export to a mesh.")

    decoder = SdfDecoder(model_path, device)
    decoder.run(resolution, batch_size)

    output_path = generate_output_path(model_path, resolution)
    decoder.save(output_path)

    if visualize:
        decoder.show()


if __name__ == "__main__":
    app()
