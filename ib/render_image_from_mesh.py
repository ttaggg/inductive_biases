"""Get image from the saved camera parameters."""

from pathlib import Path
from typing import Annotated

import typer
import open3d as o3d

from ib.utils.geometry import color_mesh_to_normal_direction
from ib.utils.logging_module import logging
from ib.metrics.lpips import extract_image_from_mesh, load_json, parse_cam_params
from ib.utils.pipeline import measure_time, resolve_and_expand_path

app = typer.Typer(add_completion=False)


@app.command(no_args_is_help=True)
@measure_time
def render_image_from_mesh(
    mesh_path: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    cam_params_path: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    color_normals: bool = True,
) -> None:
    """Get image from the saved camera parameters."""

    logging.stage("Visualizing mesh.")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if color_normals:
        color_mesh_to_normal_direction(mesh)

    cam_params = load_json(cam_params_path)
    intrinsic, extrinsic, width, height = parse_cam_params(cam_params)
    _ = extract_image_from_mesh(
        mesh,
        width,
        height,
        extrinsic,
        intrinsic,
        output_path=mesh_path.with_name(f"{mesh_path.stem}_{cam_params_path.stem}.png"),
    )


if __name__ == "__main__":
    app()
