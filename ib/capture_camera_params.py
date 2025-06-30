"""Save camera params for LPIPS metric."""

import json
from pathlib import Path
from typing import Annotated

import typer
import open3d as o3d
import open3d.visualization.rendering as rendering

from ib.utils.geometry import color_mesh_to_normal_direction
from ib.utils.logging_module import logging
from ib.utils.pipeline import measure_time, resolve_and_expand_path

app = typer.Typer(add_completion=False)


def _setup_scene(vis: o3d.visualization.O3DVisualizer) -> None:
    """Callback to set up the scene lighting and material properties."""
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    vis.scene.modify_geometry_material("mesh", material)
    vis.scene.show_skybox(False)


class CameraParamsSaver:
    """Helper class to manage multiple camera parameter saves with sequential numbering."""

    def __init__(self, base_output_dir: Path):
        self.base_output_dir = base_output_dir
        self.counter = 0

    def save_camera(self, win: o3d.visualization.O3DVisualizer) -> None:
        """Callback that writes the current view to camera_params_{counter}.json"""
        output_path = (
            self.base_output_dir
            / "camera_params"
            / f"camera_params_{self.counter}.json"
        )
        camera = win.scene.camera
        cam_params = {
            "intrinsic": {
                "width": win.size.width,
                "height": win.size.height,
                "intrinsic_matrix": camera.get_projection_matrix().tolist(),
            },
            "extrinsic": camera.get_model_matrix().tolist(),
        }
        with output_path.open("w") as f:
            json.dump(cam_params, f, indent=2)
        logging.info(f"Camera params saved to {output_path}")
        self.counter += 1


@app.command(no_args_is_help=True)
@measure_time
def capture_camera_params(
    mesh_path: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
) -> None:
    """Capture camera parameters (intrinsic + extrinsic) and save to JSON."""

    logging.stage("Visualizing mesh.")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    color_mesh_to_normal_direction(mesh)

    camera_saver = CameraParamsSaver(base_output_dir=mesh_path.parent)
    o3d.visualization.draw(
        {"geometry": mesh, "name": "mesh"},
        title="Adjust view and click 'Save camera' multiple times for different positions",
        width=1280,
        height=800,
        actions=[("Save camera", camera_saver.save_camera)],
        show_ui=True,
        on_init=_setup_scene,
    )


if __name__ == "__main__":
    app()
