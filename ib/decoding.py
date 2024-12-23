"""Export INR model to mesh."""

from pathlib import Path
from typing_extensions import Annotated

import numpy as np
import open3d as o3d
import torch
import typer
from skimage import measure

from ib.utils.logging_module import logging
from ib.utils.model import query_model
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
    logging.stage("Running export.")

    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    logging.stage("Performing forward pass.")
    sdf = query_model(model, resolution, batch_size, device)

    logging.stage("Performing marching cubes.")
    verts, faces, normals, _ = measure.marching_cubes(sdf, level=0)

    logging.stage("Creating mesh.")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

    output_path = generate_output_path(model_path, resolution)
    o3d.io.write_triangle_mesh(output_path, mesh)
    logging.stage(f"Mesh was written to {output_path}")

    if visualize:
        o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)


if __name__ == "__main__":
    app()
