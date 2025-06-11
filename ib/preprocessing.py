from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import open3d as o3d

from ib.utils.data import load_ply, write_ply
from ib.utils.labels import compute_labels
from ib.utils.logging_module import logging
from ib.utils.pipeline import measure_time, resolve_and_expand_path
from ib.utils.pointcloud import (
    normalize_pointcloud,
    normalize_pointcloud_with_margin,
    filter_pointcloud,
)

app = typer.Typer(add_completion=False)


def generate_output_path(file_path: Path, margin: float) -> Path:
    margin = str(margin).replace(".", "")
    return file_path.with_name(
        f"{file_path.stem}_filtered_trimmed_both_{margin}{file_path.suffix}"
    )


def generate_output_recon_mesh_path(file_path: Path, margin: float) -> Path:
    margin = str(margin).replace(".", "")
    return file_path.with_name(
        f"{file_path.stem}_recon_mesh_{margin}{file_path.suffix}"
    )


@app.command(no_args_is_help=True)
@measure_time
def evaluation(
    input_path: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    mesh_path: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    labels_path: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    x_range: tuple[float, float] = typer.Option((-1, 1)),
    y_range: tuple[float, float] = typer.Option((-1, 1)),
    z_range: tuple[float, float] = typer.Option((-1, 1)),
    margin: float = typer.Option(0.1),
    recon_mesh: bool = typer.Option(False),
) -> None:
    """Preprocess pointcloud.
    
    Example:
    uv run preprocessing \
        --input-path=<path_to_data>/pc_aligned.ply \
        --mesh-path=<path_to_data>/mesh_aligned_0.05.ply \
        --labels-path=<path_to_data>/segments_anno.json \
        --x-range 0 1 \
        --y-range -0.2 1 \
        --z-range -1 0.2 \
        --margin 0.1
    """

    logging.stage("Running preprocessing.")
    logging.info(f"Pointcloud: {input_path}")
    logging.info(f"Reference mesh: {mesh_path}")
    logging.info(f"Reference labels: {labels_path}")
    logging.info(f"Thresholds: x: {x_range}, y: {y_range}, z: {z_range}")
    logging.info(f"Margin: {margin}")

    # Load and normalize pointcloud.
    data = load_ply(input_path)
    points, normals, colors = data["points"], data["normals"], data["colors"]

    # Deduce labels from the reference annotated mesh.
    labels = compute_labels(points, mesh_path, labels_path)

    # Fit the scene into a bounding box for thresholds.
    points_norm = normalize_pointcloud(points)

    # Filter the pointcloud by thresholds.
    thresholds = (x_range, y_range, z_range)
    points_filtered, normals_final, colors_final, labels_final = filter_pointcloud(
        points_norm, normals, colors, labels, thresholds=thresholds, random_seed=42
    )

    # Normalize the pointcloud with a margin.
    points_final = normalize_pointcloud_with_margin(points_filtered, margin=margin)

    # Save the pointcloud.
    output_path = generate_output_path(input_path, margin)
    write_ply(output_path, points_final, normals_final, colors_final, labels_final)

    if recon_mesh:

        logging.info("Generating reconstruction mesh. It takes 15+ minutes.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_final)
        pcd.normals = o3d.utility.Vector3dVector(normals_final)
        pcd.colors = o3d.utility.Vector3dVector(colors_final)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=10, width=0, scale=1.1, linear_fit=False
        )

        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.01)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        output_recon_mesh_path = generate_output_recon_mesh_path(input_path, margin)
        write_ply(
            output_recon_mesh_path,
            points=np.array(mesh.vertices),
            normals=np.array(mesh.vertex_normals),
            faces=np.array(mesh.triangles),
        )


if __name__ == "__main__":
    app()
