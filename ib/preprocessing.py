from pathlib import Path
from typing import Annotated

import typer

from ib.utils.data import load_ply, write_ply
from ib.utils.labels import compute_labels
from ib.utils.logging_module import logging
from ib.utils.pipeline import measure_time, resolve_and_expand_path
from ib.utils.pointcloud import (
    normalize_pointcloud_and_mesh,
    normalize_pointcloud_and_mesh_with_margin,
    filter_pointcloud_and_mesh,
)

app = typer.Typer(add_completion=False)


def generate_output_pc_path(file_path: Path, margin: float) -> Path:
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
def preprocessing(
    input_path: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    x_range: tuple[float, float] = typer.Option((-1, 1)),
    y_range: tuple[float, float] = typer.Option((-1, 1)),
    z_range: tuple[float, float] = typer.Option((-1, 1)),
    margin: float = typer.Option(0.1),
) -> None:
    """Preprocess pointcloud.
    
    Example:
    uv run preprocessing \
        --input-path=<path_to_data>/pc_aligned.ply \
        --x-range 0 1 \
        --y-range -0.2 1 \
        --z-range -1 0.2 \
        --margin 0.1
    """

    logging.stage("Running preprocessing.")

    labels_path = input_path.parent / "segments_anno.json"
    labels_mesh_path = input_path.parent / "mesh_aligned_0.05.ply"
    recon_mesh_path = input_path.parent / "simplified_0.1250_mesh_aligned.ply"

    logging.info(f"Pointcloud: {input_path}")
    logging.info(f"Reconstructed mesh: {recon_mesh_path}")
    logging.info(f"Reference mesh: {labels_mesh_path}")
    logging.info(f"Reference labels: {labels_path}")
    logging.info(f"Thresholds: x: {x_range}, y: {y_range}, z: {z_range}")
    logging.info(f"Margin: {margin}")

    # Load and normalize pointcloud.
    data = load_ply(input_path)
    points, normals, colors = data["points"], data["normals"], data["colors"]

    # Deduce labels from the reference annotated mesh.
    labels = compute_labels(points, labels_mesh_path, labels_path)

    # Load the reconstruction mesh.
    recon_mesh = load_ply(recon_mesh_path)

    # Fit the scene into a bounding box for thresholds.
    points_norm, recon_mesh["points"] = normalize_pointcloud_and_mesh(
        points,
        recon_mesh["points"],
    )

    # Filter the pointcloud by thresholds.
    thresholds = (x_range, y_range, z_range)
    points_filtered, normals_final, colors_final, labels_final, recon_mesh = (
        filter_pointcloud_and_mesh(
            points_norm,
            normals,
            colors,
            labels,
            recon_mesh,
            thresholds=thresholds,
        )
    )

    # Normalize the pointcloud with a margin.
    points_final, recon_mesh["points"] = normalize_pointcloud_and_mesh_with_margin(
        points_filtered,
        recon_mesh["points"],
        margin=margin,
    )

    # Save the pointcloud.
    output_pc_path = generate_output_pc_path(input_path, margin)
    write_ply(output_pc_path, points_final, normals_final, colors_final, labels_final)
    output_mesh_path = generate_output_recon_mesh_path(input_path, margin)
    write_ply(
        output_mesh_path,
        recon_mesh["points"],
        faces=recon_mesh["faces"],
    )


if __name__ == "__main__":
    app()
