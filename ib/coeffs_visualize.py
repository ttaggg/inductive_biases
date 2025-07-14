"""Visualize coeffs coefficients from the SIREN-M."""

import sys
from pathlib import Path
from typing import Annotated

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import typer
import torch
import open3d as o3d
from scipy.spatial import KDTree
from tqdm import tqdm

from ib.utils.geometry import sdf_to_mesh
from ib.utils.logging_module import logging
from ib.utils.model import make_coordinates
from ib.utils.pipeline import measure_time, resolve_and_expand_path

app = typer.Typer(add_completion=False)


@torch.no_grad()
def query_model(
    model: L.LightningModule,
    resolution: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logging.stage("Querying the model.")

    grid_size = (resolution, resolution, resolution)
    coords_grid = make_coordinates(grid_size=grid_size)
    coords = torch.split(coords_grid, batch_size, dim=0)

    with tqdm(
        total=len(coords),
        desc="Query model",
        unit=" steps",
        dynamic_ncols=True,
        disable=not sys.stdout.isatty(),
    ) as pbar:

        combined_sdf = []
        combined_coeffs_a = []
        combined_coeffs_b = []
        combined_coeffs_c = []
        for batch_coords in coords:
            batch_coords = batch_coords.to(device)
            batch_sdf, batch_meta = model(batch_coords, return_meta=True)
            batch_sdf = batch_sdf.detach().cpu().numpy()
            batch_meta_a = torch.stack([batch_meta[f"a/{i}"] for i in range(4)])
            batch_coeffs_a = batch_meta_a.cpu().numpy().astype(np.float16)
            batch_meta_b = torch.stack([batch_meta[f"b/{i}"] for i in range(4)])
            batch_coeffs_b = batch_meta_b.cpu().numpy().astype(np.float16)
            batch_meta_c = torch.stack([batch_meta[f"c/{i}"] for i in range(4)])
            batch_coeffs_c = batch_meta_c.cpu().numpy().astype(np.float16)
            combined_sdf.append(batch_sdf)
            combined_coeffs_a.append(batch_coeffs_a)
            combined_coeffs_b.append(batch_coeffs_b)
            combined_coeffs_c.append(batch_coeffs_c)

            pbar.update(1)

    sdf = np.concatenate(combined_sdf, axis=0)
    sdf = sdf.reshape(grid_size)
    combined_coeffs_a = np.concatenate(combined_coeffs_a, axis=1)
    combined_coeffs_a = combined_coeffs_a.squeeze(-1)
    combined_coeffs_b = np.concatenate(combined_coeffs_b, axis=1)
    combined_coeffs_b = combined_coeffs_b.squeeze(-1)
    combined_coeffs_c = np.concatenate(combined_coeffs_c, axis=1)
    combined_coeffs_c = combined_coeffs_c.squeeze(-1)
    return sdf, combined_coeffs_a, combined_coeffs_b, combined_coeffs_c, coords_grid


def coeff_to_vertex_color(
    indices: np.ndarray,
    coeffs: np.ndarray,
    minmax: bool = False,
) -> np.ndarray:
    vertex_coeffs = coeffs[indices]
    if minmax:
        vertex_coeffs = vertex_coeffs - vertex_coeffs.min(axis=0, keepdims=True)
        vertex_coeffs = vertex_coeffs / vertex_coeffs.max(axis=0, keepdims=True)
    else:
        vertex_coeffs = vertex_coeffs - vertex_coeffs.max(axis=0, keepdims=True)
        vertex_coeffs += 1.0
    rgba = plt.cm.jet(vertex_coeffs)
    colors = rgba[:, :3]
    return colors


def plot_for_coeff(
    coeff: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    indices: np.ndarray,
    output_dir: Path,
    minmax: bool,
    i: int,
    lit: str,
):

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

    # Color vertices based on beta values
    colors = coeff_to_vertex_color(indices, coeff, minmax=minmax)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    logging.info(f"Coefficient statistics:")
    logging.info(f"  Min:  {coeff.min()}")
    logging.info(f"  Mean: {coeff.mean()}")
    logging.info(f"  Max:  {coeff.max()}")
    logging.info(f"  Std:  {coeff.std()}")
    logging.info(f"  Surface mean: {coeff[indices].mean()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"coeff_{minmax}_{lit}_{i}.ply"
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    logging.info(f"Mesh was written to {output_path}")


@app.command(no_args_is_help=True)
@measure_time
def coeffs_visualize(
    model_path: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    output_dir: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    mesh_resolution: int = 512,
    grid_resolution: int = 512,
    minmax: bool = False,
    batch_size: int = 100000,
    device: str = "cuda",
) -> None:
    """Visualize beta coefficients from a trained model."""

    logging.stage("Visualizing coefficients from the model.")

    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()
    model.to(device)

    # NOTE(oleg): 1024 grid is too large to use in KDTree.
    # Two forward rolls just to be absolutely sure about coordinates.
    logging.info("Querying the model for mesh.")
    sdf, _, _, _, _ = query_model(model, mesh_resolution, batch_size, device)
    logging.info("Running marching cubes.")
    verts, faces = sdf_to_mesh(sdf)
    logging.info("Querying the model for grid.")
    _, coefficients_a, coefficients_b, coefficients_c, grid_coords = query_model(
        model, grid_resolution, batch_size, device
    )

    logging.info("Creating a KDtree.")
    tree = KDTree(grid_coords)
    logging.info("Querying the KDtree.")
    _, indices = tree.query(verts, k=1, workers=-1)

    for i, coeff in enumerate(coefficients_a):
        plot_for_coeff(coeff, verts, faces, indices, output_dir, minmax, i, "a")

    for i, coeff in enumerate(coefficients_b):
        plot_for_coeff(coeff, verts, faces, indices, output_dir, minmax, i, "b")

    for i, coeff in enumerate(coefficients_c):
        plot_for_coeff(coeff, verts, faces, indices, output_dir, minmax, i, "c")


if __name__ == "__main__":
    app()
