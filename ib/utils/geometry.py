"""Utils to convert betweeen data types."""

from typing_extensions import deprecated

import numpy as np
import open3d as o3d
from skimage import measure

from ib.datasets.resamplers import Resampler


def sdf_to_mesh(sdf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resolution = sdf.shape[0]  # Assume cube.
    spacing = (1.0 / resolution, 1.0 / resolution, 1.0 / resolution)
    verts, faces, _, _ = measure.marching_cubes(sdf, level=0, spacing=spacing)
    verts = 2 * verts - 1
    return verts, faces


def mesh_to_pointcloud(
    verts: np.ndarray,
    faces: np.ndarray,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    resampler = Resampler(verts, faces)
    resampler.run(num_samples=num_points)
    return resampler.sampled_vertices, resampler.sampled_normals


@deprecated("No longer used.")
def sdf_to_pointcloud(
    sdf: np.ndarray,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = sdf_to_mesh(sdf)
    vertices, normals = mesh_to_pointcloud(verts, faces, num_points)
    return vertices, normals


@deprecated("No longer used.")
def sparse_sdf_to_sdf_volume(
    surface_coords: np.ndarray,
    surface_sdf: np.ndarray,
    resolution: int,
) -> np.ndarray:
    surface_sdf = surface_sdf.flatten()
    max_val = np.max(surface_sdf)
    sdf = max_val * np.ones((resolution, resolution, resolution))
    sdf[surface_coords[:, 0], surface_coords[:, 1], surface_coords[:, 2]] = surface_sdf
    return sdf


def orient_mesh(
    mesh: o3d.geometry.TriangleMesh,
    outside_pt: list = [0.0, 0.0, 0.0],
) -> None:

    outside_pt = np.array(outside_pt, dtype=np.float32)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    centres = verts[tris].mean(axis=1)

    tri_normals = np.asarray(mesh.triangle_normals)
    to_outside = centres - outside_pt

    sign = np.einsum("ij,ij->i", tri_normals, to_outside)

    if sign.mean() > 0:
        tris_flipped = tris.copy()
        tris_flipped[:, [1, 2]] = tris_flipped[:, [2, 1]]
        mesh.triangles = o3d.utility.Vector3iVector(tris_flipped)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()


def color_mesh_to_normal_direction(mesh: o3d.geometry.TriangleMesh) -> None:

    # Check if vertex normals exist and are valid
    if len(mesh.vertex_normals) == 0 or len(mesh.vertex_normals) != len(mesh.vertices):
        mesh.compute_vertex_normals()

    # Set the normals to unit length
    normals = np.array(mesh.vertex_normals)
    norms = np.linalg.norm(normals, axis=1)
    if np.any(norms == 0) or np.any(np.isnan(norms)):
        mesh.compute_vertex_normals()
        normals = np.array(mesh.vertex_normals)
        norms = np.linalg.norm(normals, axis=1)

    normals = normals / norms[:, None]
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    # Set the color
    normals = np.array(mesh.vertex_normals)
    normal_colors = (normals + 1.0) / 2.0
    mesh.vertex_colors = o3d.utility.Vector3dVector(normal_colors)
