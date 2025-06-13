"""Utils for pointclouds."""

from typing import Optional

import numpy as np


def normalize_pointcloud_and_mesh(
    points: np.ndarray,
    mesh_points: np.ndarray,
    bounds_min: np.ndarray = np.array([-1.0, -1.0, -1.0]),
    bounds_max: np.ndarray = np.array([1.0, 1.0, 1.0]),
) -> tuple[np.ndarray, dict]:
    """Linearly maps points to lie within [bounds_min, bounds_max]."""
    points_min = points.min(axis=0)
    points_max = points.max(axis=0)
    scale = (bounds_max - bounds_min) / (points_max - points_min)
    # Use uniform scale to preserve aspect ratio
    s = np.min(scale)
    # Center points around origin of target box
    points_centered = points - (points_min + points_max) / 2.0
    target_center = (bounds_min + bounds_max) / 2.0
    # Apply same transformation to both mesh and pointcloud.
    points_norm = points_centered * s + target_center
    mesh_points_centered = mesh_points - (points_min + points_max) / 2.0
    mesh_points_norm = mesh_points_centered * s + target_center
    return points_norm, mesh_points_norm


def normalize_pointcloud_and_mesh_with_margin(
    points: np.ndarray,
    mesh_points: np.ndarray,
    bounds_min: np.ndarray = np.array([-1.0, -1.0, -1.0]),
    bounds_max: np.ndarray = np.array([1.0, 1.0, 1.0]),
    margin: float = 0.1,
) -> np.ndarray:
    size = bounds_max - bounds_min
    new_min = bounds_min + margin * size
    new_max = bounds_max - margin * size
    return normalize_pointcloud_and_mesh(
        points,
        mesh_points,
        bounds_min=new_min,
        bounds_max=new_max,
    )


def filter_pointcloud_and_mesh(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    mesh: dict[str, np.ndarray],
    thresholds: tuple[tuple] = (
        (-1, 1),
        (-1, 1),
        (-1, 1),
    ),
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    """Cut parts of the scene based on thresholds."""

    # build boolean mask of points inside the threshold box
    mins = np.array([thresholds[0][0], thresholds[1][0], thresholds[2][0]])
    maxs = np.array([thresholds[0][1], thresholds[1][1], thresholds[2][1]])
    inside_mask_points = np.all((points >= mins) & (points <= maxs), axis=1)

    # Apply to the pointcloud.
    points = points[inside_mask_points]
    normals = normals[inside_mask_points]
    colors = colors[inside_mask_points]
    labels = labels[inside_mask_points]

    inside_mask_mesh = np.all(
        (mesh["points"] >= mins) & (mesh["points"] <= maxs), axis=1
    )

    # Apply to the mesh.
    mesh["points"] = mesh["points"][inside_mask_mesh]

    # Update mesh normals if they exist
    if "normals" in mesh:
        mesh["normals"] = mesh["normals"][inside_mask_mesh]

    # Update mesh colors if they exist
    if "colors" in mesh:
        mesh["colors"] = mesh["colors"][inside_mask_mesh]

    # Handle face filtering and vertex index remapping
    if "faces" in mesh:
        # Mapping from old vertex indices to new vertex indices
        old_to_new_idx = np.full(len(inside_mask_mesh), -1, dtype=int)
        old_to_new_idx[inside_mask_mesh] = np.arange(np.sum(inside_mask_mesh))

        # Only include faces where all vertices are present
        faces = mesh["faces"]
        valid_faces_mask = np.all(inside_mask_mesh[faces], axis=1)
        valid_faces = faces[valid_faces_mask]

        # Remap vertex indices in valid faces
        mesh["faces"] = old_to_new_idx[valid_faces]

    return points, normals, colors, labels, mesh


def filter_incorrect_normals(
    points: np.ndarray,
    normals: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Filter out invalid normals and points with zero normals.
    correct_normals = np.logical_and(
        np.linalg.norm(normals, axis=-1) != 0.0,
        np.all(np.isfinite(normals), axis=-1),
    )
    points = points[correct_normals]
    normals = normals[correct_normals]
    labels = labels[correct_normals] if labels is not None else None
    return points, normals, labels
