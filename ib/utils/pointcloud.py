"""Utils for pointclouds."""

from typing import Optional
from typing_extensions import deprecated

import numpy as np


def normalize_pointcloud(
    points: np.ndarray,
    bounds_min: np.ndarray = np.array([-1.0, -1.0, -1.0]),
    bounds_max: np.ndarray = np.array([1.0, 1.0, 1.0]),
    return_meta: bool = False,
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

    if return_meta:
        return (
            points_centered * s + target_center,
            {"scale": s, "target_center": target_center},
        )
    return points_centered * s + target_center


def normalize_pointcloud_with_margin(
    points: np.ndarray,
    bounds_min: np.ndarray = np.array([-1.0, -1.0, -1.0]),
    bounds_max: np.ndarray = np.array([1.0, 1.0, 1.0]),
    margin: float = 0.1,
) -> np.ndarray:
    size = bounds_max - bounds_min
    new_min = bounds_min + margin * size
    new_max = bounds_max - margin * size
    return normalize_pointcloud(points, bounds_min=new_min, bounds_max=new_max)


def filter_pointcloud(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    thresholds: tuple[tuple] = (
        (-1, 1),
        (-1, 1),
        (-1, 1),
    ),
    random_seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cut parts of the scene based on thresholds."""

    if random_seed is not None:
        np.random.seed(random_seed)

    # build boolean mask of points inside the threshold box
    mins = np.array([thresholds[0][0], thresholds[1][0], thresholds[2][0]])
    maxs = np.array([thresholds[0][1], thresholds[1][1], thresholds[2][1]])
    inside_mask = np.all((points >= mins) & (points <= maxs), axis=1)

    points = points[inside_mask]
    if normals is not None:
        normals = normals[inside_mask]
    if colors is not None:
        colors = colors[inside_mask]
    if labels is not None:
        labels = labels[inside_mask]

    return points, normals, colors, labels


def filter_incorrect_normals(
    points: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Filter out invalid normals and points with zero normals.
    correct_normals = np.logical_and(
        np.linalg.norm(normals, axis=-1) != 0.0,
        np.all(np.isfinite(normals), axis=-1),
    )
    return points[correct_normals], normals[correct_normals]
