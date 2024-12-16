"""Data related utils."""

from typing import Tuple

import numpy as np


def normalize_points_and_normals(
    points: np.ndarray,
    normals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalizes points and normals.

    Args:
        points (np.ndarray): Array of points (x, y, z).
        normals (np.ndarray): Array of normals (x, y, z).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - points (np.ndarray): Points in the (-1, 1) range with the original aspect ratio.
            - normals (np.ndarray): Unit normals.
    """
    points -= np.mean(points, axis=0, keepdims=True)
    coord_max = np.amax(points)
    coord_min = np.amin(points)
    points = (points - coord_min) / (coord_max - coord_min)
    points -= 0.5
    points *= 2.0

    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    return points, normals
