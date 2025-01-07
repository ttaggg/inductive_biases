"""Chamfer Distance metric."""

from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_xyz


def normalize_and_center(vertices: np.ndarray) -> np.ndarray:
    """Normalize and center points at midpoint.

    # TODO(oleg): consider moving to a better solution.
    """

    min_point = np.min(vertices, axis=0)
    max_point = np.max(vertices, axis=0)
    midpoint = (min_point + max_point) / 2
    centered_vertices = vertices - midpoint

    max_distance = np.linalg.norm(centered_vertices, axis=-1).max()
    normalized_vertices = centered_vertices / max_distance

    return normalized_vertices


class ChamferDistance:
    """Chamfer Distance metric."""

    def __init__(self, pointcloud_path: Path) -> None:
        vertices, _ = load_xyz(pointcloud_path)
        self.vertices = normalize_and_center(vertices)
        self.tree = KDTree(self.vertices)

    def __call__(self, other_vertices: np.ndarray) -> float:
        other_vertices = normalize_and_center(other_vertices)
        dist = self.tree.query(other_vertices)[0]
        return float(dist.mean())
