"""Chamfer Distance metric."""

from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_xyz, normalize_points_and_normals


class ChamferDistance:
    """Bidirectional Chamfer Distance metric."""

    def __init__(self, pointcloud_path: Path, num_points=1_000_000) -> None:
        # Load the data.
        vertices, normals = load_xyz(pointcloud_path)

        # Normalize the same way as in the data loader.
        vertices, _ = normalize_points_and_normals(vertices, normals)

        # Sample uniformly.
        num_points = min(num_points, len(vertices))
        indices = np.random.choice(len(vertices), num_points, replace=False)
        self.vertices = vertices[indices]

        # Normalize and create a KDTree.
        self.tree = KDTree(self.vertices)

    def __call__(self, other_vertices: np.ndarray) -> float:
        dist_lhs = self.tree.query(other_vertices, workers=-1)[0]
        other_tree = KDTree(other_vertices)
        dist_rhs = other_tree.query(self.vertices, workers=-1)[0]
        return float((dist_lhs.mean() + dist_rhs.mean()) / 2)

    def gt_size(self):
        return len(self.vertices)
