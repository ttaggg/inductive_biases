"""Chamfer Distance metric."""

from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_pointcloud, normalize_points_and_normals
from ib.utils.geometry import sdf_to_pointcloud


class ChamferDistance:
    """Bidirectional Chamfer Distance metric."""

    def __init__(self, vertices: np.ndarray, num_points: int) -> None:
        # Sample uniformly.
        num_points = min(num_points, len(vertices))
        indices = np.random.choice(len(vertices), num_points, replace=False)
        self.vertices = vertices[indices]
        # Create a KDTree.
        self.tree = KDTree(self.vertices)

    @classmethod
    def from_pointcloud_path(cls, pointcloud_path: Path, num_points: int):
        vertices, normals = load_pointcloud(pointcloud_path)
        vertices, _ = normalize_points_and_normals(vertices, normals)
        return cls(vertices, num_points)

    @classmethod
    def from_sdf_path(cls, sdf_path: Path, num_points: int):
        sdf = np.load(sdf_path)
        vertices, _ = sdf_to_pointcloud(sdf, num_points)
        return cls(vertices, num_points)

    def __call__(self, other_vertices: np.ndarray) -> float:
        dist_lhs = self.tree.query(other_vertices, workers=-1)[0]
        other_tree = KDTree(other_vertices)
        dist_rhs = other_tree.query(self.vertices, workers=-1)[0]
        dist = (dist_lhs.mean() + dist_rhs.mean()) / 2
        return {"metrics/chamfer": float(dist)}
