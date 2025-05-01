"""Chamfer Distance metric."""

from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_pointcloud
from ib.utils.geometry import sdf_to_pointcloud, sparse_sdf_to_sdf_volume
from ib.utils.pointcloud import normalize_points_and_normals


class ChamferDistance:
    """Bidirectional Chamfer Distance metric."""

    def __init__(
        self,
        vertices: np.ndarray,
        num_points: int,
        labels: np.ndarray = None,
    ) -> None:
        # Sample uniformly.
        num_points = min(num_points, len(vertices))
        indices = np.random.choice(len(vertices), num_points, replace=False)
        self.vertices = vertices[indices]
        self.labels = labels[indices] if labels is not None else None
        # Create a KDTree.
        self.tree = KDTree(self.vertices)

    @classmethod
    def from_pointcloud_path(cls, pointcloud_path: Path, num_points: int):
        data = load_pointcloud(pointcloud_path)
        vertices, _ = normalize_points_and_normals(data["points"], data["normals"])
        labels = data.get("labels", None)
        return cls(vertices, num_points, labels)

    @classmethod
    def from_sdf_path(cls, sdf_path: Path, num_points: int):
        sdf = np.load(sdf_path)
        vertices, _ = sdf_to_pointcloud(sdf, num_points)
        return cls(vertices, num_points)

    @classmethod
    def from_sparse_sdf_path(cls, sparse_sdf_path: Path, num_points: int):
        sparse_data = np.load(sparse_sdf_path)
        surface_coords = sparse_data["coords"].astype(np.int32)
        surface_sdf = sparse_data["sdf"].astype(np.float32).reshape(-1, 1)
        resolution = sparse_data.get("resolution", 1024)
        sdf = sparse_sdf_to_sdf_volume(surface_coords, surface_sdf, resolution)
        vertices, _ = sdf_to_pointcloud(sdf, num_points)
        return cls(vertices, num_points)

    def __call__(self, other_vertices: np.ndarray) -> float:
        dist_lhs = self.tree.query(other_vertices, workers=-1)[0]
        other_tree = KDTree(other_vertices)
        dist_rhs = other_tree.query(self.vertices, workers=-1)[0]
        dist = (dist_lhs.mean() + dist_rhs.mean()) / 2
        results = {"metrics/chamfer": float(dist)}
        if self.labels is not None:
            mask = self.labels > 0
            results["metrics/chamfer_low_res"] = float(dist_rhs[mask].mean())
            results["metrics/chamfer_others"] = float(dist_rhs[~mask].mean())
        return results
