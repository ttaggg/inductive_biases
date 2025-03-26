"""Normal distance metric."""

from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_pointcloud, normalize_points_and_normals
from ib.utils.geometry import sdf_to_pointcloud


class NormalCosineSimilarity:
    """Bidirectional Normal Cosine Similarity metric."""

    def __init__(self, vertices: np.ndarray, normals: np.ndarray, num_points: int):
        # Sample uniformly.
        num_points = min(num_points, len(vertices))
        indices = np.random.choice(len(vertices), num_points, replace=False)
        self.vertices = vertices[indices]
        self.normals = normals[indices]
        self.tree = KDTree(self.vertices)

    @classmethod
    def from_pointcloud_path(cls, pointcloud_path: Path, num_points: int):
        vertices, normals = load_pointcloud(pointcloud_path)
        vertices, normals = normalize_points_and_normals(vertices, normals)
        return cls(vertices, normals, num_points)

    @classmethod
    def from_sdf_path(cls, sdf_path: Path, num_points: int):
        sdf = np.load(sdf_path)
        vertices, normals = sdf_to_pointcloud(sdf, num_points)
        return cls(vertices, normals, num_points)

    def _compute_directional_error_tree(
        self,
        source_tree: KDTree,
        source_normals: np.ndarray,
        target_tree: KDTree,
        target_normals: np.ndarray,
        radius: float,
    ) -> float:
        # NOTE(oleg): numba is 1.5 times faster, and same memory
        # NOTE(oleg): query_ball_point is 2 times slower,
        # but does not require a lot of RAM
        neighbor_indices: list[list] = source_tree.query_ball_tree(
            target_tree, r=radius
        )
        errors = np.ones(len(source_normals))
        for i, indices in enumerate(neighbor_indices):
            if len(indices) != 0:
                sims = np.dot(target_normals[indices], source_normals[i])
                best_similarity = np.max(sims)
                errors[i] = 1 - best_similarity
        return errors.mean()

    def _compute_directional_error_closest(
        self,
        source_vertices: np.ndarray,
        source_normals: np.ndarray,
        target_tree: KDTree,
        target_normals: np.ndarray,
    ) -> float:
        _, indices = target_tree.query(source_vertices, k=1, workers=-1)
        sims = np.sum(target_normals[indices] * source_normals, axis=1)
        errors = 1 - sims
        return errors.mean()

    def __call__(
        self,
        pred_vertices: np.ndarray,
        pred_normals: np.ndarray,
        radius: float = 0.003,
    ) -> dict:
        pred_tree = KDTree(pred_vertices)

        # Compare with the best normal in some radius.
        error_target_to_pred = self._compute_directional_error_tree(
            self.tree, self.normals, pred_tree, pred_normals, radius
        )
        error_pred_to_target = self._compute_directional_error_tree(
            pred_tree, pred_normals, self.tree, self.normals, radius
        )
        avg_error_radius = (error_target_to_pred + error_pred_to_target) / 2

        # Compare with the closest normal.
        error_target_to_pred = self._compute_directional_error_closest(
            self.vertices, self.normals, pred_tree, pred_normals
        )
        error_pred_to_target = self._compute_directional_error_closest(
            pred_vertices, pred_normals, self.tree, self.normals
        )
        avg_error_closest = (error_target_to_pred + error_pred_to_target) / 2

        return {
            f"metrics/normal_error_radius": float(avg_error_radius),
            "metrics/normal_error_closest": float(avg_error_closest),
        }
