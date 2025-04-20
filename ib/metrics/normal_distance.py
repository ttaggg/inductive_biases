"""Normal distance metric."""

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_pointcloud, normalize_points_and_normals, write_ply
from ib.utils.geometry import sdf_to_pointcloud, sparse_sdf_to_sdf_volume
from ib.utils.logging_module import logging


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

    @classmethod
    def from_sparse_sdf_path(cls, sparse_sdf_path: Path, num_points: int):
        sparse_data = np.load(sparse_sdf_path)
        surface_coords = sparse_data["coords"].astype(np.int32)
        surface_sdf = sparse_data["sdf"].astype(np.float32).reshape(-1, 1)
        resolution = sparse_data.get("resolution", 1024)
        sdf = sparse_sdf_to_sdf_volume(surface_coords, surface_sdf, resolution)
        vertices, normals = sdf_to_pointcloud(sdf, num_points)
        return cls(vertices, normals, num_points)

    def _compute_directional_similarity_radius(
        self,
        source_tree: KDTree,
        source_normals: np.ndarray,
        target_tree: KDTree,
        target_normals: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        # NOTE(oleg): numba is 1.5 times faster, and same memory
        # NOTE(oleg): query_ball_point is 2 times slower,
        # but does not require a lot of RAM
        neighbor_indices: list[list] = source_tree.query_ball_tree(
            target_tree, r=radius
        )
        sims = -1 * np.ones(len(source_normals))
        for i, indices in enumerate(neighbor_indices):
            if len(indices) != 0:
                similarities = np.dot(target_normals[indices], source_normals[i])
                sims[i] = np.max(similarities)
        return sims

    def _compute_directional_similarity_closest(
        self,
        source_vertices: np.ndarray,
        source_normals: np.ndarray,
        target_tree: KDTree,
        target_normals: np.ndarray,
    ) -> np.ndarray:
        _, indices = target_tree.query(source_vertices, k=1, workers=-1)
        sims = np.sum(target_normals[indices] * source_normals, axis=1)
        return sims

    def visualize_and_save(
        self,
        pred_vertices: np.ndarray,
        pred_normals: np.ndarray,
        sims: np.ndarray,
        save_path: Path,
    ) -> None:
        # Get the error in [0, 1] range.
        norm_sims = (np.clip(sims, -1, 1) + 1.0) / 2.0
        # Only visualize points with low similarity.
        mask = norm_sims < 0.5
        norm_sims = norm_sims[mask]
        pred_vertices = pred_vertices[mask]
        pred_normals = pred_normals[mask]
        norm_error = 1 - norm_sims
        alpha_channel = (norm_error * 255).astype(np.uint8)
        # Write the PLY file.
        write_ply(save_path, pred_vertices, pred_normals, alpha_channel)
        logging.info(f"Saved pointcloud PLY to {save_path}")

    def __call__(
        self,
        pred_vertices: np.ndarray,
        pred_normals: np.ndarray,
        radius: float = 0.0025,
        save_path: Optional[Path] = None,
    ) -> dict:
        pred_tree = KDTree(pred_vertices)

        # Compare with the best normal in some radius.
        radius_sims_target_to_pred = self._compute_directional_similarity_radius(
            self.tree, self.normals, pred_tree, pred_normals, radius
        )
        radius_sims_pred_to_target = self._compute_directional_similarity_radius(
            pred_tree, pred_normals, self.tree, self.normals, radius
        )
        # Compare with the closest normal.
        closest_sims_target_to_pred = self._compute_directional_similarity_closest(
            self.vertices, self.normals, pred_tree, pred_normals
        )
        closest_sims_pred_to_target = self._compute_directional_similarity_closest(
            pred_vertices, pred_normals, self.tree, self.normals
        )

        # Take mean of both directions.
        radius_sims = (
            radius_sims_pred_to_target.mean() + radius_sims_target_to_pred.mean()
        ) / 2.0
        closest_sims = (
            closest_sims_pred_to_target.mean() + closest_sims_target_to_pred.mean()
        ) / 2.0

        if save_path is not None:
            self.visualize_and_save(
                pred_vertices,
                pred_normals,
                closest_sims_pred_to_target,
                save_path,
            )

        return {
            "metrics/normal_similarity_radius": float(radius_sims),
            "metrics/normal_similarity_closest": float(closest_sims),
        }
