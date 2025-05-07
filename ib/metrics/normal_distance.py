"""Normal distance metric."""

from pathlib import Path
from typing import Optional
from typing_extensions import deprecated

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_pointcloud, write_ply
from ib.utils.geometry import sdf_to_pointcloud, sparse_sdf_to_sdf_volume
from ib.utils.logging_module import logging
from ib.utils.pointcloud import filter_incorrect_normals


class NormalCosineSimilarity:
    """Bidirectional Normal Cosine Similarity metric."""

    def __init__(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        num_points: int,
        labels: np.ndarray = None,
    ):
        # Sample uniformly.
        num_points = min(num_points, len(vertices))
        indices = np.random.choice(len(vertices), num_points, replace=False)
        self.vertices = vertices[indices]
        self.normals = normals[indices]
        self.labels = labels[indices] if labels is not None else None
        self.tree = KDTree(self.vertices)

    @classmethod
    def from_pointcloud_path(cls, pointcloud_path: Path, num_points: int):
        data = load_pointcloud(pointcloud_path)
        vertices, normals, labels = filter_incorrect_normals(
            data["points"], data["normals"], data["labels"]
        )
        return cls(vertices, normals, num_points, labels)

    @deprecated("No longer used.")
    @classmethod
    def from_sdf_path(cls, sdf_path: Path, num_points: int):
        sdf = np.load(sdf_path)
        vertices, normals = sdf_to_pointcloud(sdf, num_points)
        return cls(vertices, normals, num_points)

    @deprecated("No longer used.")
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
        colors = (norm_error.reshape(-1, 1) * [255, 0, 0]).astype(np.uint8)
        # Write the PLY file.
        write_ply(save_path, pred_vertices, pred_normals, colors)
        logging.info(f"Saved pointcloud PLY to {save_path}")

    def __call__(
        self,
        pred_vertices: np.ndarray,
        pred_normals: np.ndarray,
        radius: float = 0.0025,
        save_path: Optional[Path] = None,
    ) -> dict[str, float]:
        pred_tree = KDTree(pred_vertices)

        # Compare with the best normal in some radius.
        radius_sims_t2p = self._compute_directional_similarity_radius(
            self.tree, self.normals, pred_tree, pred_normals, radius
        )
        radius_sims_p2t = self._compute_directional_similarity_radius(
            pred_tree, pred_normals, self.tree, self.normals, radius
        )
        # Compare with the closest normal.
        closest_sims_t2p = self._compute_directional_similarity_closest(
            self.vertices, self.normals, pred_tree, pred_normals
        )
        closest_sims_p2t = self._compute_directional_similarity_closest(
            pred_vertices, pred_normals, self.tree, self.normals
        )

        # Take mean of both directions.
        radius_sims = (radius_sims_p2t.mean() + radius_sims_t2p.mean()) / 2.0
        closest_sims = (closest_sims_p2t.mean() + closest_sims_t2p.mean()) / 2.0

        if save_path is not None:
            self.visualize_and_save(
                pred_vertices,
                pred_normals,
                closest_sims_p2t,
                save_path,
            )

        results = {
            "metrics/normal_similarity_radius": float(radius_sims),
            "metrics/normal_similarity_closest": float(closest_sims),
        }

        if self.labels is not None:
            mask_self = self.labels > 0
            _, idx_pred = self.tree.query(pred_vertices, workers=-1)
            mask_pred = self.labels[idx_pred] > 0

            # Low-resolution regions.
            low_res_t2p = closest_sims_t2p[mask_self].mean()
            low_res_p2t = closest_sims_p2t[mask_pred].mean()
            results["metrics/normal_similarity_closest_low_res_t2p"] = float(
                low_res_t2p
            )
            results["metrics/normal_similarity_closest_low_res_p2t"] = float(
                low_res_p2t
            )
            results["metrics/normal_similarity_closest_low_res"] = float(
                (low_res_t2p + low_res_p2t) / 2.0
            )

            # Other regions.
            mask_self_others = ~mask_self
            mask_pred_others = ~mask_pred
            others_t2p = closest_sims_t2p[mask_self_others].mean()
            others_p2t = closest_sims_p2t[mask_pred_others].mean()
            results["metrics/normal_similarity_closest_others_t2p"] = float(others_t2p)
            results["metrics/normal_similarity_closest_others_p2t"] = float(others_p2t)
            results["metrics/normal_similarity_closest_others"] = float(
                (others_t2p + others_p2t) / 2.0
            )

        return results
