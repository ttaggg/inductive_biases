"""Combine Chamfer and Normal distance metrics to reuse results."""

from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_pointcloud
from ib.utils.logging_module import logging
from ib.utils.pointcloud import filter_incorrect_normals


def compute_bidirectional_nn(
    pred_vertices: np.ndarray,
    target_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bidirectional nearest neighbor correspondences between source and target vertices.

    Args:
        pred_vertices (np.ndarray): Predicted point cloud vertices of shape (N, D).
        target_vertices (np.ndarray): Target point cloud vertices of shape (M, D).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            dist_p2t: Distances from each predicted point to its nearest target neighbor.
            idx_p2t: Indices of target neighbors for each predicted point.
            dist_t2p: Distances from each target point to its nearest predicted neighbor.
            idx_t2p: Indices of predicted neighbors for each target point.
    """
    # Build KD-tree for target vertices and query from source.
    target_tree = KDTree(target_vertices)
    dist_p2t, idx_p2t = target_tree.query(pred_vertices, workers=-1)
    # Build KD-tree for predicted vertices and query from target.
    pred_tree = KDTree(pred_vertices)
    dist_t2p, idx_t2p = pred_tree.query(target_vertices, workers=-1)
    return dist_p2t, idx_p2t, dist_t2p, idx_t2p


class CombinedPointcloudMetric:
    """Combined Chamfer and Normal distance metrics to reuse results."""

    def __init__(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        labels: np.ndarray | None,
        num_points: int,
    ):
        # Sample uniformly.
        num_points = min(num_points, len(vertices))
        indices = np.random.choice(len(vertices), num_points, replace=True)
        self.vertices = vertices[indices]
        self.normals = normals[indices]
        self.labels = labels[indices] if labels is not None else None

    @classmethod
    def from_pointcloud_path(cls, pointcloud_path: Path, num_points: int):
        data = load_pointcloud(pointcloud_path)
        vertices, normals, labels = filter_incorrect_normals(
            data["points"], data["normals"], data["labels"]
        )
        return cls(vertices, normals, labels, num_points)

    def _compute_directional_similarity_radius(
        self,
        source_vertices: np.ndarray,
        source_normals: np.ndarray,
        target_vertices: np.ndarray,
        target_normals: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        # NOTE(oleg): numba is 1.5 times faster, and same memory
        # NOTE(oleg): query_ball_point is 2 times slower,
        # but does not require a lot of RAM
        source_tree = KDTree(source_vertices)
        target_tree = KDTree(target_vertices)
        neighbor_indices: list[list] = source_tree.query_ball_tree(
            target_tree, r=radius
        )
        sims = np.nan * np.ones(len(source_normals))
        for i, indices in enumerate(neighbor_indices):
            if len(indices) != 0:
                similarities = np.dot(target_normals[indices], source_normals[i])
                sims[i] = np.max(similarities)
        return sims

    def __call__(
        self,
        pred_vertices: np.ndarray,
        pred_normals: np.ndarray,
        radius: float = 0.005,
    ) -> dict[str, float]:

        logging.info(
            f"Combined metric: target size is {len(self.vertices):_}, "
            f"pred size is {len(pred_vertices):_}."
        )
        dist_p2t, idx_p2t, dist_t2p, idx_t2p = compute_bidirectional_nn(
            pred_vertices, self.vertices
        )
        results = {}

        # Chamfer metric.
        results["metrics/chamfer_p2t"] = float(dist_p2t.mean())
        results["metrics/chamfer_t2p"] = float(dist_t2p.mean())

        overall = (dist_p2t.mean() + dist_t2p.mean()) / 2.0
        results["metrics/chamfer"] = float(overall)

        if self.labels is not None:
            mask_self = self.labels > 0

            # Low-resolution regions.
            low_res_t2p = dist_t2p[mask_self].mean()
            low_res_p2t = dist_p2t[mask_self[idx_p2t]].mean()
            low_res = (low_res_t2p + low_res_p2t) / 2.0
            results["metrics/chamfer_low_res_t2p"] = float(low_res_t2p)
            results["metrics/chamfer_low_res_p2t"] = float(low_res_p2t)
            results["metrics/chamfer_low_res"] = float(low_res)

            # Other regions.
            mask_others = ~mask_self
            others_t2p = dist_t2p[mask_others].mean()
            others_p2t = dist_p2t[mask_others[idx_p2t]].mean()
            others = (others_t2p + others_p2t) / 2.0
            results["metrics/chamfer_others_t2p"] = float(others_t2p)
            results["metrics/chamfer_others_p2t"] = float(others_p2t)
            results["metrics/chamfer_others"] = float(others)

        # Normal similarity metric.
        # Compare with the closest normal.
        logging.info(
            f"self.normals: {len(self.normals):_}, idx_p2t {len(idx_p2t):_}, "
            f" pred_normals {len(pred_normals):_}, idx_t2p {len(idx_t2p):_}"
        )
        closest_sims_p2t = np.sum(self.normals[idx_p2t] * pred_normals, axis=1)
        closest_sims_t2p = np.sum(pred_normals[idx_t2p] * self.normals, axis=1)
        results["metrics/normal_similarity_closest_t2p"] = float(
            closest_sims_t2p.mean()
        )
        results["metrics/normal_similarity_closest_p2t"] = float(
            closest_sims_p2t.mean()
        )
        closest_sims = (closest_sims_p2t.mean() + closest_sims_t2p.mean()) / 2.0
        results["metrics/normal_similarity_closest"] = float(closest_sims)

        # Compare with the normal in some radius.
        radius_sims_t2p = self._compute_directional_similarity_radius(
            self.vertices, self.normals, pred_vertices, pred_normals, radius
        )
        radius_sims_p2t = self._compute_directional_similarity_radius(
            pred_vertices, pred_normals, self.vertices, self.normals, radius
        )
        results["metrics/normal_similarity_radius_t2p"] = float(
            np.nanmean(radius_sims_t2p)
        )
        results["metrics/normal_similarity_radius_p2t"] = float(
            np.nanmean(radius_sims_p2t)
        )
        radius_sims = (np.nanmean(radius_sims_p2t) + np.nanmean(radius_sims_t2p)) / 2.0
        results["metrics/normal_similarity_radius"] = float(radius_sims)

        # Closest with labels.
        if self.labels is not None:
            mask_self = self.labels > 0
            mask_pred = self.labels[idx_p2t] > 0

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

        # Radius with labels.
        if self.labels is not None:
            mask_self = self.labels > 0
            mask_pred = self.labels[idx_p2t] > 0

            # Low-resolution regions.
            low_res_t2p = np.nanmean(radius_sims_t2p[mask_self])
            low_res_p2t = np.nanmean(radius_sims_p2t[mask_pred])
            results["metrics/normal_similarity_radius_low_res_t2p"] = float(low_res_t2p)
            results["metrics/normal_similarity_radius_low_res_p2t"] = float(low_res_p2t)
            results["metrics/normal_similarity_radius_low_res"] = float(
                (low_res_t2p + low_res_p2t) / 2.0
            )

            # Other regions.
            mask_self_others = ~mask_self
            mask_pred_others = ~mask_pred
            others_t2p = np.nanmean(radius_sims_t2p[mask_self_others])
            others_p2t = np.nanmean(radius_sims_p2t[mask_pred_others])
            results["metrics/normal_similarity_radius_others_t2p"] = float(others_t2p)
            results["metrics/normal_similarity_radius_others_p2t"] = float(others_p2t)
            results["metrics/normal_similarity_radius_others"] = float(
                (others_t2p + others_p2t) / 2.0
            )

        return results
