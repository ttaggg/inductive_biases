"""Curvature Normal Change Rate."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from ib.utils.data import load_pointcloud
from ib.utils.pointcloud import filter_incorrect_normals


class CurvatureNormalChangeRate:
    """Curvature Normal Change Rate."""

    def __init__(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        labels: np.ndarray | None,
    ):
        np.random.seed(42)
        self.vertices = vertices
        self.normals = normals
        self.labels = labels
        self.tree = KDTree(self.vertices)

    @classmethod
    def from_pointcloud(
        cls,
        target_vertices: np.ndarray,
        target_normals: np.ndarray,
        target_labels: np.ndarray | None,
    ):
        vertices, normals, labels = filter_incorrect_normals(
            target_vertices, target_normals, target_labels
        )
        return cls(vertices, normals, labels)

    @classmethod
    def from_pointcloud_path(cls, pointcloud_path: Path):
        data = load_pointcloud(pointcloud_path)
        vertices, normals, labels = filter_incorrect_normals(
            data["points"], data["normals"], data["labels"]
        )
        return cls(vertices, normals, labels)

    def _filter_by_label(
        self,
        points: np.ndarray,
        normals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Deduce labels by the closest reference point.
        _, idx_pred = self.tree.query(points, k=1, workers=-1)
        mask_low_freq = self.labels[idx_pred] > 0
        return points[mask_low_freq], normals[mask_low_freq]

    def _filter_by_normal_direction(
        self,
        points: np.ndarray,
        normals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Only use points with outside-pointed normals.
        outside_pt = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
        center = points.mean(axis=0)
        to_outside = np.reshape(outside_pt - center, (1, 3))
        sign = np.einsum("ij,ij->i", normals, to_outside)
        points = points[sign > 0]
        normals = normals[sign > 0]
        return points, normals

    def _calculate_curvature_metric(self, points: np.ndarray) -> float:
        # Center the points
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        # Calculate covariance matrix
        cov_matrix = np.cov(centered_points.T)
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(eigenvalues)

        if np.sum(eigenvalues) == 0:
            return np.nan

        # Curvature metric: min(eigenvalues) / sum(eigenvalues)
        curvature_metric = eigenvalues[0] / np.sum(eigenvalues)
        return curvature_metric

    def __call__(
        self,
        pred_vertices: np.ndarray,
        pred_normals: np.ndarray,
        radius: float = 0.03,
        min_neighbors: int = 10,
        num_points: int = 100_000,
    ) -> dict[str, float]:

        pred_vertices, pred_normals = self._filter_by_normal_direction(
            pred_vertices,
            pred_normals,
        )
        pred_vertices, pred_normals = self._filter_by_label(
            pred_vertices,
            pred_normals,
        )
        pred_tree = KDTree(pred_vertices)

        # Cannot evaluate for every point, so sample a subset.
        # Sequence:
        # 1. Sample a subset of points in predicted pointcloud.
        # 2. For each point in the predicted subset, find the closest point in the target pointcloud.
        # 3. Compute the curvature metric for each pair of points.
        # 4. Compute the difference between curvatures in target and predicted.
        sample_size = min(num_points, len(pred_vertices))
        rand_inx = np.random.randint(0, len(pred_vertices), size=sample_size)
        center_pred_vertices = pred_vertices[rand_inx]
        _, indices = self.tree.query(center_pred_vertices, k=1, workers=-1)
        center_target_vertices = self.vertices[indices]

        curvature_metrics = np.ones(sample_size) * np.nan
        for i, (center_pred, center_target) in tqdm(
            enumerate(zip(center_pred_vertices, center_target_vertices)),
            total=len(center_pred_vertices),
            desc="Compute curvature metrics",
            unit=" points",
            dynamic_ncols=True,
            disable=not sys.stdout.isatty(),
        ):
            pred_neighbor_indices = pred_tree.query_ball_point(center_pred, radius)
            target_neighbor_indices = self.tree.query_ball_point(center_target, radius)

            if (
                len(pred_neighbor_indices) < min_neighbors
                or len(target_neighbor_indices) < min_neighbors
            ):
                continue

            pred_neighbor_points = pred_vertices[pred_neighbor_indices]
            pred_curvature_metric = self._calculate_curvature_metric(
                pred_neighbor_points
            )
            target_neighbor_points = self.vertices[target_neighbor_indices]
            target_curvature_metric = self._calculate_curvature_metric(
                target_neighbor_points
            )
            curvature_metrics[i] = np.abs(
                pred_curvature_metric - target_curvature_metric
            )

        return {
            "curvature_mean": float(np.nanmean(curvature_metrics)),
            "curvature_median": float(np.nanmedian(curvature_metrics)),
        }
