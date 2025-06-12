"""Completeness."""

from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_pointcloud
from ib.utils.labels import LABELS
from ib.utils.pointcloud import filter_incorrect_normals


class Completeness:
    """Completeness."""

    def __init__(
        self,
        vertices: np.ndarray,
        labels: np.ndarray | None,
    ):
        self.vertices = vertices
        self.labels = labels
        self.tree = KDTree(self.vertices)

    @classmethod
    def from_pointcloud(
        cls,
        target_vertices: np.ndarray,
        target_normals: np.ndarray,
        target_labels: np.ndarray | None,
    ):
        vertices, _, labels = filter_incorrect_normals(
            target_vertices, target_normals, target_labels
        )
        return cls(vertices, labels)

    @classmethod
    def from_pointcloud_path(cls, pointcloud_path: Path):
        data = load_pointcloud(pointcloud_path)
        vertices, _, labels = filter_incorrect_normals(
            data["points"], data["normals"], data["labels"]
        )
        return cls(vertices, labels)

    def _compute_completeness(self, pred_vertices: KDTree, radius: float) -> np.ndarray:
        pred_tree = KDTree(pred_vertices)
        neighbor_indices: list[list] = self.tree.query_ball_tree(pred_tree, r=radius)
        match = np.zeros(len(pred_vertices))
        for i, indices in enumerate(neighbor_indices):
            if len(indices) != 0:
                match[i] = 1.0
        return match

    def __call__(
        self,
        pred_vertices: np.ndarray,
        radii: list[float] = [0.002, 0.003],  # 0.2, 0.3 % of a scene
    ) -> dict[str, float]:

        results = {}

        for radius in radii:

            str_radius = str(radius).replace(".", "")
            matches = self._compute_completeness(pred_vertices, radius)
            results[f"metrics_main/completeness_{str_radius}"] = float(matches.mean())

            for label_name, label_inx in LABELS.items():
                mask_label = self.labels == label_inx
                results[f"metrics_labels/completeness_{label_name}_{str_radius}"] = (
                    float(matches[mask_label].mean())
                )

            mask_label = self.labels < 0
            results[f"metrics_main/completeness_high_freq_{str_radius}"] = float(
                matches[mask_label].mean()
            )

            mask_label = self.labels > 0
            results[f"metrics_main/completeness_low_freq_{str_radius}"] = float(
                matches[mask_label].mean()
            )

        return results
