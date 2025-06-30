"""Completeness."""

from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_pointcloud
from ib.utils.labels import INX_TO_LABEL
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

    def _compute_completeness(
        self,
        source_vertices: np.ndarray,
        source_tree: KDTree,
        target_tree,
        radius: float,
    ) -> np.ndarray:
        neighbor_indices: list[list] = target_tree.query_ball_tree(
            source_tree, r=radius
        )
        match = np.zeros(len(source_vertices))
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

        pred_tree = KDTree(pred_vertices)

        # Completeness
        for radius in radii:

            str_radius = str(radius).replace(".", "")
            matches = self._compute_completeness(
                pred_vertices, pred_tree, self.tree, radius
            )
            results[f"metrics_main/completeness_{str_radius}"] = float(matches.mean())

            label_indices = np.unique(self.labels)
            for label_inx in label_indices:
                label_name = INX_TO_LABEL.get(label_inx, "unknown")
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

        # Artifacts
        for radius in radii:

            str_radius = str(radius).replace(".", "")
            matches = self._compute_completeness(
                self.vertices, self.tree, pred_tree, radius
            )
            results[f"metrics_main/artifacts_{str_radius}"] = float(1 - matches.mean())

            label_indices = np.unique(self.labels)
            for label_inx in label_indices:
                label_name = INX_TO_LABEL.get(label_inx, "unknown")
                mask_label = self.labels == label_inx
                results[f"metrics_labels/artifacts_{label_name}_{str_radius}"] = float(
                    1 - matches[mask_label].mean()
                )

            mask_label = self.labels < 0
            results[f"metrics_main/artifacts_high_freq_{str_radius}"] = float(
                1 - matches[mask_label].mean()
            )

            mask_label = self.labels > 0
            results[f"metrics_main/artifacts_low_freq_{str_radius}"] = float(
                1 - matches[mask_label].mean()
            )

        return results
