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

    def __call__(
        self,
        pred_vertices: np.ndarray,
        radii: list[float] = [0.002, 0.003],  # 0.2%, 0.3 % of a scene
    ) -> dict[str, float]:

        results = {}

        pred_tree = KDTree(pred_vertices)
        distances, _ = pred_tree.query(self.vertices, workers=-1)
        for radius in radii:

            matches = (distances <= radius).astype(float)
            str_radius = str(radius).replace(".", "")
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

        return results
