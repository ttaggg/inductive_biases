"""Chamfer Distance metric."""

from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_pointcloud
from ib.utils.labels import INX_TO_LABEL
from ib.utils.pointcloud import filter_incorrect_normals


class ChamferDistance:
    """Bidirectional Chamfer Distance metric."""

    def __init__(self, vertices: np.ndarray, labels: np.ndarray | None) -> None:
        self.vertices = vertices
        self.labels = labels
        self.target_tree = KDTree(self.vertices)

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

    def __call__(self, pred_vertices: np.ndarray) -> dict[str, float]:

        dist_p2t, idx_p2t = self.target_tree.query(pred_vertices, workers=-1)
        pred_tree = KDTree(pred_vertices)
        dist_t2p, _ = pred_tree.query(self.vertices, workers=-1)
        overall = (dist_p2t.mean() + dist_t2p.mean()) / 2.0

        results = {}
        results["metrics/chamfer_p2t"] = float(dist_p2t.mean())
        results["metrics_main/chamfer_t2p"] = float(dist_t2p.mean())
        results["metrics/chamfer"] = float(overall)

        label_indices = np.unique(self.labels)
        for label_inx in label_indices:
            # Target to predicted.
            label_name = INX_TO_LABEL.get(label_inx, "unknown")
            mask_label = self.labels == label_inx
            label_dist_t2p_mean = dist_t2p[mask_label].mean()
            results[f"metrics_labels/chamfer_{label_name}_t2p"] = float(
                label_dist_t2p_mean
            )
            # Predicted to target.
            pred_labels = self.labels[idx_p2t]
            mask_pred_label = pred_labels == label_inx
            label_dist_p2t_mean = dist_p2t[mask_pred_label].mean()
            results[f"metrics_labels/chamfer_{label_name}_p2t"] = float(
                label_dist_p2t_mean
            )
            # Mean.
            results[f"metrics_labels/chamfer_{label_name}"] = float(
                (label_dist_t2p_mean + label_dist_p2t_mean) / 2.0
            )

        if self.labels is not None:
            mask_self = self.labels > 0

            # Low-resolution regions.
            low_freq_t2p = dist_t2p[mask_self]
            low_freq_p2t = dist_p2t[mask_self[idx_p2t]]
            low_freq_t2p_mean = low_freq_t2p.mean()
            low_freq_p2t_mean = low_freq_p2t.mean()
            results["metrics_main/chamfer_low_freq_t2p"] = float(low_freq_t2p_mean)
            results["metrics/chamfer_low_freq_p2t"] = float(low_freq_p2t_mean)
            results["metrics/chamfer_low_freq"] = float(
                (low_freq_t2p_mean + low_freq_p2t_mean) / 2.0
            )

            # Other regions.
            mask_others = self.labels < 0
            high_freq_t2p = dist_t2p[mask_others]
            high_freq_p2t = dist_p2t[mask_others[idx_p2t]]
            high_freq_t2p_mean = high_freq_t2p.mean()
            high_freq_p2t_mean = high_freq_p2t.mean()
            results["metrics_main/chamfer_high_freq_t2p"] = float(high_freq_t2p_mean)
            results["metrics/chamfer_high_freq_p2t"] = float(high_freq_p2t_mean)
            results["metrics/chamfer_high_freq"] = float(
                (high_freq_t2p_mean + high_freq_p2t_mean) / 2.0
            )
        return results
