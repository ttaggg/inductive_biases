"""Chamfer Distance metric."""

from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_pointcloud
from ib.utils.pointcloud import filter_incorrect_normals


class ChamferDistance:
    """Bidirectional Chamfer Distance metric."""

    def __init__(
        self,
        vertices: np.ndarray,
        labels: np.ndarray | None,
        num_points: int,
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
        vertices, _, labels = filter_incorrect_normals(
            data["points"], data["normals"], data["labels"]
        )
        return cls(vertices, labels, num_points)

    def __call__(self, pred_vertices: np.ndarray) -> dict[str, float]:
        dist_p2t, idx_p2t = self.tree.query(pred_vertices, workers=-1)

        pred_tree = KDTree(pred_vertices)
        dist_t2p, _ = pred_tree.query(self.vertices, workers=-1)

        overall = (dist_p2t.mean() + dist_t2p.mean()) / 2.0
        results = {"metrics/chamfer": float(overall)}
        results["metrics/chamfer_p2t"] = float(dist_p2t.mean())
        results["metrics/chamfer_t2p"] = float(dist_t2p.mean())

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

        return results
