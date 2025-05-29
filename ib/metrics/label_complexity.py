from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_pointcloud
from ib.utils.labels import LABELS
from ib.utils.logging_module import logging

INX2LABEL = {v: k for k, v in LABELS.items()}


def _spherical_angles(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spherical coordinates."""
    z = np.clip(normals[:, 2], -1.0, 1.0)
    theta = np.arccos(z)
    phi = np.mod(np.arctan2(normals[:, 1], normals[:, 0]), 2.0 * np.pi)
    return theta, phi


def normal_entropy(
    points: np.ndarray,
    normals: np.ndarray,
    radius: float = 0.1,
    bins: int = 32,
) -> tuple[np.ndarray, np.ndarray]:

    points = np.ascontiguousarray(points, dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    N = points.shape[0]

    tree = KDTree(points)
    neighbours = tree.query_ball_point(points, radius)

    # Pre-compute histogram bin edges on the sphere
    n_theta = bins // 2
    theta_edges = np.linspace(0.0, np.pi, n_theta + 1, dtype=np.float64)
    phi_edges = np.linspace(
        0.0, np.nextafter(2.0 * np.pi, 4.0 * np.pi), bins + 1, dtype=np.float64
    )
    total_bins = n_theta * bins
    entropy = np.full(N, np.nan, dtype=np.float64)

    subset_sizes = []
    for i, idx_list in enumerate(neighbours):
        k = len(idx_list)

        subset_sizes.append(k)
        idx = np.asarray(idx_list, dtype=np.int64)

        # Entropy for normals
        theta, phi = _spherical_angles(normals[idx])
        hist, _ = np.histogramdd(
            (theta, phi),
            bins=(theta_edges, phi_edges),
        )
        prob = hist.ravel()
        prob = prob[prob > 0] / k

        ent = -np.sum(prob * np.log2(prob))
        H_max = np.log2(min(k, total_bins))
        entropy[i] = ent / H_max if H_max > 0 else np.nan

    mean_entropy = np.nanmean(entropy)
    logging.info(f"subset_sizes: {np.mean(subset_sizes)}")

    return mean_entropy


class LabelComplexity:
    """Evaluate complexity of sub‑clouds."""

    def evaluate_pointcloud(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        labels: np.ndarray,
        max_samples: int = 50_000,
    ) -> dict[str, float]:

        if points.shape[0] != labels.shape[0]:
            raise ValueError(
                "Points and labels must have the same length "
                f"(got {points.shape[0]} vs {labels.shape[0]})"
            )

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]  # exclude unlabeled
        logging.info(f"Evaluating complexity for {len(unique_labels)} labels")

        results = {}
        for label in unique_labels:
            label_mask = labels == label
            label_points = points[label_mask]
            label_normals = normals[label_mask]
            label_name = INX2LABEL[label] if label in INX2LABEL else f"unknown_{label}"

            # Sample a fixed number of points
            num_points = label_points.shape[0]
            if num_points > max_samples:
                indices = np.random.choice(num_points, size=max_samples, replace=False)
                label_points = label_points[indices]
                label_normals = label_normals[indices]

            results[f"{label_name}_entropy"] = normal_entropy(
                label_points, label_normals
            )

        return results

    def evaluate_from_file(self, pointcloud_path: Path) -> dict[str, float]:
        data = load_pointcloud(pointcloud_path)
        return self.evaluate_pointcloud(data["points"], data["normals"], data["labels"])
