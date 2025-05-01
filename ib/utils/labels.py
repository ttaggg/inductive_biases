import json
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from ib.utils.data import load_ply, write_ply
from ib.utils.logging_module import logging

LOW_RESOLUTION_LABELS = {
    "whiteboard": 1,
    "window": 2,
    "wall": 3,
    "floor": 4,
}


def _load_lowres_labels(filepath: Path) -> list[tuple]:
    outputs = []
    with open(filepath) as f:
        data = json.load(f)
        for obj in data["segGroups"]:
            if obj["label"] in LOW_RESOLUTION_LABELS.keys():
                outputs.append((obj["label"], obj["segments"]))
    return outputs


def _annotate_ref_points(ref_points: np.ndarray, labels: list[tuple]) -> np.ndarray:
    ref_labels = np.zeros(len(ref_points))
    for label, segment_indices in labels:
        ref_labels[segment_indices] = LOW_RESOLUTION_LABELS[label]
    return ref_labels


def compute_labels(
    points: np.ndarray,
    mesh_path: Path,
    labels_path: Path,
) -> np.ndarray:
    """Convert labels from Scannet++ mesh to the pointcloud."""

    # Load labels and annotate reference points.
    ref_points = load_ply(mesh_path)["points"]
    labels = _load_lowres_labels(labels_path)
    ref_labels = _annotate_ref_points(ref_points, labels)

    logging.info("Computing labels for the pointcloud via nearest neighbors.")
    ref_tree = KDTree(ref_points)
    _, nn_idx = ref_tree.query(points, workers=-1)
    labels = ref_labels[nn_idx]

    return labels
