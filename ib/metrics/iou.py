"""Module to calculate IoU between two SDFs."""

import numpy as np

from pathlib import Path


def _calculate_iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    intersection = np.sum(np.logical_and(lhs, rhs))
    union = np.sum(np.logical_or(lhs, rhs))
    iou = intersection / float(union) if union != 0 else 0.0
    return float(iou)


class Iou:
    def __init__(self, sdf_path: Path) -> None:
        self.gt_sdf = np.load(sdf_path)

    def __call__(self, predicted_sdf: np.ndarray) -> dict[str, float]:

        # The whole volume.
        predicted_voxels = predicted_sdf <= 0
        gt_voxels = self.gt_sdf <= 0
        voxel_iou = _calculate_iou(predicted_voxels, gt_voxels)

        # Near the surface.
        surface_threshold = 2 / predicted_sdf.shape[0]  # Assuming cube.
        predicted_surface = np.abs(predicted_sdf) <= surface_threshold
        gt_surface = np.abs(self.gt_sdf) <= surface_threshold
        surface_iou = _calculate_iou(predicted_surface, gt_surface)

        return {
            "metrics/voxel_iou": float(voxel_iou),
            "metrics/surface_iou": float(surface_iou),
        }
