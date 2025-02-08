"""Module to calculate IoU between two SDFs."""

import numpy as np

from pathlib import Path


class Iou:
    def __init__(self, sdf_path: Path) -> None:
        sdf = np.load(sdf_path)
        self.gt_voxels = sdf <= 0

    def __call__(self, predicted_sdf: np.ndarray) -> float:

        predicted_voxels = predicted_sdf <= 0

        intersection = np.sum(np.logical_and(predicted_voxels, self.gt_voxels))
        union = np.sum(np.logical_or(predicted_voxels, self.gt_voxels))

        if union == 0:
            return 0.0

        iou = intersection / float(union)
        return float(iou)
