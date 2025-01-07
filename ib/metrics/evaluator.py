"""Evaluator class."""

from enum import Enum
from typing import Union
from pathlib import Path

import torch
from skimage import measure
from torch import nn

from ib.datasets.resamplers import Resampler
from ib.metrics.chamfer_distance import ChamferDistance
from ib.utils.model import query_model


class Metric(str, Enum):
    chamfer = "chamfer"


class Evaluator:

    def __init__(
        self,
        model: nn.Module,
        pointcloud_path: Path,
        device: str,
    ) -> None:

        self.model = model
        # TODO(oleg): if used from the BaseModel during validation,
        # check if the model switches to the train mode after.
        self.model.eval()
        self.device = device

        self.metrics_dict = {
            Metric.chamfer: ChamferDistance(pointcloud_path),
        }

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Path,
        pointcloud_path: Path,
        device: str,
    ):
        model = torch.load(model_path, weights_only=False, map_location=device)
        return cls(model, pointcloud_path, device)

    def run(self, metric_names, resolution: int, batch_size: int):

        # Run once for all the metrics.
        sdf = query_model(self.model, resolution, batch_size, self.device)

        results = {}
        if Metric.chamfer in metric_names:
            vertices, faces, _, _ = measure.marching_cubes(sdf, level=0)
            resampler = Resampler(vertices, faces)
            resampler.run(num_samples=1_000_000)
            results["chamfer"] = self.metrics_dict[Metric.chamfer](
                resampler.sampled_vertices
            )

        return results
