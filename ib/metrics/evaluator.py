"""Evaluator class."""

from enum import Enum
from pathlib import Path

from skimage import measure

from ib.datasets.resamplers import Resampler, SamplingException
from ib.metrics.chamfer_distance import ChamferDistance
from ib.utils.logging_module import logging
from ib.utils.model import query_model


class Metric(str, Enum):
    chamfer = "chamfer"


class Evaluator:

    def __init__(self, pointcloud_path: Path) -> None:
        # TODO(oleg): initialize Chamfer distance class only when needed.
        self.chamfer = ChamferDistance(pointcloud_path)

    def run(self, model, metric_names, resolution: int, batch_size: int):
        is_training = model.training
        model.eval()

        # Run once for all the metrics.
        sdf = query_model(model, resolution, batch_size, model.device)

        results = {}
        try:
            if Metric.chamfer in metric_names:
                vertices, faces, _, _ = measure.marching_cubes(sdf, level=0)
                resampler = Resampler(vertices, faces)
                resampler.run(num_samples=min(5_000_000, 2 * len(faces)))
                results["metrics/chamfer"] = self.chamfer(resampler.sampled_vertices)
        except SamplingException as e:
            logging.info(str(e))
            logging.info(
                "Cannot sample from the generated mesh. "
                "Skipping Chamfer metric evaluation."
            )

        model.train(is_training)
        return results
