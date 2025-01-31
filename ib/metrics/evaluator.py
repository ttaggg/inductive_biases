"""Evaluator class."""

from enum import Enum
from pathlib import Path

import torch
from torch import nn

from ib.datasets.resamplers import Resampler, SamplingException
from ib.metrics.chamfer_distance import ChamferDistance
from ib.models.decoders import SdfDecoder
from ib.utils.logging_module import logging
from ib.utils.pipeline import generate_output_mesh_path


class Metric(str, Enum):
    chamfer = "chamfer"


class Evaluator:

    def __init__(self, pointcloud_path: Path) -> None:
        # TODO(oleg): initialize Chamfer distance class only when needed.
        self.chamfer = ChamferDistance(pointcloud_path)

    def run(
        self,
        model: nn.Module,
        metric_names: list[Metric],
        resolution: int,
        batch_size: int,
        save_mesh: bool,
    ) -> dict:
        return self._run(
            model,
            model.current_epoch,
            metric_names,
            resolution,
            batch_size,
            save_mesh,
        )

    def run_from_path(
        self,
        model_path: Path,
        device: str,
        metric_names: list[Metric],
        resolution: int,
        batch_size: int,
        save_mesh: bool,
    ) -> dict:
        model = torch.load(model_path, weights_only=False, map_location=device)
        model.to(device)
        current_epoch = int(model_path.stem.split("_")[-1])
        return self._run(
            model,
            current_epoch,
            metric_names,
            resolution,
            batch_size,
            save_mesh,
        )

    def _run(
        self,
        model: nn.Module,
        current_epoch: int,
        metric_names: list[Metric],
        resolution: int,
        batch_size: int,
        save_mesh: bool = True,
    ) -> dict:
        is_training = model.training
        model.eval()

        # Run once for all the metrics.
        decoder = SdfDecoder(model)
        decoder.run(resolution, batch_size)

        if save_mesh:
            output_path = generate_output_mesh_path(
                model.model_cfg.paths.saved_models / f"model_epoch_{current_epoch}.pt",
                resolution,
            )
            decoder.save(output_path)

        results = {}
        try:
            if Metric.chamfer in metric_names:
                resampler = Resampler(decoder.vertices, decoder.faces)
                resampler.run(num_samples=min(64_000_000, 2 * len(decoder.faces)))
                results["metrics/chamfer"] = self.chamfer(resampler.sampled_vertices)
        except SamplingException as e:
            logging.info(str(e))
            logging.info(
                "Cannot sample from the generated mesh. "
                "Skipping Chamfer metric evaluation."
            )

        model.train(is_training)
        return results
