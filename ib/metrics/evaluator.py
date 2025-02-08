"""Evaluator class."""

from enum import Enum
from pathlib import Path

import torch
from torch import nn

from ib.datasets.resamplers import SimpleResampler
from ib.metrics.chamfer_distance import ChamferDistance
from ib.metrics.iou import Iou
from ib.models.decoders import SdfDecoder
from ib.utils.pipeline import generate_output_mesh_path


class Metric(str, Enum):
    chamfer = "chamfer"
    iou = "iou"


class Evaluator:

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

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

        if Metric.chamfer in metric_names:
            chamfer_dist = ChamferDistance(self.file_path)
            resampler = SimpleResampler(decoder.vertices, decoder.faces)
            resampler.run(num_samples=chamfer_dist.gt_size())
            results["metrics/chamfer"] = chamfer_dist(resampler.sampled_vertices)

        if Metric.iou in metric_names:
            iou_dist = Iou(self.file_path)
            results["metrics/iou"] = iou_dist(decoder.sdf)

        model.train(is_training)
        return results
