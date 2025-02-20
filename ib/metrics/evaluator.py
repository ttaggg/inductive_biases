"""Evaluator class."""

from enum import Enum
from pathlib import Path

import torch
from torch import nn

from ib.metrics.chamfer_distance import ChamferDistance
from ib.metrics.iou import Iou
from ib.models.decoders import SdfDecoder
from ib.utils.geometry import mesh_to_pointcloud, sdf_to_pointcloud
from ib.utils.pipeline import generate_output_mesh_path
from ib.utils.logging_module import logging


class Metric(str, Enum):
    chamfer = "chamfer"
    iou = "iou"


class FileType(str, Enum):
    sdf = "sdf"
    pc = "pointcloud"


class Evaluator:

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_type = FileType.sdf if file_path.suffix == ".npy" else FileType.pc

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
        try:
            decoder = SdfDecoder(model)
            decoder.run(resolution, batch_size)
        except Exception as e:
            logging.info(f"An error occurred: {e}.")
            logging.info("Cannot decode, returning empty metrics.")
            model.train(is_training)
            return {}

        if save_mesh:
            output_path = generate_output_mesh_path(
                model.model_cfg.paths.saved_models / f"model_epoch_{current_epoch}.pt",
                resolution,
            )
            decoder.save(output_path)

        results = {}

        if Metric.chamfer in metric_names:

            if self.file_type is FileType.pc:
                chamfer_fn = ChamferDistance.from_pointcloud_path(self.file_path)
                pred_verts = mesh_to_pointcloud(
                    decoder.vertices, decoder.faces, num_samples=chamfer_fn.gt_size()
                )
                results.update(chamfer_fn(pred_verts))

            elif self.file_type is FileType.sdf:
                chamfer_fn = ChamferDistance.from_sdf_path(self.file_path)
                pred_verts = sdf_to_pointcloud(decoder.sdf, num_samples=1_000_000)
                results.update(chamfer_fn(pred_verts))

        if Metric.iou in metric_names and self.file_type is FileType.sdf:
            iou_dist = Iou(self.file_path)
            results.update(iou_dist(decoder.sdf))

        model.train(is_training)
        return results
