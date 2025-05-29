"""Evaluator class."""

from enum import Enum
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ib.metrics.combined_metric import CombinedPointcloudMetric
from ib.metrics.chamfer_distance import ChamferDistance
from ib.metrics.fourier_freq import FourierFrequency
from ib.metrics.normal_distance import NormalCosineSimilarity
from ib.models.decoders import SdfDecoder
from ib.utils.data import load_pointcloud
from ib.utils.geometry import mesh_to_pointcloud
from ib.utils.pipeline import generate_output_mesh_path
from ib.utils.pointcloud import filter_incorrect_normals
from ib.utils.logging_module import logging


class Metric(str, Enum):
    chamfer = "chamfer"
    ff = "fourier_freq"
    normals = "normals"
    combined = "combined"


def _resolve_metrics(metric: list[Metric], gt_data: dict[str, np.ndarray]) -> dict:

    mapping = {}
    if Metric.chamfer in metric:
        mapping[Metric.chamfer] = ChamferDistance.from_pointcloud(
            gt_data["points"], gt_data["normals"], gt_data["labels"]
        )
    if Metric.normals in metric:
        mapping[Metric.normals] = NormalCosineSimilarity.from_pointcloud(
            gt_data["points"], gt_data["normals"], gt_data["labels"]
        )
    if Metric.ff in metric:
        mapping[Metric.ff] = FourierFrequency()

    if Metric.combined in metric:
        mapping[Metric.combined] = CombinedPointcloudMetric.from_pointcloud(
            gt_data["points"], gt_data["normals"], gt_data["labels"]
        )
    return mapping


class Evaluator:

    def __init__(self, file_path: Path, metric: list[Metric]) -> None:
        self.gt_data = load_pointcloud(file_path)
        self.gt_data["points"], self.gt_data["normals"], self.gt_data["labels"] = (
            filter_incorrect_normals(
                self.gt_data["points"],
                self.gt_data["normals"],
                self.gt_data["labels"],
            )
        )
        self.metrics = _resolve_metrics(metric, self.gt_data)
        self.num_samples = len(self.gt_data["points"])

    def run(
        self,
        model: nn.Module,
        resolution: int,
        batch_size: int,
        save_mesh: bool,
    ) -> dict:
        return self._run(
            model,
            model.current_epoch,
            resolution,
            batch_size,
            save_mesh,
        )

    def run_from_path(
        self,
        model_path: Path,
        device: str,
        resolution: int,
        batch_size: int,
        save_mesh: bool,
        float32_matmul_precision: str = "high",
    ) -> dict:
        torch.set_float32_matmul_precision(float32_matmul_precision)
        model = torch.load(model_path, weights_only=False, map_location=device)
        model = torch.compile(model)
        model.to(device)
        current_epoch = int(model_path.stem.split("_")[-1])
        return self._run(
            model,
            current_epoch,
            resolution,
            batch_size,
            save_mesh,
        )

    def _run(
        self,
        model: nn.Module,
        current_epoch: int,
        resolution: int,
        batch_size: int,
        save_mesh: bool = True,
    ) -> dict:
        is_training = model.training
        model.eval()

        run_name = ""
        if hasattr(model.model_cfg, "run_name"):
            run_name = model.model_cfg.run_name.replace("/", "_")
        output_mesh_path = generate_output_mesh_path(
            model.model_cfg.paths.saved_models
            / f"model_{run_name}_epoch_{current_epoch}.pt",
            resolution,
        )

        # Run once for all the metrics.
        try:
            decoder = SdfDecoder(model)
            decoder.run(resolution, batch_size)
            decoder.trim_mesh(self.gt_data["points"])
        except Exception as e:
            logging.info(f"An error occurred: {e}.")
            logging.info("Cannot decode, returning empty metrics.")
            model.train(is_training)
            return {}

        if save_mesh:
            decoder.save(output_mesh_path)

        # Run once for Chamfer and Normal distances.
        pred_verts, pred_normals = mesh_to_pointcloud(
            decoder.vertices, decoder.faces, self.num_samples
        )

        results = {}
        if self.metrics is None:
            return results

        if Metric.chamfer in self.metrics:
            logging.info(f"Computing Chamfer distance.")
            results.update(self.metrics[Metric.chamfer](pred_verts))

        if Metric.normals in self.metrics:
            logging.info(f"Computing Normal distance.")
            results.update(
                self.metrics[Metric.normals](
                    pred_verts,
                    pred_normals,
                    save_path=Path(output_mesh_path).parent
                    / f"normals_similarity_epoch_{current_epoch}",
                )
            )

        if Metric.ff in self.metrics:
            logging.info(f"Computing Fourier frequency.")
            results.update(self.metrics[Metric.ff](decoder.sdf))

        if Metric.combined in self.metrics:
            logging.info(f"Computing combined metric.")
            results.update(self.metrics[Metric.combined](pred_verts, pred_normals))

        model.train(is_training)
        return results
