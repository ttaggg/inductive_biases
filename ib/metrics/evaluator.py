"""Evaluator class."""

import json
from enum import Enum
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch import nn

from ib.metrics.chamfer_distance import ChamferDistance
from ib.metrics.completeness import Completeness
from ib.metrics.local_curvature import CurvatureNormalChangeRate
from ib.metrics.lpips import LpipsMetric
from ib.metrics.normal_distance import NormalCosineSimilarity
from ib.models.decoders import SdfDecoder
from ib.utils.data import load_pointcloud, load_ply, make_o3d_mesh
from ib.utils.geometry import mesh_to_pointcloud
from ib.utils.pipeline import (
    decode_path,
    generate_output_mesh_path,
    generate_output_results_path,
)
from ib.utils.pointcloud import filter_incorrect_normals
from ib.utils.logging_module import logging


class Metric(str, Enum):
    chamfer = "chamfer"
    normals = "normals"
    curve = "curve"
    lpips = "lpips"
    complete = "complete"


def _resolve_metrics(
    metric: list[Metric], gt_data: dict[str, np.ndarray], file_path: Path
) -> dict:

    mapping = {}
    if Metric.chamfer in metric:
        mapping[Metric.chamfer] = ChamferDistance.from_pointcloud(
            gt_data["points"], gt_data["normals"], gt_data["labels"]
        )
    if Metric.normals in metric:
        mapping[Metric.normals] = NormalCosineSimilarity.from_pointcloud(
            gt_data["points"], gt_data["normals"], gt_data["labels"]
        )
    if Metric.curve in metric:
        mapping[Metric.curve] = CurvatureNormalChangeRate.from_pointcloud(
            gt_data["points"], gt_data["normals"], gt_data["labels"]
        )
    if Metric.lpips in metric:
        mapping[Metric.lpips] = LpipsMetric.from_mesh_dir(file_path.parent)

    if Metric.complete in metric:
        mapping[Metric.complete] = Completeness.from_pointcloud(
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
        self.metrics = _resolve_metrics(metric, self.gt_data, file_path)
        self.num_samples = len(self.gt_data["points"])

    def run_from_model_path(
        self,
        model_path: Path,
        device: str,
        resolution: int,
        batch_size: int,
        float32_matmul_precision: str = "high",
    ) -> dict:
        torch.set_float32_matmul_precision(float32_matmul_precision)
        model = torch.load(model_path, weights_only=False, map_location=device)
        model = torch.compile(model)
        model.to(device)
        current_epoch = int(model_path.stem.split("_")[-1])
        return self.run_from_model(
            model,
            resolution,
            batch_size,
            current_epoch,
        )

    def run_from_model(
        self,
        model: nn.Module,
        resolution: int,
        batch_size: int,
        current_epoch: int | None = None,
    ) -> dict:
        is_training = model.training
        model.eval()

        if current_epoch is None:
            current_epoch = model.current_epoch

        # Figure out the name.
        run_name = ""
        if hasattr(model.model_cfg, "run_name"):
            run_name = model.model_cfg.run_name.replace("/", "_")
        model_path = model.model_cfg.paths.saved_models

        output_mesh_path = generate_output_mesh_path(
            model_path.parent, run_name, current_epoch, resolution
        )
        output_results_path = generate_output_results_path(
            model_path.parent, run_name, current_epoch, resolution
        )

        # Run once for all the metrics.
        try:
            decoder = SdfDecoder(model)
            decoder.run(resolution, batch_size)
            decoder.trim_mesh(self.gt_data["points"])
            decoder.save(output_mesh_path)
        except Exception as e:
            logging.info(f"An error occurred: {e}.")
            logging.info("Cannot decode, returning empty metrics.")
            model.train(is_training)
            return {}

        # Run once for Chamfer and Normal distances.
        pred_verts, pred_normals = mesh_to_pointcloud(
            decoder.vertices, decoder.faces, self.num_samples
        )
        pred_mesh = decoder.mesh
        results = self._run(
            pred_verts,
            pred_normals,
            pred_mesh,
            resolution,
            current_epoch,
            save_mesh_dir=output_mesh_path.parent,
            save_results_file=output_results_path,
        )
        model.train(is_training)
        return results

    def run_from_mesh_path(self, mesh_path: Path) -> dict:
        data = load_ply(mesh_path)
        vertices, faces = data["points"], data["faces"]
        pred_verts, pred_normals = mesh_to_pointcloud(vertices, faces, self.num_samples)
        pred_mesh = make_o3d_mesh(vertices, faces)
        run_name, current_epoch, resolution = decode_path(mesh_path)
        output_results_path = generate_output_results_path(
            mesh_path.parent, run_name, current_epoch, resolution
        )

        return self._run(
            pred_verts,
            pred_normals,
            pred_mesh,
            resolution,
            current_epoch,
            save_mesh_dir=mesh_path.parent,
            save_results_file=output_results_path,
        )

    def _run(
        self,
        pred_verts: np.ndarray,
        pred_normals: np.ndarray,
        pred_mesh: o3d.geometry.TriangleMesh,
        resolution: int,
        current_epoch: int,
        save_mesh_dir: Path,
        save_results_file: Path,
    ) -> dict:
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
                    save_path=save_mesh_dir
                    / f"normals_similarity_epoch_{current_epoch}_res_{resolution}",
                )
            )

        if Metric.curve in self.metrics:
            logging.info(f"Computing Curvature Normal Change Rate.")
            results.update(self.metrics[Metric.curve](pred_verts, pred_normals))

        if Metric.lpips in self.metrics:
            logging.info(f"Computing LPIPS.")
            results.update(
                self.metrics[Metric.lpips](
                    pred_mesh,
                    save_path=save_mesh_dir
                    / f"image_wall_{current_epoch}_res_{resolution}",
                )
            )

        if Metric.complete in self.metrics:
            logging.info(f"Computing Completeness.")
            results.update(self.metrics[Metric.complete](pred_verts))

        with open(save_results_file, "w") as f:
            json.dump(dict(sorted(results.items())), f, indent=4)
            logging.info(f"Evaluation results are saved to {save_results_file}")

        return results
