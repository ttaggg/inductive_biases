"""Evaluator class."""

from enum import Enum
from pathlib import Path

import torch
from torch import nn

from ib.metrics.chamfer_distance import ChamferDistance
from ib.metrics.fourier_freq import FourierFrequency
from ib.metrics.iou import Iou
from ib.metrics.normal_distance import NormalCosineSimilarity
from ib.models.decoders import SdfDecoder
from ib.utils.geometry import mesh_to_pointcloud, sdf_to_pointcloud
from ib.utils.pipeline import generate_output_mesh_path
from ib.utils.logging_module import logging


class Metric(str, Enum):
    chamfer = "chamfer"
    iou = "iou"
    ff = "fourier_freq"
    normals = "normals"


class FileType(str, Enum):
    sdf = "sdf"
    sparse_sdf = "sparse_sdf"
    pc = "pointcloud"


def _resolve_file_type(file_path: Path):
    match file_path.suffix:
        case ".npy":
            return FileType.sdf
        case ".npz":
            return FileType.sparse_sdf
        case _:
            return FileType.pc


def _resolve_metrics(file_path: Path, metric: list[Metric], num_samples: int) -> dict:

    file_type = _resolve_file_type(file_path)
    mapping = {}

    if Metric.chamfer in metric:
        if file_type is FileType.pc:
            mapping[Metric.chamfer] = ChamferDistance.from_pointcloud_path(
                file_path, num_samples
            )
        elif file_type is FileType.sdf:
            mapping[Metric.chamfer] = ChamferDistance.from_sdf_path(
                file_path, num_samples
            )
        elif file_type is FileType.sparse_sdf:
            mapping[Metric.chamfer] = ChamferDistance.from_sparse_sdf_path(
                file_path, num_samples
            )

    if Metric.iou in metric and file_type is FileType.sdf:
        mapping[Metric.iou] = Iou(file_path)

    if Metric.ff in metric and file_type is FileType.sdf:
        mapping[Metric.ff] = FourierFrequency(file_path)

    if Metric.normals in metric:
        if file_type is FileType.pc:
            mapping[Metric.normals] = NormalCosineSimilarity.from_pointcloud_path(
                file_path, num_samples
            )
        elif file_type is FileType.sdf:
            mapping[Metric.normals] = NormalCosineSimilarity.from_sdf_path(
                file_path, num_samples
            )
        elif file_type is FileType.sparse_sdf:
            mapping[Metric.normals] = NormalCosineSimilarity.from_sparse_sdf_path(
                file_path, num_samples
            )

    return mapping


class Evaluator:

    def __init__(
        self,
        file_path: Path,
        metric: list[Metric],
        num_samples: int = 1_000_000,
    ) -> None:
        self.file_type = _resolve_file_type(file_path)
        self.metrics = _resolve_metrics(file_path, metric, num_samples)
        self.num_samples = num_samples

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
    ) -> dict:
        model = torch.load(model_path, weights_only=False, map_location=device)
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

        output_mesh_path = generate_output_mesh_path(
            model.model_cfg.paths.saved_models / f"model_epoch_{current_epoch}.pt",
            resolution,
        )

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
            decoder.save(output_mesh_path)

        # Run once for Chamfer and Normal distances.
        if Metric.chamfer in self.metrics or Metric.normals in self.metrics:
            if self.file_type is FileType.pc:
                pred_verts, pred_normals = mesh_to_pointcloud(
                    decoder.vertices, decoder.faces, self.num_samples
                )
            elif self.file_type in {FileType.sdf, FileType.sparse_sdf}:
                pred_verts, pred_normals = sdf_to_pointcloud(
                    decoder.sdf, self.num_samples
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
                    output_mesh_path.parent / f"normals_closest_{current_epoch}.ply",
                )
            )

        if Metric.iou in self.metrics:
            logging.info(f"Computing IoU.")
            results.update(self.metrics[Metric.iou](decoder.sdf))

        if Metric.ff in self.metrics:
            logging.info(f"Computing Fourier frequency.")
            results.update(self.metrics[Metric.ff](decoder.sdf))

        model.train(is_training)
        return results
