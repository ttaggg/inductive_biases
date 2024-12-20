"""Dataset classes for point cloud data."""

from abc import abstractmethod
from typing import Dict, Tuple

import numpy as np
from torch.utils.data import Dataset

from ib.utils.data import load_obj, load_xyz, normalize_points_and_normals
from ib.utils.logging_module import logging


class PointCloudDataset(Dataset):
    """Dataset class for point cloud data."""

    def __init__(
        self,
        file_path: str,
        batch_size: int,
        off_surface_ratio: float,
    ) -> None:
        points, normals = self.load(file_path)
        self.points, self.normals = normalize_points_and_normals(points, normals)
        self.batch_size = batch_size
        self.off_surface_ratio = off_surface_ratio

        assert self.batch_size <= len(self.points)
        logging.info(f"Dataset size: {len(self.points)}.")

    def __getitem__(self, _: int) -> Dict[str, np.ndarray]:
        """Retrieve the whole batch of data."""

        off_num_samples = int(self.batch_size * self.off_surface_ratio)
        on_num_samples = self.batch_size - off_num_samples

        off_points = np.random.uniform(-1, 1, size=(off_num_samples, 3))
        off_normals = -1 * np.ones_like(off_points)
        off_sdf = -1 * np.ones(shape=(off_num_samples, 1))

        rand_idx = np.random.choice(len(self.points), size=on_num_samples)
        on_points = self.points[rand_idx]
        on_normals = self.normals[rand_idx]
        on_sdf = np.ones(shape=(on_num_samples, 1))

        points = np.concatenate([off_points, on_points], axis=0)
        normals = np.concatenate([off_normals, on_normals], axis=0)
        sdf = np.concatenate([off_sdf, on_sdf], axis=0)

        return {
            "inputs": points.astype(np.float32),
            "normals": normals.astype(np.float32),
            "sdf": sdf.astype(np.float32),
        }

    def __len__(self) -> int:
        """__len__ method of torch.utils.data.Dataset."""
        return len(self.points) // self.batch_size

    @abstractmethod
    def load(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads a pointcloud from the file.

        Args:
            file_path (str): Path to the file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of points and normals.
        """


class ObjDataset(PointCloudDataset):
    """Dataset class for OBJ format data."""

    def load(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        return load_obj(file_path, {"v": float, "vn": float})


class XyzDataset(PointCloudDataset):
    """Dataset class for XYZ format data."""

    def load(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        return load_xyz(file_path)
