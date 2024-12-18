"""Dataset classes for point cloud data."""

from abc import abstractmethod
from typing import Dict, Tuple

import numpy as np
from torch.utils.data import Dataset

from ib.utils.data import load_obj, load_xyz, normalize_points_and_normals
from ib.utils.logging_module import logging


class PointCloudDataset(Dataset):
    """Dataset class for point cloud data."""

    def __init__(self, file_path: str, off_surface_ratio: float) -> None:
        points, normals = self.load(file_path)
        self._points, self._normals = normalize_points_and_normals(points, normals)
        self._off_surface_ratio = off_surface_ratio

        logging.info(f"Dataset size: {len(points)}.")

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Retrieve a data sample by index."""
        if np.random.rand(1) > self._off_surface_ratio:
            point = np.random.uniform(-1, 1, size=(3,))
            normal = -1 * np.ones_like(point)
            sdf = np.array([-1.0])
        else:
            point = self._points[idx]
            normal = self._normals[idx]
            sdf = np.array([0.0])

        return {
            "inputs": point.astype(np.float32),
            "normals": normal.astype(np.float32),
            "sdf": sdf.astype(np.float32),
        }

    def __len__(self) -> int:
        """__len__ method of torch.utils.data.Dataset."""
        return len(self._points)

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
