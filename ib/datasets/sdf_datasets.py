"""Datasets that loads SDF volume."""

import numpy as np
from torch.utils.data import Dataset

from ib.utils.logging_module import logging


class SdfDataset(Dataset):
    """Dataset class for SDF data."""

    def __init__(
        self,
        file_path: str,
        batch_size: int,
        off_surface_ratio: float,
        clip_sdf: float = np.inf,
        sdf_threshold_coeff: float = 2.0,
    ) -> None:
        self.sdf = np.load(file_path)
        self.dim = self.sdf.shape
        self.batch_size = batch_size
        self.off_surface_ratio = off_surface_ratio

        # Clamp SDF.
        self.sdf = np.clip(self.sdf, -clip_sdf, clip_sdf)

        # Precompute for future.
        sdf_threshold = sdf_threshold_coeff * (2.0 / self.dim[0])
        self.surface_indices = np.array(np.where(np.abs(self.sdf) < sdf_threshold)).T

        logging.info(f"Dataset size: {self.num_samples} samples.")
        logging.info(f"Dataset size: {len(self)} batches.")
        logging.warning("This dataset class was abandoned.")
        logging.warning("The current evaluator class does not work with SDFs.")

    def __getitem__(self, _: int) -> dict[str, np.ndarray]:

        off_num_samples = int(self.batch_size * self.off_surface_ratio)
        on_num_samples = self.batch_size - off_num_samples

        random_idx = np.random.randint(0, self.dim[0], size=(off_num_samples, 3))
        surface_idx = self.surface_indices[
            np.random.choice(len(self.surface_indices), on_num_samples, replace=False)
        ]
        indices = np.concatenate((random_idx, surface_idx), axis=0)

        coords = (indices / (np.array(self.sdf.shape) - 1)) * 2 - 1
        sdf_values = self.sdf[indices[:, 0], indices[:, 1], indices[:, 2]]
        sdf_values = np.expand_dims(sdf_values, -1)

        return {
            "inputs": coords.astype(np.float32),
            "sdf": sdf_values.astype(np.float32),
        }

    @property
    def num_samples(self):
        return int(len(self.surface_indices) / (1 - self.off_surface_ratio))

    def __len__(self) -> int:
        """__len__ method of torch.utils.data.Dataset."""
        return self.num_samples // self.batch_size


class SparseSdfDataset(Dataset):
    """Dataset class for sparse SDF data.

    Works with data that contains coordinates and corresponding SDF values,
    rather than a dense 3D volume.
    """

    def __init__(
        self,
        file_path: str,
        batch_size: int,
        off_surface_ratio: float,
        clip_sdf: float = np.inf,
        volume_size: int = 1024,
    ) -> None:
        """Initialize SparseSdfDataset."""

        self.batch_size = batch_size
        self.off_surface_ratio = off_surface_ratio
        self.volume_size = volume_size

        sparse_data = np.load(file_path)
        self.surface_coords = sparse_data["coords"].astype(np.int32)
        self.surface_sdf = sparse_data["sdf"].astype(np.float32).reshape(-1, 1)

        # Use a dedicated random generator
        self.rng = np.random.default_rng()

        # Check values, clip, find maximum for off surface sampling.
        sdf_min, sdf_mean, sdf_max = (
            self.surface_sdf.min(),
            self.surface_sdf.mean(),
            self.surface_sdf.max(),
        )
        logging.info(
            "Min-mean-max of SDF before clipping: " f"{sdf_min}, {sdf_mean}, {sdf_max}."
        )
        if np.isfinite(clip_sdf):
            self.surface_sdf = np.clip(self.surface_sdf, -clip_sdf, clip_sdf)

        self.max_sdf = np.max(np.abs(self.surface_sdf))

        logging.info(f"SDF dataset loaded with {len(self.surface_coords)} points.")
        logging.info(f"Dataset size: {len(self)} batches.")
        sdf_min, sdf_mean, sdf_max = (
            self.surface_sdf.min(),
            self.surface_sdf.mean(),
            self.surface_sdf.max(),
        )
        logging.info(
            "Min-mean-max of SDF after clipping: " f"{sdf_min}, {sdf_mean}, {sdf_max}."
        )

        logging.warning("This dataset class was abandoned.")
        logging.warning("The current evaluator class does not work with SDFs.")

    def __getitem__(self, _: int) -> dict[str, np.ndarray]:

        off_num_samples = int(self.batch_size * self.off_surface_ratio)
        on_num_samples = self.batch_size - off_num_samples

        coords = np.empty((self.batch_size, 3), dtype=np.float32)
        sdf_values = np.empty((self.batch_size, 1), dtype=np.float32)

        # These values might not be correct if we land near the actual surface.
        flat_idx = self.rng.integers(
            0, self.volume_size**3, size=off_num_samples, endpoint=False
        )
        z, y, x = np.unravel_index(flat_idx, (self.volume_size,) * 3)
        coords[:off_num_samples] = np.stack([x, y, z], axis=-1).astype(np.float32)
        sdf_values[:off_num_samples] = self.max_sdf

        surface_idx = self.rng.choice(len(self.surface_coords), size=on_num_samples)
        coords[off_num_samples:] = self.surface_coords[surface_idx]
        sdf_values[off_num_samples:] = self.surface_sdf[surface_idx]

        coords = (coords / (self.volume_size - 1)) * 2 - 1

        return {
            "inputs": coords,
            "sdf": sdf_values,
        }

    @property
    def num_samples(self):
        return int(len(self.surface_coords) / (1 - self.off_surface_ratio))

    def __len__(self) -> int:
        """__len__ method of torch.utils.data.Dataset."""
        return self.num_samples // self.batch_size
