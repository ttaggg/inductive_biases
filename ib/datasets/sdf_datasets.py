"""Datasets that loads SDF volume."""

import numpy as np
from torch.utils.data import Dataset


class SdfDataset(Dataset):
    """Dataset class for SDF data."""

    def __init__(
        self,
        file_path: str,
        batch_size: int,
        off_surface_ratio: float,
        clip_sdf: float = np.inf,
    ) -> None:
        self.sdf = np.load(file_path)
        self.dim = self.sdf.shape
        self.batch_size = batch_size
        self.off_surface_ratio = off_surface_ratio

        # Clamp SDF.
        self.sdf = np.clip(self.sdf, -clip_sdf, clip_sdf)

        # Precompute for future.
        sdf_threshold = 5 * (2.0 / self.dim[0])
        self.surface_indices = np.array(np.where(np.abs(self.sdf) < sdf_threshold)).T

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

    def __len__(self) -> int:
        """__len__ method of torch.utils.data.Dataset."""
        return (
            int(len(self.surface_indices) / (1 - self.off_surface_ratio))
            // self.batch_size
        )
