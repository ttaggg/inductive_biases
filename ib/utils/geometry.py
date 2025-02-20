"""Utils to convert betweeen data types."""

import numpy as np
from skimage import measure

from ib.datasets.resamplers import SimpleResampler


def sdf_to_mesh(sdf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resolution = sdf.shape[0]  # Assume cube.
    spacing = (1.0 / resolution, 1.0 / resolution, 1.0 / resolution)
    verts, faces, _, _ = measure.marching_cubes(sdf, level=0, spacing=spacing)
    verts = 2 * verts - 1
    return verts, faces


def mesh_to_pointcloud(verts: np.ndarray, faces: np.ndarray, num_samples) -> np.ndarray:
    resampler = SimpleResampler(verts, faces)
    resampler.run(num_samples=num_samples)
    return resampler.sampled_vertices


def sdf_to_pointcloud(sdf: np.ndarray, num_samples: int) -> np.ndarray:
    verts, faces = sdf_to_mesh(sdf)
    return mesh_to_pointcloud(verts, faces, num_samples)
