"""INR decoders."""

import os

# Fix the conflict between OpenMP versions for PyTorch and Open3D
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import torch
import open3d as o3d
from skimage import measure

from ib.utils.logging_module import logging
from ib.utils.model import query_model


class SdfDecoder:
    """Decode INR representing SDF to the corresponding mesh."""

    def __init__(self, model) -> None:
        self.sdf = None
        self.mesh = None
        self.model = model

    @classmethod
    def from_model_path(cls, model_path: Path, device: str):
        model = torch.load(model_path, weights_only=False, map_location=device)
        model.eval()
        return cls(model)

    def run(self, resolution: int, batch_size: int) -> None:
        """Decode INR.

        1. Run model on the dense grid of a given resolution.
        2. Perform Marching Cubes on the SDF.
        3. Initialize TriangleMesh mesh.
        """

        # Get mesh.
        self.sdf = query_model(self.model, resolution, batch_size, self.model.device)
        spacing = (1.0 / resolution, 1.0 / resolution, 1.0 / resolution)
        verts, faces, _, _ = measure.marching_cubes(self.sdf, level=0, spacing=spacing)

        # Create TriangleMesh object.
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        self.mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        self.mesh.compute_vertex_normals()
        logging.info(f"Mesh contains {len(verts)} vertices and {len(faces)} faces.")

    def save(self, file_path: Path) -> None:
        o3d.io.write_triangle_mesh(file_path, self.mesh)
        logging.info(f"Mesh was written to {file_path}")

    def show(self) -> None:
        """Visualize current mesh."""
        o3d.visualization.draw_geometries([self.mesh], mesh_show_wireframe=True)

    @property
    def vertices(self):
        if self.mesh is None:
            raise ValueError("Mesh was not initialized.")
        return np.asarray(self.mesh.vertices).astype(np.float32)

    @property
    def faces(self):
        if self.mesh is None:
            raise ValueError("Mesh was not initialized.")
        return np.asarray(self.mesh.triangles)
