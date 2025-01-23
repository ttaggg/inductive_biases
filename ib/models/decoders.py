"""INR decoders."""

from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from skimage import measure

from ib.utils.logging_module import logging
from ib.utils.model import query_model


class SdfDecoder:
    """Decode INR representing SDF to the corresponding mesh."""

    def __init__(self, model_path: Path, device: str) -> None:
        self.model = torch.load(model_path, weights_only=False, map_location=device)
        self.model.eval()
        self.device = device
        self.mesh = None

    def run(self, resolution: int, batch_size: int) -> None:
        """Decode INR.

        1. Run model on the dense grid of a given resolution.
        2. Perform Marching Cubes on the SDF.
        3. Initialize TriangleMesh mesh.
        """

        # Get mesh.
        sdf = query_model(self.model, resolution, batch_size, self.device)
        spacing = (1.0 / resolution, 1.0 / resolution, 1.0 / resolution)
        verts, faces, _, _ = measure.marching_cubes(sdf, level=0, spacing=spacing)

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
