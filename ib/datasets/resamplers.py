"""Resample points on OBJ mesh."""

import numpy as np
from pathlib import Path

from ib.utils.data import load_obj, write_obj
from ib.utils.logging_module import logging


def compute_face_normal(v0, v1, v2):
    """Compute the normal of a triangle face given its 3 vertices."""
    normal = np.cross(v1 - v0, v2 - v0)
    unit_normal = normal / np.linalg.norm(normal)
    return unit_normal


def compute_face_area(v0, v1, v2):
    """Compute the area of a triangle given its vertices using cross product."""
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def sample_point_in_triangle(v0, v1, v2):
    """Sample a point uniformly inside a triangle using barycentric coordinates."""
    r1 = np.sqrt(np.random.rand())  # sqrt ensures uniform sampling
    r2 = np.random.rand()
    u = 1 - r1
    v = r1 * (1 - r2)
    w = r1 * r2
    return u * v0 + v * v1 + w * v2


class Resampler:
    """Resample points on mesh.

    NOTE(oleg): this implementation assumes that:
        1. faces are positive indices (starting from 1)
        2. vertices in a face are ordered counterclockwise (CCW)
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.vertices = vertices
        self.faces = faces
        self.sampled_vertices = None
        self.sampled_normals = None

    @classmethod
    def from_obj_file(cls, file_path: Path):
        vertices, faces = load_obj(
            file_path,
            {
                "v": float,  # Vertices.
                "f": lambda x: int(x.split("/")[0]) - 1,  # Faces.
            },
        )
        return cls(vertices, faces)

    def run(self, num_samples: int) -> None:
        """Sample vertices and normals.

        TODO(oleg): explore an option to replace this function
            with mesh.sample_points_uniformly() from open3d.
        """
        logging.stage("Sampling vertices and normals.")

        # We sample based on the area.
        face_areas = np.array(
            [
                compute_face_area(
                    self.vertices[face[0]],
                    self.vertices[face[1]],
                    self.vertices[face[2]],
                )
                for face in self.faces
            ]
        )
        face_probs = face_areas / face_areas.sum()
        sampled_faces = np.random.choice(
            len(self.faces), size=num_samples, p=face_probs
        )

        # Assign normal from the face to the new vertex.
        self.sampled_vertices = np.empty(shape=(num_samples, 3))
        self.sampled_normals = np.empty(shape=(num_samples, 3))
        for i, face_idx in enumerate(sampled_faces):
            v0, v1, v2 = self.vertices[self.faces[face_idx]]
            self.sampled_vertices[i] = sample_point_in_triangle(v0, v1, v2)
            self.sampled_normals[i] = compute_face_normal(v0, v1, v2)

            if i % int(num_samples * 0.1) == 0:
                logging.info(f"{i} / {num_samples} steps are done.")

    def save(self, file_path: Path) -> None:
        field_data = {"v": self.sampled_vertices, "vn": self.sampled_normals}
        write_obj(file_path, field_data)
        logging.info(f"Pointcloud was saved to {file_path}")

    def show(self) -> None:
        """Visualize sampled points and normals with open3d."""
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.sampled_vertices)
        pcd.normals = o3d.utility.Vector3dVector(self.sampled_normals)
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)
