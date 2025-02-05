"""Resample points on OBJ mesh."""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from ib.utils.data import load_obj, write_obj
from ib.utils.logging_module import logging


class SamplingException(Exception):
    """Raise sampling exception."""


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
        3. number of samples is bigger or equal to number of faces
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.vertices = vertices.astype(np.float32)
        self.faces = faces.astype(np.int32)
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
        if 2 * len(self.faces) > num_samples:
            raise SamplingException(
                f"Number of samples {num_samples} is fewer than "
                f"number of faces {len(self.faces)} x 2."
            )

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

        # Step 1: Sample one point per face.
        per_face_points, per_face_normals = self.sample_points_normals_from_faces(
            num_samples=len(self.faces),
            face_probs=None,
        )

        # Step 2: Sample one point per face.
        per_face_points2, per_face_normals2 = self.sample_points_normals_from_faces(
            num_samples=len(self.faces),
            face_probs=None,
        )

        # Step 3: Sample additional points proportionally to face areas.
        additional_samples = num_samples - len(self.faces) - len(self.faces)
        additional_points, additional_normals = self.sample_points_normals_from_faces(
            num_samples=additional_samples,
            face_probs=face_probs,
        )

        self.sampled_vertices = np.concatenate(
            (per_face_points, per_face_points2, additional_points)
        )
        self.sampled_normals = np.concatenate(
            (per_face_normals, per_face_normals2, additional_normals)
        )

    def sample_points_normals_from_faces(
        self,
        num_samples: int,
        face_probs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Helper function to sample points and normals from faces."""

        if face_probs is None:
            sampled_faces = range(len(self.faces))
        else:
            sampled_faces = np.random.choice(
                len(self.faces), size=num_samples, p=face_probs
            )

        sampled_points = np.empty((num_samples, 3), dtype=np.float32)
        sampled_normals = np.empty((num_samples, 3), dtype=np.float32)

        with tqdm(
            total=len(sampled_faces),
            desc="Sampling vertices and normals",
            unit=" vertices",
            dynamic_ncols=True,
            disable=not sys.stdout.isatty(),
        ) as pbar:
            for i, face_idx in enumerate(sampled_faces):
                v0, v1, v2 = self.vertices[self.faces[face_idx]]
                sampled_points[i] = sample_point_in_triangle(v0, v1, v2)
                sampled_normals[i] = compute_face_normal(v0, v1, v2)
                pbar.update(1)

        return sampled_points, sampled_normals

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
