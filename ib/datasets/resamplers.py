"""Resample points on OBJ mesh."""

from pathlib import Path
from typing import Optional

import numpy as np

from ib.utils.data import load_obj, write_obj
from ib.utils.logging_module import logging
from ib.utils.pointcloud import filter_incorrect_normals


class SamplingException(Exception):
    """Raise sampling exception."""


def compute_face_normals(vertices, faces):
    """Compute normals for multiple triangle faces in a vectorized manner."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


def compute_face_areas(vertices, faces):
    """Compute the areas of multiple triangles in a vectorized manner."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross_product = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(cross_product, axis=1)
    return face_areas


def sample_points_in_triangles(vertices, faces, sampled_faces):
    """Sample points uniformly inside multiple triangles using barycentric coordinates."""
    num_samples = len(sampled_faces)

    v0, v1, v2 = (
        vertices[faces[sampled_faces, 0]],
        vertices[faces[sampled_faces, 1]],
        vertices[faces[sampled_faces, 2]],
    )

    # Generate random barycentric coordinates
    r1 = np.sqrt(np.random.rand(num_samples))[:, np.newaxis]
    r2 = np.random.rand(num_samples)[:, np.newaxis]
    u = 1 - r1
    v = r1 * (1 - r2)
    w = r1 * r2

    sampled_points = u * v0 + v * v1 + w * v2

    return sampled_points.astype(np.float32)


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
        if 3 * len(self.faces) > num_samples:
            raise SamplingException(
                f"Number of samples {num_samples} is fewer than "
                f"number of faces {len(self.faces)} x 3."
            )

        # We sample based on the area.
        face_areas = compute_face_areas(self.vertices, self.faces)
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

        self.sampled_vertices, self.sampled_normals, _ = filter_incorrect_normals(
            self.sampled_vertices,
            self.sampled_normals,
        )

    def sample_points_normals_from_faces(
        self,
        num_samples: int,
        face_probs: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Helper function to sample points and normals from faces."""

        if face_probs is None:
            sampled_faces = np.array(range(len(self.faces)))
        else:
            sampled_faces = np.random.choice(
                len(self.faces), size=num_samples, p=face_probs
            )
        sampled_faces = sampled_faces.astype(np.int32)

        face_normals = compute_face_normals(self.vertices, self.faces)
        sampled_points = sample_points_in_triangles(
            self.vertices, self.faces, sampled_faces
        )
        sampled_normals = face_normals[sampled_faces]

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

    def __len__(self):
        if self.sampled_vertices is None:
            return 0
        return len(self.sampled_vertices)


class SimpleResampler(Resampler):

    def run(self, num_samples: int) -> None:
        """Sample vertices and normals."""
        logging.stage("Sampling vertices and normals.")

        face_areas = compute_face_areas(self.vertices, self.faces)
        face_probs = face_areas / face_areas.sum()

        self.sampled_vertices, self.sampled_normals = (
            self.sample_points_normals_from_faces(
                num_samples=num_samples,
                face_probs=face_probs,
            )
        )
        self.sampled_vertices, self.sampled_normals, _ = filter_incorrect_normals(
            self.sampled_vertices,
            self.sampled_normals,
        )
