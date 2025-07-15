"""Data related utils."""

from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement

from ib.utils.logging_module import logging


def load_pointcloud(file_path: Path) -> dict[str, np.ndarray]:
    if file_path.suffix == ".ply":
        pc_data = load_ply(file_path)
    else:
        raise ValueError(
            "Only .ply are supported in evaluation, " f"given: {file_path.suffix}."
        )
    return pc_data


def load_ply(file_path: str) -> dict[str, np.ndarray]:
    """Loads data from a PLY file."""
    ply_data = PlyData.read(file_path)
    ply = ply_data["vertex"]
    data = {}
    data["points"] = np.vstack((ply["x"], ply["y"], ply["z"])).T
    if "nx" in ply:
        data["normals"] = np.vstack((ply["nx"], ply["ny"], ply["nz"])).T
    if "red" in ply:
        data["colors"] = np.vstack((ply["red"], ply["green"], ply["blue"])).T
    if "label" in ply:
        data["labels"] = ply["label"]

    if "face" in ply_data:
        face_data = ply_data["face"].data
        prop_name = face_data.dtype.names[0]
        face_indices = face_data[prop_name]
        faces_list = []
        for f in face_indices:
            if len(f) == 3:
                faces_list.append(tuple(f))
            else:
                for i in range(1, len(f) - 1):
                    faces_list.append((f[0], f[i], f[i + 1]))
        data["faces"] = np.array(faces_list, dtype=np.int64)

    return data


def make_o3d_mesh(vertices: np.ndarray, faces: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Create mesh object from vertices and faces."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh


def write_ply(
    file_path: Path,
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    face_colors: Optional[np.ndarray] = None,
) -> None:
    """Write PLY pointcloud."""

    # Define PLY dtype
    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
    ]
    if normals is not None:
        vertex_dtype.extend(
            [
                ("nx", "f4"),
                ("ny", "f4"),
                ("nz", "f4"),
            ]
        )
    if colors is not None:
        vertex_dtype.extend(
            [
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
    if labels is not None:
        vertex_dtype.extend(
            [
                ("label", "i4"),
            ]
        )
    ply_vertex_dtype = np.dtype(vertex_dtype)

    # Build vertex structured array
    vertex_array = np.empty(len(points), dtype=ply_vertex_dtype)
    vertex_array["x"] = points[:, 0]
    vertex_array["y"] = points[:, 1]
    vertex_array["z"] = points[:, 2]
    if normals is not None:
        vertex_array["nx"] = normals[:, 0]
        vertex_array["ny"] = normals[:, 1]
        vertex_array["nz"] = normals[:, 2]
    if colors is not None:
        vertex_array["red"] = colors[:, 0]
        vertex_array["green"] = colors[:, 1]
        vertex_array["blue"] = colors[:, 2]
    if labels is not None:
        vertex_array["label"] = labels

    # Create PLY elements list
    ply_elements = [PlyElement.describe(vertex_array, "vertex")]

    # Handle faces if provided
    if faces is not None:
        # Define face PLY dtype
        face_dtype = [("vertex_indices", "i4", (3,))]
        if face_colors is not None:
            face_dtype.extend(
                [
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ]
            )
        ply_face_dtype = np.dtype(face_dtype)

        # Build face structured array
        face_array = np.empty(len(faces), dtype=ply_face_dtype)
        face_array["vertex_indices"] = faces
        if face_colors is not None:
            face_array["red"] = face_colors[:, 0]
            face_array["green"] = face_colors[:, 1]
            face_array["blue"] = face_colors[:, 2]

        ply_elements.append(PlyElement.describe(face_array, "face"))

    # Write PLY file
    PlyData(ply_elements).write(str(file_path))
    mesh_info = "mesh" if faces is not None else "pointcloud"
    logging.info(f"Saved {mesh_info} PLY to {file_path}")
