"""Data related utils."""

from pathlib import Path
from typing import Callable, Optional
from typing_extensions import deprecated

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement

from ib.utils.logging_module import logging


@deprecated("Do not use obj files")
def load_obj(
    file_path: str,
    field_func: dict[str, Callable],
) -> tuple[np.ndarray, ...]:
    """Load OBJ pointcloud.

    Args:
        file_path (str): path to the file.
        field_func (Dict[str, Callable]): specification which fields
            to read from file (key) and which function to apply (value).
    """
    parsed_data = {k: [] for k in field_func.keys()}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if not parts or parts[0] not in parsed_data:
                continue
            key = parts[0]
            value = tuple(map(field_func[key], parts[1:]))

            if key == "f" and len(value) > 3:
                # Triangulate n-vertex faces
                for i in range(1, len(value) - 1):
                    parsed_data[key].append((value[0], value[i], value[i + 1]))
            else:
                parsed_data[key].append(value)

    return [np.array(parsed_data[key]) for key in field_func]


def load_pointcloud(file_path: Path) -> dict[str, np.ndarray]:
    if file_path.suffix == ".xyz":
        pc_data = load_xyz(file_path)
    elif file_path.suffix == ".ply":
        pc_data = load_ply(file_path)
    else:
        raise ValueError(
            "Only .xyz and .ply are supported in evaluation, "
            f"given: {file_path.suffix}."
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


def load_xyz(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load XYZ pointcloud like SIREN authors."""
    point_cloud = np.genfromtxt(file_path)
    points = point_cloud[:, :3]
    normals = point_cloud[:, 3:]
    return points, normals


def make_o3d_mesh(vertices: np.ndarray, faces: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Create mesh object from vertices and faces."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh


@deprecated("Do not use obj files")
def write_obj(
    file_path: Path,
    field_data: dict[str, np.array],
) -> None:
    """Write OBJ pointcloud.

    Args:
        file_path (str): path to the file.
        field_data (Dict[str, np.array]): name of the field to write and
            its values in the array, line by line.
    """
    with open(file_path, "w") as outfile:
        for field, data in field_data.items():
            for row in data:
                row = list(map(str, row))
                outfile.write(f"{field} {' '.join(row)}\n")


def write_xyz(file_path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    """Write XYZ pointcloud like SIREN authors."""
    point_cloud = np.concatenate([points, normals], axis=-1)
    np.savetxt(file_path, point_cloud)


def write_ply(
    file_path: Path,
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> None:
    """Write PLY pointcloud."""

    # Define PLY dtype
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
    ]
    if normals is not None:
        dtype.extend(
            [
                ("nx", "f4"),
                ("ny", "f4"),
                ("nz", "f4"),
            ]
        )
    if colors is not None:
        dtype.extend(
            [
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
    if labels is not None:
        dtype.extend(
            [
                ("label", "i4"),
            ]
        )
    ply_dtype = np.dtype(dtype)

    # Build structured array
    vertex_array = np.empty(len(points), dtype=ply_dtype)
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

    ply_el = PlyElement.describe(vertex_array, "vertex")
    PlyData([ply_el]).write(str(file_path))
    logging.info(f"Saved pointcloud PLY to {file_path}")
