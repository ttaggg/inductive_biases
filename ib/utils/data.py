"""Data related utils."""

from pathlib import Path
from typing import Callable, Optional

from plyfile import PlyData, PlyElement

import numpy as np


def filter_incorrect_normals(
    points: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Filter out invalid normals and points with zero normals.
    correct_normals = np.logical_and(
        np.linalg.norm(normals, axis=-1) != 0.0,
        np.all(np.isfinite(normals), axis=-1),
    )
    return points[correct_normals], normals[correct_normals]


def normalize_points_and_normals(
    points: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalizes points and normals.

    Args:
        points (np.ndarray): Array of points (x, y, z).
        normals (np.ndarray): Array of normals (x, y, z).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - points (np.ndarray): Points in the (-1, 1) range with the original aspect ratio.
            - normals (np.ndarray): Unit normals.
    """

    points, normals = filter_incorrect_normals(points, normals)
    points -= np.mean(points, axis=0, keepdims=True)
    # TODO(oleg): consider normalization without preserving aspect ratio.
    coord_max = np.amax(points)
    coord_min = np.amin(points)
    points = (points - coord_min) / (coord_max - coord_min)
    points -= 0.5
    points *= 2.0

    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    return points, normals


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


def load_pointcloud(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if file_path.suffix == ".xyz":
        vertices, normals = load_xyz(file_path)
    elif file_path.suffix == ".ply":
        vertices, normals = load_ply(file_path)
    else:
        raise ValueError(
            "Only .xyz and .ply are supported in evaluation, "
            f"given: {file_path.suffix}."
        )
    return vertices, normals


def load_ply(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads vertices and normals from a PLY file."""
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data["vertex"]
    points = np.vstack((vertex_data["x"], vertex_data["y"], vertex_data["z"])).T
    normals = (
        np.vstack((vertex_data["nx"], vertex_data["ny"], vertex_data["nz"])).T
        if "nx" in vertex_data
        else None
    )
    return points, normals


def load_xyz(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load XYZ pointcloud like SIREN authors."""
    point_cloud = np.genfromtxt(file_path)
    points = point_cloud[:, :3]
    normals = point_cloud[:, 3:]
    return points, normals


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
    alpha_channel: Optional[np.ndarray] = None,
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
    if alpha_channel is not None:
        dtype.extend(
            [
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("alpha", "u1"),
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
    if alpha_channel is not None:
        vertex_array["red"] = 255.0
        vertex_array["green"] = 0.0
        vertex_array["blue"] = 0.0
        vertex_array["alpha"] = alpha_channel

    ply_el = PlyElement.describe(vertex_array, "vertex")
    PlyData([ply_el], text=True).write(str(file_path))
