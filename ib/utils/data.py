"""Data related utils."""

from typing import Tuple

import numpy as np


def normalize_points_and_normals(
    points: np.ndarray,
    normals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalizes points and normals.

    Args:
        points (np.ndarray): Array of points (x, y, z).
        normals (np.ndarray): Array of normals (x, y, z).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - points (np.ndarray): Points in the (-1, 1) range with the original aspect ratio.
            - normals (np.ndarray): Unit normals.
    """
    points -= np.mean(points, axis=0, keepdims=True)
    coord_max = np.amax(points)
    coord_min = np.amin(points)
    points = (points - coord_min) / (coord_max - coord_min)
    points -= 0.5
    points *= 2.0

    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    return points, normals


def load_obj(
    file_path: str,
    field_func: Dict[str, Callable],
) -> Tuple[np.ndarray, ...]:
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


def load_xyz(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load XYZ pointcloud like SIREN authors."""
    point_cloud = np.genfromtxt(file_path)
    points = point_cloud[:, :3]
    normals = point_cloud[:, 3:]
    return points, normals
