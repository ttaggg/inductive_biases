"""The Learned Perceptual Image Patch Similarity."""

import json
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ib.utils.logging_module import logging

# This is about self.load_state_dict(torch.load(model_path,
# map_location="cpu"), strict=False) being dangerous.
warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics")


def _set_unit_normals(mesh: o3d.geometry.TriangleMesh) -> None:
    # Check if vertex normals exist and are valid
    if len(mesh.vertex_normals) == 0 or len(mesh.vertex_normals) != len(mesh.vertices):
        mesh.compute_vertex_normals()

    normals = np.array(mesh.vertex_normals)

    norms = np.linalg.norm(normals, axis=1)
    if np.any(norms == 0) or np.any(np.isnan(norms)):
        mesh.compute_vertex_normals()
        normals = np.array(mesh.vertex_normals)
        norms = np.linalg.norm(normals, axis=1)

    normals = normals / norms[:, None]
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)


def _set_vertex_colors(mesh: o3d.geometry.TriangleMesh) -> None:
    normals = np.array(mesh.vertex_normals)
    normal_colors = (normals + 1.0) / 2.0
    mesh.vertex_colors = o3d.utility.Vector3dVector(normal_colors)


def _parse_cam_params(cam_params: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    width = cam_params["intrinsic"]["width"]
    height = cam_params["intrinsic"]["height"]
    intrinsic_matrix = np.array(cam_params["intrinsic"]["intrinsic_matrix"])
    extrinsic_matrix = np.array(cam_params["extrinsic"])
    return intrinsic_matrix, extrinsic_matrix, width, height


def o3d_to_tensor(image: o3d.geometry.Image) -> torch.Tensor:
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


class LpipsMetric:
    """The Learned Perceptual Image Patch Similarity."""

    def __init__(
        self,
        mesh: o3d.geometry.TriangleMesh,
        cam_params_low: dict,
        cam_params_high: dict,
    ):
        self.mesh = mesh
        self.prepare_mesh(self.mesh)
        self.cam_params_low = cam_params_low
        self.cam_params_high = cam_params_high
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        )

    def prepare_mesh(self, mesh: o3d.geometry.TriangleMesh) -> None:
        _set_unit_normals(mesh)
        _set_vertex_colors(mesh)

    @classmethod
    def from_mesh_dir(cls, base_dir: Path):
        mesh_path = base_dir / "pc_aligned_recon_mesh_0005.ply"
        cam_params_path_low = base_dir / "camera_params_low.json"
        cam_params_path_high = base_dir / "camera_params_high.json"

        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        if not cam_params_path_low.exists():
            raise FileNotFoundError(
                f"Camera parameters file not found: {cam_params_path_low}"
            )
        if not cam_params_path_high.exists():
            raise FileNotFoundError(
                f"Camera parameters file not found: {cam_params_path_high}"
            )

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        with open(cam_params_path_low, "r") as f:
            cam_params_low = json.load(f)
        with open(cam_params_path_high, "r") as f:
            cam_params_high = json.load(f)
        return cls(mesh, cam_params_low, cam_params_high)

    def extract_image_from_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        width: int,
        height: int,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        output_path: Path | None = None,
    ) -> torch.Tensor:

        # Create offscreen renderer
        render = rendering.OffscreenRenderer(width, height)

        # Add the mesh to the scene with a basic material
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"

        render.scene.add_geometry("mesh", mesh, material)
        render.scene.scene.set_sun_light([0.707, 0.0, -0.707], [1.0, 1.0, 1.0], 75000)
        render.scene.scene.enable_sun_light(True)

        # Extract camera position from the model matrix
        camera_position = extrinsic[:3, 3]
        forward = -extrinsic[:3, 2]
        up = extrinsic[:3, 1]

        # Calculate the look-at point (arbitrary distance along forward direction)
        look_at = camera_position + forward

        # Extract field of view from projection matrix
        fov_rad = 2 * np.arctan(1.0 / intrinsic[1, 1])
        fov_deg = np.degrees(fov_rad)

        # Setup camera
        render.setup_camera(
            fov_deg, look_at.tolist(), camera_position.tolist(), up.tolist()
        )

        # Render the image
        img = render.render_to_image()

        if output_path is not None:
            o3d.io.write_image(str(output_path), img, 9)
            logging.info(f"Image saved to {output_path}")

        return o3d_to_tensor(img)

    def __call__(
        self, other_mesh: o3d.geometry.TriangleMesh, save_path: Path
    ) -> dict[str, float]:
        self.prepare_mesh(other_mesh)

        results = {}
        for mode, cam_params in (
            ("low", self.cam_params_low),
            ("high", self.cam_params_high),
        ):
            intrinsic, extrinsic, width, height = _parse_cam_params(cam_params)
            gt_image = self.extract_image_from_mesh(
                self.mesh,
                width,
                height,
                extrinsic,
                intrinsic,
                save_path.with_name(f"{save_path.stem}_gt_{mode}.png"),
            )
            pred_image = self.extract_image_from_mesh(
                other_mesh,
                width,
                height,
                extrinsic,
                intrinsic,
                save_path.with_name(f"{save_path.stem}_pred_{mode}.png"),
            )
            lpips_score = self.lpips(pred_image, gt_image)
            results[f"metrics_main/lpips_{mode}"] = float(lpips_score.item())

        return results
