"""The Learned Perceptual Image Patch Similarity."""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ib.utils.geometry import color_mesh_to_normal_direction
from ib.utils.logging_module import logging

# This is about self.load_state_dict(torch.load(model_path,
# map_location="cpu"), strict=False) being dangerous.
warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics")


def _load_json(cam_params_path: Path) -> dict:
    with open(cam_params_path, "r") as f:
        cam_params = json.load(f)
    return cam_params


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
        cam_params: list[tuple[str, dict]],
    ):
        self.mesh = mesh
        color_mesh_to_normal_direction(self.mesh)
        self.all_cam_params: list[tuple[str, dict]] = cam_params
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        )

    @classmethod
    def from_mesh_dir(cls, base_dir: Path):
        mesh_path = base_dir / "pc_aligned_recon_mesh_0005.ply"
        cam_params_dir = base_dir / "camera_params"

        # Find all camera_params_*.json files
        cam_param_files = list(cam_params_dir.glob("camera_params_*.json"))
        pattern = re.compile(r"camera_params_(.+)\.json")

        cam_params = []
        for cam_file in cam_param_files:
            match = pattern.match(cam_file.name)
            if match:
                mode = match.group(1)
                cam_params_dict = _load_json(cam_file)
                cam_params.append((mode, cam_params_dict))

        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        if not cam_params_dir.exists() or len(cam_params) == 0:
            raise FileNotFoundError(
                f"Camera parameters files not found in: {cam_params_dir}. "
            )

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        return cls(mesh, cam_params)

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
        color_mesh_to_normal_direction(other_mesh)

        results = {}
        aggregate_lpips = []
        for mode, cam_params in self.all_cam_params:
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
            if mode not in {"low", "high"}:
                aggregate_lpips.append(float(lpips_score.item()))

        if len(aggregate_lpips) > 0:
            results[f"metrics_main/lpips"] = float(np.mean(aggregate_lpips))

        return results
