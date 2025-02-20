"""Siren SDF loss.

Adapted from:
https://github.com/vsitzmann/siren
"""

import torch
import torch.nn.functional as F
from torch import nn


def compute_gradients(pred_sdf: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(
        pred_sdf,
        coords,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
    )[0]
    return grad


class SirenSdfLoss(nn.Module):
    def __init__(self, lambda_1: float, lambda_2: float, lambda_3: float) -> None:
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        super().__init__()

    def forward(
        self,
        model: nn.Module,
        model_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        coords = model_inputs["inputs"].detach().requires_grad_(True)
        on_surface_mask = model_inputs["sdf"]
        gt_normals = model_inputs["normals"]

        pred_sdf = model(coords)

        # Gradient constraint.
        gradient = compute_gradients(pred_sdf, coords)
        grad_constraint = torch.abs(gradient.norm(dim=-1) - 1.0).mean()

        # SDF constraint: only penalize known SDF points:
        # for points on the surface (mask=True), the constraint is pred_sdf,
        # for points off the surface (mask=False) it is zero.
        sdf_constraint = torch.where(
            on_surface_mask, pred_sdf, torch.zeros_like(pred_sdf)
        )
        sdf_constraint = torch.abs(sdf_constraint).mean()

        # Normal constraint: only penalize points on the surface.
        normal_constraint = torch.where(
            on_surface_mask,
            1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
            torch.zeros_like(gradient[..., :1]),
        )
        normal_constraint = normal_constraint.mean()

        # "Interior" constraint: for points off the surface (mask=False).
        inter_constraint = torch.where(
            on_surface_mask,
            torch.zeros_like(pred_sdf),
            torch.exp(-1e2 * torch.abs(pred_sdf)),
        )
        inter_constraint = inter_constraint.mean()

        return {
            "losses/grad": grad_constraint * self.lambda_1,
            "losses/sdf": sdf_constraint * self.lambda_2,
            "losses/normal": normal_constraint * self.lambda_3,
            "losses/inter": inter_constraint * self.lambda_3,
        }
