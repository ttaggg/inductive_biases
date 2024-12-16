"""Siren SDF loss.

Adapted from:
https://github.com/vsitzmann/siren
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


def compute_gradients(
    y: torch.Tensor, x: torch.Tensor, grad_outputs: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class SirenSdfLoss(nn.Module):
    def __init__(self, lambda_1: float, lambda_2: float, lambda_3: float) -> None:
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        super().__init__()

    def forward(
        self,
        model_inputs: Dict[str, torch.Tensor],
        model_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        gt_sdf = model_inputs["sdf"]
        gt_normals = model_inputs["normals"]

        coords = model_outputs["coords"]
        pred_sdf = model_outputs["output"]

        gradient = compute_gradients(pred_sdf, coords)

        # Wherever boundary_values is not equal to zero,
        # we interpret it as a boundary constraint.
        grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

        sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))

        normal_constraint = torch.where(
            gt_sdf != -1,
            1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
            torch.zeros_like(gradient[..., :1]),
        )

        inter_constraint = torch.where(
            gt_sdf != -1,
            torch.zeros_like(pred_sdf),
            torch.exp(-1e2 * torch.abs(pred_sdf)),
        )
        return {
            "grad": grad_constraint.mean() * self.lambda_1,  # 50
            "sdf": torch.abs(sdf_constraint).mean() * self.lambda_2,  # 3000
            "normal": normal_constraint.mean() * self.lambda_3,  # 100
            "inter": inter_constraint.mean() * self.lambda_3,  # 100 or 3000 (?!)
        }
