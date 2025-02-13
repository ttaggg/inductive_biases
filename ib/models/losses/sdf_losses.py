"""Losses for direct SDF regression."""

import torch
from torch import nn


class L2Loss(nn.Module):
    def __init__(self, clip_sdf: float = torch.inf) -> None:
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()
        self.clip_sdf = clip_sdf

    def forward(
        self,
        model: nn.Module,
        model_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        target_sdf = model_inputs["sdf"]
        pred_sdf = model(model_inputs["inputs"])
        pred_sdf = torch.clip(pred_sdf, -self.clip_sdf, self.clip_sdf)

        loss = self.loss_fn(pred_sdf, target_sdf)
        return {"losses/mse": loss.mean()}


class L1Loss(nn.Module):
    def __init__(self, clip_sdf: float = torch.inf) -> None:
        super().__init__()
        self.loss_fn = torch.nn.L1Loss()
        self.clip_sdf = clip_sdf

    def forward(
        self,
        model: nn.Module,
        model_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        target_sdf = model_inputs["sdf"]
        pred_sdf = model(model_inputs["inputs"])
        pred_sdf = torch.clip(pred_sdf, -self.clip_sdf, self.clip_sdf)

        loss = self.loss_fn(pred_sdf, target_sdf)
        return {"losses/mae": loss.mean()}
