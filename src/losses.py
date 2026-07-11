from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t).pow(self.gamma) * bce).mean()


class BCEWithLogitsLossWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets.float())


def build_loss(config: dict) -> nn.Module:
    if config.get("type", "focal") == "bce":
        return BCEWithLogitsLossWrapper()
    return FocalLoss(alpha=float(config.get("focal_alpha", 0.75)), gamma=float(config.get("focal_gamma", 2.0)))


def compute_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    aux_losses: dict[str, torch.Tensor] | None = None,
    loss_fn: nn.Module | None = None,
    lambda_mmd: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_fn = loss_fn or FocalLoss()
    cls_loss = loss_fn(logits, labels.float())
    total = cls_loss
    log = {"classification": float(cls_loss.detach().cpu())}
    aux_losses = aux_losses or {}
    if "mmd" in aux_losses:
        mmd = aux_losses["mmd"]
        total = total + lambda_mmd * mmd
        log["mmd"] = float(mmd.detach().cpu())
    for name, value in aux_losses.items():
        if name == "mmd":
            continue
        total = total + value
        log[name] = float(value.detach().cpu())
    log["total"] = float(total.detach().cpu())
    return total, log
