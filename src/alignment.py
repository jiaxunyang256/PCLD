from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def interpolate_align_to_text(evidence: torch.Tensor, text_hidden: torch.Tensor) -> torch.Tensor:
    if evidence is None:
        return None
    if evidence.size(1) == text_hidden.size(1):
        return evidence
    x = evidence.transpose(1, 2)
    x = F.interpolate(x, size=text_hidden.size(1), mode="linear", align_corners=False)
    return x.transpose(1, 2)


def cosine_align_to_text(evidence: torch.Tensor, text_hidden: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(F.normalize(text_hidden, dim=-1), F.normalize(evidence, dim=-1).transpose(1, 2))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, evidence)


def align_evidence_to_text(
    evidence: torch.Tensor | None,
    text_hidden: torch.Tensor,
    method: str = "interpolate",
) -> torch.Tensor | None:
    if evidence is None:
        return None
    if method == "cosine":
        return cosine_align_to_text(evidence, text_hidden)
    return interpolate_align_to_text(evidence, text_hidden)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, bandwidths=(1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    dist = torch.cdist(x, y, p=2).pow(2)
    kernels = [torch.exp(-dist / (2 * bw)) for bw in bandwidths]
    return sum(kernels) / len(kernels)


def mmd_loss(x: torch.Tensor | None, y: torch.Tensor | None, mask: torch.Tensor | None = None) -> torch.Tensor:
    if x is None or y is None:
        device = y.device if y is not None else torch.device("cpu")
        return torch.zeros((), device=device)
    if mask is not None:
        keep = mask.float().view(-1) > 0
        if keep.sum() == 0:
            return torch.zeros((), device=x.device)
        x = x[keep]
        y = y[keep]
    x = x.reshape(-1, x.size(-1))
    y = y.reshape(-1, y.size(-1))
    if x.numel() == 0 or y.numel() == 0:
        return torch.zeros((), device=x.device)
    k_xx = _rbf_kernel(x, x).mean()
    k_yy = _rbf_kernel(y, y).mean()
    k_xy = _rbf_kernel(x, y).mean()
    return torch.clamp(k_xx + k_yy - 2 * k_xy, min=0.0)
