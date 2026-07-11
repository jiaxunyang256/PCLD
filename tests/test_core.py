import numpy as np
import torch

from src.losses import FocalLoss
from src.metrics import compute_metrics


def test_focal_loss_is_finite():
    loss = FocalLoss()(torch.tensor([0.2, -0.4]), torch.tensor([1.0, 0.0]))
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_metrics_contains_primary_scores():
    result = compute_metrics(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.7, 0.2]))
    assert result["macro_f1"] == 1.0
    assert result["auc"] == 1.0
