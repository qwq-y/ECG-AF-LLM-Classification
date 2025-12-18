from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    metric_weight: float = 10.0,
    accuracy_weight: float = 1.0,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Binary classification loss enhanced with differentiable precision/recall/F1 penalties."""
    orig_dtype = logits.dtype
    logits = logits.float()
    targets = targets.float()
    probs = torch.sigmoid(logits)

    tp = torch.sum(probs * targets)
    fp = torch.sum(probs * (1.0 - targets))
    fn = torch.sum((1.0 - probs) * targets)
    tn = torch.sum((1.0 - probs) * (1.0 - targets))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    metric_penalty = (1.0 - precision) + (1.0 - recall) + (1.0 - f1)
    accuracy_penalty = 1.0 - accuracy

    bce = F.binary_cross_entropy_with_logits(logits, targets)
    total_loss = bce + metric_weight * metric_penalty + accuracy_weight * accuracy_penalty

    metrics: Dict[str, float] = {
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f1": float(f1.item()),
        "accuracy": float(accuracy.item()),
        "loss_bce": float(bce.item()),
        "loss_metric_penalty": float(metric_penalty.item()),
        "loss_accuracy_penalty": float(accuracy_penalty.item()),
        "tp": float(tp.item()),
        "fp": float(fp.item()),
        "fn": float(fn.item()),
        "tn": float(tn.item()),
        "count": float(targets.numel()),
    }
    return total_loss.to(orig_dtype), metrics


def aggregate_metrics(
    totals: Dict[str, float],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Aggregate running tp/fp/fn/tn counts into precision/recall/F1/accuracy."""
    tp = totals.get("tp", 0.0)
    fp = totals.get("fp", 0.0)
    fn = totals.get("fn", 0.0)
    tn = totals.get("tn", 0.0)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
