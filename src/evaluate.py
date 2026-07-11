from __future__ import annotations

import csv
import json
import os
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import load_config, save_json
from .dataset import CPCLTextCentricDataset, make_collate_fn
from .metrics import best_threshold_on_dev, compute_metrics
from .model import TextCentricCPCLModel


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[list[float], list[float], list[dict], dict]:
    model.eval()
    labels, probs, rows = [], [], []
    gate_accum: dict[str, list[float]] = {}
    for batch in loader:
        batch["label"] = batch["label"].to(device)
        out = model(batch)
        p = out["prob"].detach().cpu().tolist()
        y = batch["label"].detach().cpu().tolist()
        labels.extend(y)
        probs.extend(p)
        for k, v in out.get("gate_stats", {}).items():
            gate_accum.setdefault(k, []).append(float(v))
        for i, vid in enumerate(batch["video_id"]):
            rows.append({
                "video_id": vid,
                "label": y[i],
                "prob": p[i],
                "split": batch["split"][i],
                "group": batch["group"][i],
                "num_comments": len(batch["comments"][i]),
                "has_visual": int(batch["modality_mask"]["visual"][i].item()),
                "has_audio": int(batch["modality_mask"]["audio"][i].item()),
                "has_face": int(batch["modality_mask"]["face"][i].item()),
                "has_knowledge": int(batch["modality_mask"]["knowledge"][i].item()),
            })
    gate_stats = {k: sum(v) / len(v) for k, v in gate_accum.items() if v}
    return labels, probs, rows, gate_stats


def write_predictions(rows: list[dict], path: str, threshold: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["video_id", "label", "prob", "pred", "split", "group", "num_comments", "has_visual", "has_audio", "has_face", "has_knowledge"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["pred"] = int(float(out["prob"]) >= threshold)
            writer.writerow(out)


def classification_report_text(metrics: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in metrics.items())


def evaluate_checkpoint(config_path: str, checkpoint: str, split: str = "test") -> dict[str, Any]:
    config = load_config(config_path)
    device = config["train"].get("device", "cpu")
    dataset = CPCLTextCentricDataset(config["data"]["index_csv"], split=split, config=config)
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=False, collate_fn=make_collate_fn(config))
    model = TextCentricCPCLModel(config).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    labels, probs, _, gate_stats = predict(model, loader, device)
    threshold = state.get("threshold", config["metrics"].get("threshold", 0.5)) if isinstance(state, dict) else config["metrics"].get("threshold", 0.5)
    metrics = compute_metrics(labels, probs, float(threshold))
    metrics["gate_stats"] = gate_stats
    return metrics
