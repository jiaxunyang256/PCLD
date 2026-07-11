from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, save_config, save_json, set_seed
from src.dataset import CPCLTextCentricDataset, make_collate_fn
from src.evaluate import classification_report_text, predict, write_predictions
from src.losses import build_loss, compute_total_loss
from src.metrics import best_threshold_on_dev, compute_metrics
from src.model import TextCentricCPCLModel


def make_loader(config: dict, split: str, shuffle: bool) -> DataLoader:
    dataset = CPCLTextCentricDataset(config["data"]["index_csv"], split, config)
    if not dataset:
        raise ValueError(f"No samples found for split={split}")
    return DataLoader(dataset, batch_size=int(config["train"]["batch_size"]), shuffle=shuffle,
                      collate_fn=make_collate_fn(config))


def run(config: dict) -> Path:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = str(config["train"].get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    out_dir = ROOT / config.get("output_root", "outputs") / config["experiment_name"] / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(out_dir / "config.yaml"))
    loaders = {s: make_loader(config, s, s == "train") for s in ("train", "dev", "test")}
    model = TextCentricCPCLModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["lr"]),
                                  weight_decay=float(config["train"]["weight_decay"]))
    loss_fn = build_loss(config["loss"])
    best_score, bad_epochs = -1.0, 0
    best_path = out_dir / "best.pt"

    for epoch in range(int(config["train"]["epochs"])):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            batch["label"] = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, _ = compute_total_loss(output["logits"], batch["label"], output.get("losses"), loss_fn,
                                         float(config["alignment"].get("lambda_mmd", 0.05)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["gradient_clip"]))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        labels, probs, _, _ = predict(model, loaders["dev"], device)
        threshold, metrics = best_threshold_on_dev(labels, probs)
        metrics.update(epoch=epoch + 1, train_loss=sum(train_losses) / max(1, len(train_losses)))
        print(json.dumps(metrics, ensure_ascii=False))
        score = float(metrics[config["train"].get("selection_metric", "macro_f1")])
        if score > best_score:
            best_score, bad_epochs = score, 0
            torch.save({"model": model.state_dict(), "threshold": threshold, "config": config}, best_path)
            save_json(metrics, str(out_dir / "dev_metrics.json"))
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["train"].get("patience", 5)):
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    labels, probs, rows, gate_stats = predict(model, loaders["test"], device)
    metrics = compute_metrics(labels, probs, float(checkpoint["threshold"]))
    save_json(metrics, str(out_dir / "test_metrics.json"))
    save_json(gate_stats, str(out_dir / "gate_stats_test.json"))
    write_predictions(rows, str(out_dir / "predictions_test.csv"), float(checkpoint["threshold"]))
    (out_dir / "classification_report.txt").write_text(classification_report_text(metrics), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PCLD text-centric evidence model")
    parser.add_argument("--config", default="configs/pclmmplus.yaml")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    print(f"Artifacts: {run(config)}")


if __name__ == "__main__":
    main()
