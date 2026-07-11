from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "experiment_name": "textcentric_debug",
    "output_root": "outputs",
    "annotation_csv": "data/Annotation.csv",
    "data": {
        "index_csv": "data/index.csv",
        "asr_root": "data/TXT",
        "comments_root": "data/comments",
        "text_feature_root": "features/TEXT_features",
        "audio_feature_root": "features/AUDIO_features",
        "video_feature_root": "features/VIT_features",
        "face_feature_root": "features/extracted_features_without_xml",
        "skg_root": "data/skg",
        "comments_mode": "all",
        "comment_remove_keywords": ["CPCL", "PCL", "歧视", "低俗", "嘲讽", "看不起", "可怜", "同情", "高高在上", "爹味", "凝视", "冒犯"],
        "max_visual_len": 64,
        "max_audio_len": 8,
        "max_face_len": 64,
        "debug_small_sample": None,
    },
    "model": {
        "hidden_dim": 128,
        "text_dim": 768,
        "visual_dim": 768,
        "audio_dim": 40,
        "face_dim": 768,
        "dropout": 0.1,
        "use_asr": True,
        "use_comments": True,
        "use_text_features": True,
        "use_visual": False,
        "use_audio": False,
        "use_face": False,
        "use_knowledge": False,
        "use_mamba": False,
        "fusion_strategy": "gated",
    },
    "evidence_encoder": {
        "type": "none",
        "num_layers": 1,
        "num_heads": 4,
        "max_len": 256,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
    },
    "alignment": {"method": "interpolate", "use_mmd": False, "lambda_mmd": 0.05},
    "knowledge": {
        "mode": "lightweight",
        "global_skg_path": "data/skg/mined_skg_chinese_with_label.json",
        "input_dim": 768,
        "max_matches": 64,
        "bert_model_name": "bert-base-chinese",
        "sentence_model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "similarity_threshold": 0.6,
        "max_length": 256,
    },
    "loss": {"type": "focal", "focal_alpha": 0.75, "focal_gamma": 2.0},
    "train": {
        "batch_size": 8,
        "epochs": 5,
        "lr": 1.0e-4,
        "weight_decay": 1.0e-3,
        "gradient_clip": 1.0,
        "patience": 3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "metrics": {"threshold": 0.5, "tune_threshold_on_dev": True},
}


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        config = deep_update(config, loaded)
    if overrides:
        config = deep_update(config, overrides)
    return config


def as_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{k: as_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [as_namespace(v) for v in value]
    return value


def save_config(config: Mapping[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, allow_unicode=True, sort_keys=False)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug_small_sample", type=int, default=None)
    return parser.parse_args()
