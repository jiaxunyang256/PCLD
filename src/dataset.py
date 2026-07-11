from __future__ import annotations

import csv
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .feature_utils import ensure_2d_feature, pad_sequence_features


TEXT_COLUMNS = ("comment", "comments", "text", "content", "message", "danmaku")
DEFAULT_COMMENT_KEYWORDS = ("CPCL", "PCL", "歧视", "低俗", "嘲讽", "看不起", "可怜", "同情", "高高在上", "爹味", "凝视", "冒犯")


def read_text_file(path: str | float | None) -> str:
    if not path or pd.isna(path) or not os.path.exists(str(path)):
        return ""
    with open(str(path), "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def read_comments_csv(path: str | float | None) -> list[str]:
    if not path or pd.isna(path) or not os.path.exists(str(path)):
        return []
    path = str(path)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            out = []
            for item in obj:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    for col in TEXT_COLUMNS:
                        if col in item and item[col]:
                            out.append(str(item[col]))
                            break
            return out
        if isinstance(obj, dict):
            for col in TEXT_COLUMNS:
                value = obj.get(col)
                if isinstance(value, list):
                    return [str(x) for x in value if str(x).strip()]
                if isinstance(value, str):
                    return [value]
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    rows.append(row[0])
        return [str(x).strip() for x in rows if str(x).strip()]
    column = next((c for c in TEXT_COLUMNS if c in df.columns), None)
    if column is None:
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        column = object_cols[0] if object_cols else df.columns[0]
    return [str(x).strip() for x in df[column].fillna("").tolist() if str(x).strip()]


def load_pickle_feature(path: str | float | None, target_dim: int | None = None):
    if not path or pd.isna(path) or not os.path.exists(str(path)):
        return None
    with open(str(path), "rb") as f:
        obj = pickle.load(f)
    return ensure_2d_feature(obj, default_dim=target_dim)


class CPCLTextCentricDataset(Dataset):
    def __init__(
        self,
        index_csv: str,
        split: str | None = None,
        config: dict[str, Any] | None = None,
        debug_small_sample: int | None = None,
    ) -> None:
        self.index_csv = index_csv
        self.config = config or {}
        self.model_cfg = self.config.get("model", {})
        self.data_cfg = self.config.get("data", {})
        if not os.path.exists(index_csv):
            raise FileNotFoundError(f"index_csv not found: {index_csv}")
        df = pd.read_csv(index_csv)
        if split is not None and "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == split.lower()]
        if debug_small_sample:
            df = df.head(debug_small_sample)
        self.rows = df.reset_index(drop=True).to_dict("records")

    def _apply_comment_mode(self, comments: list[str], video_id: str) -> list[str]:
        mode = str(self.data_cfg.get("comments_mode", "all")).lower()
        if mode in {"all", "full"}:
            return comments
        if mode in {"none", "no_comments"}:
            return []
        if mode == "top1":
            return comments[:1]
        if mode == "top3":
            return comments[:3]
        if mode == "random":
            if not comments:
                return []
            rng = random.Random(f"{self.config.get('seed', 42)}::{video_id}")
            return [rng.choice(comments)]
        if mode in {"keyword_removed", "keyword-removed"}:
            keywords = self.data_cfg.get("comment_remove_keywords") or DEFAULT_COMMENT_KEYWORDS
            filtered = []
            for comment in comments:
                text = str(comment)
                for keyword in keywords:
                    text = text.replace(str(keyword), "")
                if text.strip():
                    filtered.append(text.strip())
            return filtered
        return comments

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        video_id = str(row.get("video_id", ""))
        comments = self._apply_comment_mode(read_comments_csv(row.get("comments_path")), video_id)
        text_dim = int(self.model_cfg.get("text_dim", 768))
        visual_dim = int(self.model_cfg.get("visual_dim", 768))
        audio_dim = int(self.model_cfg.get("audio_dim", 40))
        face_dim = int(self.model_cfg.get("face_dim", 768))
        item = {
            "video_id": video_id,
            "label": float(row.get("label", 0)),
            "split": str(row.get("split", "")),
            "group": str(row.get("group", "")),
            "asr_text": read_text_file(row.get("asr_text_path")),
            "comments": comments,
            "text_feats": load_pickle_feature(row.get("text_feature_path"), target_dim=text_dim),
            "visual_feats": load_pickle_feature(row.get("video_feature_path"), target_dim=visual_dim),
            "audio_feats": load_pickle_feature(row.get("audio_feature_path"), target_dim=audio_dim),
            "face_feats": load_pickle_feature(row.get("face_feature_path"), target_dim=face_dim),
            "skg_cache_path": row.get("skg_cache_path", ""),
        }
        item["modality_mask"] = {
            "asr": bool(item["asr_text"]),
            "comments": len(comments) > 0,
            "text_feature": item["text_feats"] is not None,
            "visual": item["visual_feats"] is not None,
            "audio": item["audio_feats"] is not None,
            "face": item["face_feats"] is not None,
            "knowledge": bool(item["skg_cache_path"]) and Path(str(item["skg_cache_path"])).exists(),
        }
        return item


def collate_fn(batch: list[dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    text_dim = int(model_cfg.get("text_dim", 768))
    visual_dim = int(model_cfg.get("visual_dim", 768))
    audio_dim = int(model_cfg.get("audio_dim", 40))
    face_dim = int(model_cfg.get("face_dim", 768))
    text_feats, text_seq_mask = pad_sequence_features(
        [b["text_feats"] for b in batch], text_dim, max_len=data_cfg.get("max_text_feature_len")
    )
    visual_feats, visual_seq_mask = pad_sequence_features(
        [b["visual_feats"] for b in batch], visual_dim, max_len=int(data_cfg.get("max_visual_len", 64))
    )
    audio_feats, audio_seq_mask = pad_sequence_features(
        [b["audio_feats"] for b in batch], audio_dim, max_len=int(data_cfg.get("max_audio_len", 8))
    )
    face_feats, face_seq_mask = pad_sequence_features(
        [b["face_feats"] for b in batch], face_dim, max_len=int(data_cfg.get("max_face_len", 64))
    )
    masks = {}
    for name in ["asr", "comments", "text_feature", "visual", "audio", "face", "knowledge"]:
        masks[name] = torch.tensor([float(b["modality_mask"][name]) for b in batch], dtype=torch.float32)
    return {
        "video_id": [b["video_id"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        "split": [b["split"] for b in batch],
        "group": [b["group"] for b in batch],
        "asr_text": [b["asr_text"] for b in batch],
        "comments": [b["comments"] for b in batch],
        "text_feats": text_feats,
        "text_seq_mask": text_seq_mask,
        "visual_feats": visual_feats,
        "visual_seq_mask": visual_seq_mask,
        "audio_feats": audio_feats,
        "audio_seq_mask": audio_seq_mask,
        "face_feats": face_feats,
        "face_seq_mask": face_seq_mask,
        "skg_cache_path": [b["skg_cache_path"] for b in batch],
        "modality_mask": masks,
    }


def make_collate_fn(config: dict[str, Any]):
    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_fn(batch, config=config)

    return _collate
