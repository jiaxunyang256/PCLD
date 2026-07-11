from __future__ import annotations

import json
import os
import pickle
from os import PathLike
from typing import Any

import torch
from torch import nn

from .feature_utils import ensure_2d_feature
from .skg import SentimentKnowledgeGraph


class KnowledgeAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        use_knowledge: bool = False,
        input_dim: int = 768,
        global_skg_path: str | None = None,
        max_matches: int = 64,
        mode: str = "lightweight",
        bert_model_name: str = "bert-base-chinese",
        sentence_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.6,
        max_length: int = 256,
    ) -> None:
        super().__init__()
        self.use_knowledge = use_knowledge
        self.mode = mode
        self.global_skg_path = global_skg_path
        self.max_matches = max_matches
        self.skg = SentimentKnowledgeGraph(global_skg_path)
        self.full_encoder = None
        full_dim = input_dim
        if self.use_knowledge and self.mode == "full":
            if not global_skg_path:
                raise ValueError("knowledge.mode=full requires knowledge.global_skg_path")
            from .skg_full import FullSKGEncoder

            self.full_encoder = FullSKGEncoder(
                skg_file=global_skg_path,
                bert_model_name=bert_model_name,
                sentence_model_name=sentence_model_name,
                similarity_threshold=similarity_threshold,
                max_matches=max_matches,
                max_length=max_length,
            )
            full_dim = self.full_encoder.hidden_size
        self.proj = nn.Linear(full_dim, hidden_dim)

    def match_global_skg(self, text: str) -> torch.Tensor | None:
        return self.skg.match_to_vector(text, dim=self.proj.in_features, max_matches=self.max_matches)

    def load_knowledge_feature(self, path: str | None) -> torch.Tensor | None:
        if not self.use_knowledge or not isinstance(path, (str, bytes, PathLike)) or not path or not os.path.exists(path):
            return None
        try:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    obj: Any = json.load(f)
            else:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
        except Exception:
            return None
        arr = ensure_2d_feature(obj, default_dim=self.proj.in_features)
        if arr is None:
            return None
        return torch.tensor(arr.mean(axis=0), dtype=torch.float32)

    def forward(
        self,
        paths: list[str],
        device: torch.device,
        texts: list[str] | None = None,
        comments: list[list[str]] | None = None,
    ) -> torch.Tensor | None:
        if not self.use_knowledge:
            return None
        if self.mode == "full":
            if self.full_encoder is None:
                return None
            texts = texts or [""] * len(paths)
            comments = comments or [[] for _ in paths]
            merged = [str(texts[i]) + " " + " ".join(str(x) for x in comments[i]) for i in range(len(paths))]
            x = self.full_encoder(merged, device=device)
            return self.proj(x)
        vectors = []
        any_valid = False
        texts = texts or [""] * len(paths)
        comments = comments or [[] for _ in paths]
        for idx, path in enumerate(paths):
            vec = self.load_knowledge_feature(path)
            if vec is None:
                merged_text = str(texts[idx]) + " " + " ".join(str(x) for x in comments[idx])
                vec = self.match_global_skg(merged_text)
            if vec is None:
                vectors.append(torch.zeros(self.proj.in_features))
            else:
                any_valid = True
                vectors.append(vec)
        if not any_valid:
            return None
        x = torch.stack(vectors).to(device)
        return self.proj(x)
