from __future__ import annotations

import hashlib
import json
import os

import torch


class SentimentKnowledgeGraph:
    """Lightweight SKG loader for mined aspect-sentiment JSON files."""

    def __init__(self, file_path: str | None = None) -> None:
        self.file_path = file_path
        self._triples: list[tuple[str, str, float]] | None = None
        self._vector_cache: dict[tuple[str, int, int], torch.Tensor | None] = {}

    @property
    def triples(self) -> list[tuple[str, str, float]]:
        if self._triples is None:
            self._triples = self._load()
        return self._triples

    def _load(self) -> list[tuple[str, str, float]]:
        triples: list[tuple[str, str, float]] = []
        if not self.file_path or not os.path.exists(self.file_path):
            return triples
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for aspect, sentiment_items in data.items():
            if not isinstance(sentiment_items, list):
                continue
            for item in sentiment_items:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        polarity = float(item[1])
                    except Exception:
                        polarity = 0.0
                    triples.append((str(aspect), str(item[0]), polarity))
        return triples

    @staticmethod
    def hash_index(text: str, dim: int) -> int:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return int(digest, 16) % dim

    def match_to_vector(self, text: str, dim: int, max_matches: int = 64) -> torch.Tensor | None:
        cache_key = (text, dim, max_matches)
        if cache_key in self._vector_cache:
            return self._vector_cache[cache_key]
        if not text or not self.triples:
            return None
        vec = torch.zeros(dim, dtype=torch.float32)
        matches = 0
        for aspect, sentiment, polarity in self.triples:
            if aspect and sentiment and (aspect in text or sentiment in text):
                sign = 1.0 if polarity >= 2 else -1.0
                vec[self.hash_index(aspect, dim)] += sign
                vec[self.hash_index(sentiment, dim)] += sign
                matches += 1
                if matches >= max_matches:
                    break
        if matches == 0:
            self._vector_cache[cache_key] = None
            return None
        result = vec / matches
        self._vector_cache[cache_key] = result
        return result
