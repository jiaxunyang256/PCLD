from __future__ import annotations

from functools import lru_cache
from typing import Any

import jieba
import torch
from torch import nn

from .skg import SentimentKnowledgeGraph


class FullSKGEncoder(nn.Module):
    """BERT/SentenceTransformer based SKG encoder adapted from legacy skg.py.

    This module is intentionally optional and imported only when
    knowledge.mode == "full". It returns a knowledge-enhanced CLS vector rather
    than acting as an independent probability ensemble.
    """

    def __init__(
        self,
        skg_file: str,
        bert_model_name: str = "bert-base-chinese",
        sentence_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.6,
        min_triples: int = 1,
        max_matches: int = 16,
        max_length: int = 256,
    ) -> None:
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer, util
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "knowledge.mode=full requires transformers and sentence-transformers. "
                "Install sentence-transformers or use knowledge.mode=lightweight."
            ) from exc

        self.util = util
        self.similarity_threshold = similarity_threshold
        self.max_matches = max_matches
        self.max_length = max_length
        self.skg = SentimentKnowledgeGraph(skg_file)
        triples = self.skg.triples
        if min_triples > 1:
            aspect_counts: dict[str, int] = {}
            for aspect, _, _ in triples:
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
            triples = [t for t in triples if aspect_counts.get(t[0], 0) >= min_triples]
        self.triples = triples
        self.aspects = sorted({a for a, _, _ in triples})
        self.sentiments = sorted({s for _, s, _ in triples})
        self.sentiment_map: dict[str, list[tuple[str, float]]] = {}
        for aspect, sentiment, polarity in triples:
            self.sentiment_map.setdefault(sentiment, []).append((aspect, polarity))

        if not self.aspects or not self.sentiments:
            raise ValueError(f"No usable SKG triples found in {skg_file}")

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.sbert = SentenceTransformer(sentence_model_name)
        self._aspect_embeddings: torch.Tensor | None = None
        self._sentiment_embeddings: torch.Tensor | None = None

    @property
    def hidden_size(self) -> int:
        return int(self.bert.config.hidden_size)

    def _ensure_embeddings(self, device: torch.device) -> None:
        if self._aspect_embeddings is None:
            self._aspect_embeddings = self.sbert.encode(self.aspects, convert_to_tensor=True, show_progress_bar=False)
        if self._sentiment_embeddings is None:
            self._sentiment_embeddings = self.sbert.encode(self.sentiments, convert_to_tensor=True, show_progress_bar=False)
        self._aspect_embeddings = self._aspect_embeddings.to(device)
        self._sentiment_embeddings = self._sentiment_embeddings.to(device)

    @lru_cache(maxsize=4096)
    def _segment_cached(self, text: str) -> tuple[str, ...]:
        return tuple(tok for tok in jieba.cut(text) if str(tok).strip())

    def find_similar_triples(self, text: str, device: torch.device) -> list[tuple[str, str, float]]:
        tokens = list(self._segment_cached(text))
        if not tokens:
            return []
        self._ensure_embeddings(device)
        sentence_embeddings = self.sbert.encode(tokens, convert_to_tensor=True, show_progress_bar=False).to(device)
        aspect_sim = self.util.cos_sim(sentence_embeddings, self._aspect_embeddings)
        sentiment_sim = self.util.cos_sim(sentence_embeddings, self._sentiment_embeddings)
        aspect_matches = torch.where(aspect_sim > self.similarity_threshold)
        sentiment_matches = torch.where(sentiment_sim > self.similarity_threshold)
        sentiment_by_token: dict[int, list[str]] = {}
        for token_idx, sent_idx in zip(sentiment_matches[0].tolist(), sentiment_matches[1].tolist()):
            sentiment_by_token.setdefault(int(token_idx), []).append(self.sentiments[int(sent_idx)])
        relevant: set[tuple[str, str, float]] = set()
        for token_idx, aspect_idx in zip(aspect_matches[0].tolist(), aspect_matches[1].tolist()):
            aspect = self.aspects[int(aspect_idx)]
            for sentiment in sentiment_by_token.get(int(token_idx), []):
                for mapped_aspect, polarity in self.sentiment_map.get(sentiment, []):
                    if mapped_aspect == aspect:
                        relevant.add((aspect, sentiment, float(polarity)))
                        if len(relevant) >= self.max_matches:
                            return list(relevant)
        return list(relevant)

    def build_augmented_inputs(self, text: str, triples: list[tuple[str, str, float]], device: torch.device) -> dict[str, torch.Tensor]:
        base = self.tokenizer(
            text,
            truncation=True,
            max_length=max(8, self.max_length - 3 * len(triples)),
            add_special_tokens=True,
        )
        input_ids = list(base["input_ids"])
        for aspect, sentiment, polarity in triples:
            knowledge_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(f"{aspect} {sentiment} {int(polarity)}"))
            if not knowledge_tokens:
                continue
            if len(input_ids) + len(knowledge_tokens) > self.max_length:
                break
            input_ids.extend(knowledge_tokens)
        input_ids = input_ids[: self.max_length]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=device),
        }

    def encode_one(self, text: str, device: torch.device) -> torch.Tensor:
        triples = self.find_similar_triples(text, device)
        inputs = self.build_augmented_inputs(text, triples, device)
        outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0)

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        self.bert.to(device)
        self.sbert.to(str(device))
        vectors = [self.encode_one(text, device) for text in texts]
        return torch.stack(vectors, dim=0)
