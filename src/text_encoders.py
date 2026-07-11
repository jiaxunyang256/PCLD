from __future__ import annotations

import hashlib
from typing import Iterable

import torch
from torch import nn


def _hash_token(token: str, vocab_size: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % vocab_size


class HashingTextEncoder(nn.Module):
    """Small dependency-free text encoder used for smoke tests and fallbacks."""

    def __init__(self, hidden_dim: int, vocab_size: int = 8192, max_length: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.norm = nn.LayerNorm(hidden_dim)

    def encode_tokens(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.zeros((len(texts), self.max_length), dtype=torch.long, device=device)
        mask = torch.zeros((len(texts), self.max_length), dtype=torch.float32, device=device)
        for i, text in enumerate(texts):
            tokens = list(str(text).strip())[: self.max_length - 1]
            tokens = ["[CLS]"] + tokens
            for j, token in enumerate(tokens):
                ids[i, j] = _hash_token(token, self.vocab_size - 1) + 1
                mask[i, j] = 1.0
            if not tokens:
                ids[i, 0] = 1
                mask[i, 0] = 1.0
        return ids, mask

    def forward(self, texts: list[str], device: torch.device | None = None) -> dict[str, torch.Tensor]:
        device = device or self.embedding.weight.device
        ids, attention_mask = self.encode_tokens(texts, device)
        hidden = self.norm(self.embedding(ids))
        return {"hidden": hidden, "cls": hidden[:, 0], "attention_mask": attention_mask}


class SimpleCommentEncoder(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int = 8192, max_comments: int = 16, max_length: int = 96) -> None:
        super().__init__()
        self.max_comments = max_comments
        self.encoder = HashingTextEncoder(hidden_dim, vocab_size=vocab_size, max_length=max_length)
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, comments: list[list[str]], device: torch.device | None = None) -> dict[str, torch.Tensor]:
        device = device or self.score[0].weight.device
        batch_embeddings = []
        weights_out = []
        for sample_comments in comments:
            selected = [c for c in sample_comments if str(c).strip()][: self.max_comments]
            if not selected:
                batch_embeddings.append(torch.zeros(self.score[0].in_features, device=device))
                weights_out.append(torch.zeros(1, device=device))
                continue
            enc = self.encoder(selected, device=device)["cls"]
            score = self.score(enc).squeeze(-1)
            weight = torch.softmax(score, dim=0)
            batch_embeddings.append(torch.sum(enc * weight.unsqueeze(-1), dim=0))
            weights_out.append(weight.detach())
        return {"embedding": torch.stack(batch_embeddings, dim=0), "weights": weights_out}


class ASRTextEncoder(HashingTextEncoder):
    pass
