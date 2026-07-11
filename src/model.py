from __future__ import annotations

import torch
from torch import nn

from .alignment import align_evidence_to_text, mmd_loss
from .evidence_adapter import GatedResidualEvidenceAdapter
from .evidence_sequence import build_evidence_sequence_encoder
from .knowledge_adapter import KnowledgeAdapter
from .text_encoders import ASRTextEncoder, SimpleCommentEncoder


class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        return self.net(x.float())


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.score(hidden).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask <= 0, -1e4)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(hidden * weights.unsqueeze(-1), dim=1)


class TextCentricCPCLModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.model_cfg = config.get("model", config)
        self.align_cfg = config.get("alignment", {})
        self.hidden_dim = int(self.model_cfg.get("hidden_dim", 128))
        dropout = float(self.model_cfg.get("dropout", 0.1))
        self.use_visual = bool(self.model_cfg.get("use_visual", False))
        self.use_audio = bool(self.model_cfg.get("use_audio", False))
        self.use_face = bool(self.model_cfg.get("use_face", False))
        self.use_asr = bool(self.model_cfg.get("use_asr", True))
        self.use_text_features = bool(self.model_cfg.get("use_text_features", True))
        self.use_comments = bool(self.model_cfg.get("use_comments", True))
        self.use_knowledge = bool(self.model_cfg.get("use_knowledge", False))
        self.use_mamba = bool(self.model_cfg.get("use_mamba", False))
        self.fusion_strategy = str(self.model_cfg.get("fusion_strategy", "gated")).lower()
        self.evidence_dropout = float(self.model_cfg.get("evidence_dropout", 0.0))
        self.evidence_dropout_modalities = set(
            self.model_cfg.get("evidence_dropout_modalities", ["visual", "audio", "face", "knowledge"])
        )

        self.text_projector = FeatureProjector(int(self.model_cfg.get("text_dim", 768)), self.hidden_dim, dropout)
        self.asr_encoder = ASRTextEncoder(self.hidden_dim)
        self.comment_encoder = SimpleCommentEncoder(
            self.hidden_dim,
            vocab_size=int(self.model_cfg.get("comment_vocab_size", 8192)),
            max_comments=int(self.model_cfg.get("max_comments", 16)),
            max_length=int(self.model_cfg.get("comment_max_length", 96)),
        )
        self.visual_projector = FeatureProjector(int(self.model_cfg.get("visual_dim", 768)), self.hidden_dim, dropout)
        self.audio_projector = FeatureProjector(int(self.model_cfg.get("audio_dim", 40)), self.hidden_dim, dropout)
        self.face_projector = FeatureProjector(int(self.model_cfg.get("face_dim", 768)), self.hidden_dim, dropout)
        self.evidence_sequence_encoder = build_evidence_sequence_encoder(config, self.hidden_dim, dropout)
        knowledge_cfg = config.get("knowledge", {})
        self.knowledge_adapter = KnowledgeAdapter(
            self.hidden_dim,
            use_knowledge=self.use_knowledge,
            input_dim=int(knowledge_cfg.get("input_dim", self.model_cfg.get("knowledge_dim", 768))),
            global_skg_path=knowledge_cfg.get("global_skg_path"),
            max_matches=int(knowledge_cfg.get("max_matches", 64)),
            mode=knowledge_cfg.get("mode", "lightweight"),
            bert_model_name=knowledge_cfg.get("bert_model_name", "bert-base-chinese"),
            sentence_model_name=knowledge_cfg.get("sentence_model_name", "paraphrase-multilingual-MiniLM-L12-v2"),
            similarity_threshold=float(knowledge_cfg.get("similarity_threshold", 0.6)),
            max_length=int(knowledge_cfg.get("max_length", 256)),
        )
        gate_cfg = config.get("gate", {})
        self.adapter = GatedResidualEvidenceAdapter(
            self.hidden_dim,
            dropout=dropout,
            residual_scale=float(gate_cfg.get("residual_scale", 1.0)),
            learnable_scale=bool(gate_cfg.get("learnable_scale", False)),
            gate_with_confidence=bool(gate_cfg.get("gate_with_confidence", False)),
            modality_gate=bool(gate_cfg.get("modality_gate", False)),
            gate_l1_weight=float(gate_cfg.get("gate_l1_weight", 0.0)),
            gate_entropy_weight=float(gate_cfg.get("gate_entropy_weight", 0.0)),
        )
        self.attn_pool = AttentionPool(self.hidden_dim)
        self.concat_pool = AttentionPool(self.hidden_dim)
        self.concat_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 7, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.interaction_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 17, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def _text_hidden(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        text_feats = batch.get("text_feats")
        text_mask = batch.get("text_seq_mask")
        has_text_feature = batch.get("modality_mask", {}).get("text_feature")
        if (
            self.use_text_features
            and text_feats is not None
            and has_text_feature is not None
            and has_text_feature.float().sum().item() == float(text_feats.size(0))
        ):
            hidden = self.text_projector(text_feats.to(device))
            mask = text_mask.to(device) if text_mask is not None else torch.ones(hidden.shape[:2], device=device)
            return hidden, mask
        if not self.use_asr:
            batch_size = len(batch["label"])
            hidden = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
            mask = torch.ones(batch_size, 1, device=device)
            return hidden, mask
        encoded = self.asr_encoder(batch.get("asr_text", [""] * len(batch["label"])), device=device)
        return encoded["hidden"], encoded["attention_mask"]

    def _expand_vector(self, vector: torch.Tensor | None, length: int) -> torch.Tensor | None:
        if vector is None:
            return None
        return vector.unsqueeze(1).expand(-1, length, -1)

    def _sequence_mask(self, batch: dict, name: str, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        mask = batch.get(f"{name}_seq_mask")
        if mask is None:
            return torch.ones(tensor.shape[:2], device=device)
        return mask.to(device)

    def _masked_pool(self, hidden: torch.Tensor | None, mask: torch.Tensor | None = None) -> torch.Tensor:
        device = next(self.parameters()).device
        if hidden is None:
            return torch.zeros(1, self.hidden_dim, device=device)
        if mask is None:
            return hidden.mean(dim=1)
        mask = mask.to(hidden.device).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _zero_vec(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def _apply_evidence_dropout(
        self,
        evidences: dict[str, torch.Tensor | None],
        evidence_masks: dict[str, torch.Tensor],
    ) -> None:
        if not self.training or self.evidence_dropout <= 0:
            return
        for name in list(evidences.keys()):
            if name not in self.evidence_dropout_modalities:
                continue
            mask = evidence_masks.get(name)
            if mask is None:
                continue
            keep = torch.bernoulli(torch.full_like(mask.float(), 1.0 - self.evidence_dropout))
            evidence_masks[name] = mask.float() * keep

    def _evidence_vector(
        self,
        name: str,
        evidences: dict[str, torch.Tensor | None],
        evidence_masks: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        pool_mask: torch.Tensor | None = None,
        fallback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if fallback is not None:
            vec = fallback
        elif name in evidences:
            vec = self._masked_pool(evidences.get(name), pool_mask)
        else:
            vec = self._zero_vec(batch_size, device)
        modality_mask = evidence_masks.get(name)
        if modality_mask is not None:
            vec = vec * modality_mask.to(device).float().view(-1, 1)
        return vec

    def _interaction_features(self, text_vec: torch.Tensor, evidence_vec: torch.Tensor) -> list[torch.Tensor]:
        return [evidence_vec, text_vec * evidence_vec, torch.abs(text_vec - evidence_vec)]

    def forward(self, batch: dict) -> dict:
        device = next(self.parameters()).device
        labels = batch.get("label")
        text_hidden, text_mask = self._text_hidden(batch)
        length = text_hidden.size(1)
        masks = batch.get("modality_mask", {})
        comments_vec = None
        if self.use_comments:
            comments_vec = self.comment_encoder(batch.get("comments", [[] for _ in range(text_hidden.size(0))]), device=device)["embedding"]

        evidences = {}
        evidence_masks = {}
        if comments_vec is not None:
            evidences["comments"] = self._expand_vector(comments_vec, length)
            evidence_masks["comments"] = masks.get("comments", torch.ones(text_hidden.size(0))).to(device)
        if self.use_visual:
            visual = self.visual_projector(batch["visual_feats"].to(device))
            visual = self.evidence_sequence_encoder(visual, self._sequence_mask(batch, "visual", visual, device))
            visual = align_evidence_to_text(visual, text_hidden, self.align_cfg.get("method", "interpolate"))
            evidences["visual"] = visual
            evidence_masks["visual"] = masks.get("visual", torch.ones(text_hidden.size(0))).to(device)
        if self.use_audio:
            audio = self.audio_projector(batch["audio_feats"].to(device))
            audio = self.evidence_sequence_encoder(audio, self._sequence_mask(batch, "audio", audio, device))
            audio = align_evidence_to_text(audio, text_hidden, self.align_cfg.get("method", "interpolate"))
            evidences["audio"] = audio
            evidence_masks["audio"] = masks.get("audio", torch.ones(text_hidden.size(0))).to(device)
        if self.use_face:
            face = self.face_projector(batch["face_feats"].to(device))
            face = self.evidence_sequence_encoder(face, self._sequence_mask(batch, "face", face, device))
            face = align_evidence_to_text(face, text_hidden, self.align_cfg.get("method", "interpolate"))
            evidences["face"] = face
            evidence_masks["face"] = masks.get("face", torch.ones(text_hidden.size(0))).to(device)
        knowledge_vec = self.knowledge_adapter(
            batch.get("skg_cache_path", []),
            device=device,
            texts=batch.get("asr_text"),
            comments=batch.get("comments"),
        )
        if knowledge_vec is not None:
            evidences["knowledge"] = self._expand_vector(knowledge_vec, length)
            knowledge_mask = masks.get("knowledge")
            if knowledge_mask is None or knowledge_mask.float().sum() == 0:
                knowledge_mask = torch.ones(text_hidden.size(0))
            evidence_masks["knowledge"] = knowledge_mask.to(device)
        self._apply_evidence_dropout(evidences, evidence_masks)

        if self.fusion_strategy == "concat":
            batch_size = text_hidden.size(0)
            z_text_cls = text_hidden[:, 0]
            z_text_attn = self.concat_pool(text_hidden, text_mask)
            vectors = [
                z_text_cls,
                z_text_attn,
                self._evidence_vector("comments", evidences, evidence_masks, batch_size, device, fallback=comments_vec),
                self._evidence_vector("visual", evidences, evidence_masks, batch_size, device, pool_mask=text_mask),
                self._evidence_vector("audio", evidences, evidence_masks, batch_size, device, pool_mask=text_mask),
                self._evidence_vector("face", evidences, evidence_masks, batch_size, device, pool_mask=text_mask),
                self._evidence_vector("knowledge", evidences, evidence_masks, batch_size, device, fallback=knowledge_vec),
            ]
            logits = self.concat_classifier(torch.cat(vectors, dim=-1)).squeeze(-1)
            losses = {}
            if self.align_cfg.get("use_mmd", False):
                total_mmd = torch.zeros((), device=device)
                for name in ("visual", "audio", "face"):
                    if name in evidences:
                        total_mmd = total_mmd + mmd_loss(evidences[name], text_hidden, evidence_masks.get(name))
                losses["mmd"] = total_mmd
            return {"logits": logits, "prob": torch.sigmoid(logits), "losses": losses, "gate_stats": {}}

        if self.fusion_strategy in {"interaction", "bilinear", "late_interaction"}:
            batch_size = text_hidden.size(0)
            z_text_cls = text_hidden[:, 0]
            z_text_attn = self.concat_pool(text_hidden, text_mask)
            evidence_vectors = [
                self._evidence_vector("comments", evidences, evidence_masks, batch_size, device, fallback=comments_vec),
                self._evidence_vector("visual", evidences, evidence_masks, batch_size, device, pool_mask=text_mask),
                self._evidence_vector("audio", evidences, evidence_masks, batch_size, device, pool_mask=text_mask),
                self._evidence_vector("face", evidences, evidence_masks, batch_size, device, pool_mask=text_mask),
                self._evidence_vector("knowledge", evidences, evidence_masks, batch_size, device, fallback=knowledge_vec),
            ]
            vectors = [z_text_cls, z_text_attn]
            for evidence_vec in evidence_vectors:
                vectors.extend(self._interaction_features(z_text_attn, evidence_vec))
            logits = self.interaction_classifier(torch.cat(vectors, dim=-1)).squeeze(-1)
            losses = {}
            if self.align_cfg.get("use_mmd", False):
                total_mmd = torch.zeros((), device=device)
                for name in ("visual", "audio", "face"):
                    if name in evidences:
                        total_mmd = total_mmd + mmd_loss(evidences[name], text_hidden, evidence_masks.get(name))
                losses["mmd"] = total_mmd
            return {"logits": logits, "prob": torch.sigmoid(logits), "losses": losses, "gate_stats": {}}

        if self.fusion_strategy in {"nogate", "no_gate", "residual"}:
            delta_sum = torch.zeros_like(text_hidden)
            for name, evidence in evidences.items():
                if evidence is None:
                    continue
                if evidence.size(1) != text_hidden.size(1):
                    evidence = align_evidence_to_text(evidence, text_hidden)
                mask = evidence_masks.get(name)
                if mask is not None:
                    evidence = evidence * mask.to(device).float().view(-1, 1, 1)
                delta_sum = delta_sum + evidence
            enhanced = self.adapter.norm(text_hidden + delta_sum)
            gate_stats = {}
        else:
            enhanced, gate_stats, gate_losses = self.adapter(text_hidden, evidences, evidence_masks, return_aux=True)
        z_cls = enhanced[:, 0]
        z_attn = self.attn_pool(enhanced, text_mask)
        if comments_vec is None:
            comments_vec = torch.zeros_like(z_cls)
        logits = self.classifier(torch.cat([z_cls, z_attn, comments_vec], dim=-1)).squeeze(-1)
        losses = {}
        if self.fusion_strategy not in {"nogate", "no_gate", "residual"}:
            losses.update(gate_losses)
        if self.align_cfg.get("use_mmd", False):
            total_mmd = torch.zeros((), device=device)
            for name in ("visual", "audio", "face"):
                if name in evidences:
                    total_mmd = total_mmd + mmd_loss(evidences[name], text_hidden, evidence_masks.get(name))
            losses["mmd"] = total_mmd
        return {
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "losses": losses,
            "gate_stats": gate_stats,
        }
