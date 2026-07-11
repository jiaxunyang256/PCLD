from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .alignment import align_evidence_to_text


class SingleEvidenceGate(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        gate_with_confidence: bool = False,
        modality_gate: bool = False,
    ) -> None:
        super().__init__()
        self.gate_with_confidence = gate_with_confidence
        self.modality_gate = modality_gate
        gate_input_dim = hidden_dim * 4 + (3 if gate_with_confidence else 0)
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        if modality_gate:
            self.modality_gate_net = nn.Sequential(
                nn.Linear(hidden_dim * 4 + 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.modality_gate_net = None

    def _confidence_features(
        self,
        text_hidden: torch.Tensor,
        evidence_hidden: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cosine = torch.nn.functional.cosine_similarity(text_hidden, evidence_hidden, dim=-1, eps=1e-6).unsqueeze(-1)
        text_norm = text_hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        norm_ratio = (evidence_hidden.norm(dim=-1, keepdim=True) / text_norm).clamp(max=10.0) / 10.0
        if mask is None:
            mask_feature = torch.ones_like(cosine)
        else:
            mask_feature = mask.float().view(-1, 1, 1).expand_as(cosine)
        return torch.cat([mask_feature, cosine, norm_ratio], dim=-1)

    def forward(
        self,
        text_hidden: torch.Tensor,
        evidence_hidden: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if evidence_hidden is None:
            return torch.zeros_like(text_hidden), None, None
        if evidence_hidden.size(1) != text_hidden.size(1):
            evidence_hidden = align_evidence_to_text(evidence_hidden, text_hidden)
        pieces = [
            [text_hidden, evidence_hidden, text_hidden - evidence_hidden, text_hidden * evidence_hidden],
        ][0]
        confidence = self._confidence_features(text_hidden, evidence_hidden, mask)
        if self.gate_with_confidence:
            pieces.append(confidence)
        gate_input = torch.cat(pieces, dim=-1)
        gate = self.gate(gate_input)
        modality_gate = None
        if self.modality_gate_net is not None:
            pooled = torch.cat(
                [
                    text_hidden.mean(dim=1),
                    evidence_hidden.mean(dim=1),
                    (text_hidden - evidence_hidden).abs().mean(dim=1),
                    (text_hidden * evidence_hidden).mean(dim=1),
                    confidence.mean(dim=1),
                ],
                dim=-1,
            )
            modality_gate = self.modality_gate_net(pooled)
            gate = gate * modality_gate.view(-1, 1, 1)
        if mask is not None:
            gate = gate * mask.float().view(-1, 1, 1)
        return gate * evidence_hidden, gate, modality_gate


class GatedResidualEvidenceAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        evidence_names: list[str] | None = None,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        learnable_scale: bool = False,
        gate_with_confidence: bool = False,
        modality_gate: bool = False,
        gate_l1_weight: float = 0.0,
        gate_entropy_weight: float = 0.0,
    ) -> None:
        super().__init__()
        evidence_names = evidence_names or ["comments", "visual", "audio", "face", "knowledge"]
        self.gate_with_confidence = gate_with_confidence
        self.modality_gate = modality_gate
        self.gate_l1_weight = gate_l1_weight
        self.gate_entropy_weight = gate_entropy_weight
        self.gates = nn.ModuleDict(
            {
                name: SingleEvidenceGate(
                    hidden_dim,
                    dropout=dropout,
                    gate_with_confidence=gate_with_confidence,
                    modality_gate=modality_gate,
                )
                for name in evidence_names
            }
        )
        if learnable_scale:
            self.residual_scales = nn.ParameterDict(
                {name: nn.Parameter(torch.tensor(float(residual_scale))) for name in evidence_names}
            )
        else:
            self.register_buffer("fixed_residual_scale", torch.tensor(float(residual_scale)))
            self.residual_scales = None
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_hidden: torch.Tensor,
        evidences: dict[str, torch.Tensor | None],
        masks: dict[str, torch.Tensor] | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]] | tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        delta_sum = torch.zeros_like(text_hidden)
        gate_stats: dict[str, float] = {}
        aux_losses: dict[str, torch.Tensor] = {}
        gate_l1_terms = []
        gate_entropy_terms = []
        masks = masks or {}
        for name, evidence in evidences.items():
            if evidence is None:
                continue
            if name not in self.gates:
                self.gates[name] = SingleEvidenceGate(
                    text_hidden.size(-1),
                    gate_with_confidence=self.gate_with_confidence,
                    modality_gate=self.modality_gate,
                ).to(text_hidden.device)
                if self.residual_scales is not None:
                    self.residual_scales[name] = nn.Parameter(torch.tensor(1.0, device=text_hidden.device))
            delta, gate, modality_gate = self.gates[name](text_hidden, evidence, masks.get(name))
            if self.residual_scales is None:
                scale = self.fixed_residual_scale.to(text_hidden.device)
            else:
                scale = self.residual_scales[name].to(text_hidden.device)
            delta_sum = delta_sum + scale * delta
            if gate is not None:
                gate_stats[name] = float(gate.detach().mean().cpu())
                gate_stats[f"{name}_scale"] = float(scale.detach().cpu())
                gate_l1_terms.append(gate.mean())
                clipped_gate = gate.clamp(1e-6, 1.0 - 1e-6)
                entropy = -(clipped_gate * clipped_gate.log() + (1.0 - clipped_gate) * (1.0 - clipped_gate).log())
                gate_entropy_terms.append(entropy.mean())
            if modality_gate is not None:
                gate_stats[f"{name}_modality_gate"] = float(modality_gate.detach().mean().cpu())
        if gate_l1_terms and self.gate_l1_weight > 0:
            aux_losses["gate_l1"] = torch.stack(gate_l1_terms).mean() * self.gate_l1_weight
        if gate_entropy_terms and self.gate_entropy_weight != 0:
            aux_losses["gate_entropy"] = torch.stack(gate_entropy_terms).mean() * self.gate_entropy_weight
        if return_aux:
            return self.norm(text_hidden + delta_sum), gate_stats, aux_losses
        return self.norm(text_hidden + delta_sum), gate_stats


class MultiEvidenceAdapter(GatedResidualEvidenceAdapter):
    pass
