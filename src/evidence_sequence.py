from __future__ import annotations

import torch
from torch import nn


class IdentityEvidenceSequenceEncoder(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return x
        return x * mask.to(x.device).unsqueeze(-1)


class TransformerEvidenceSequenceEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            num_heads = 1
        self.position = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        length = x.size(1)
        if length > self.position.size(1):
            raise ValueError(f"Evidence sequence length {length} exceeds max_len {self.position.size(1)}")
        out = x + self.position[:, :length]
        out = self.encoder(out)
        out = self.norm(out + x)
        if mask is not None:
            out = out * mask.to(out.device).unsqueeze(-1)
        return out


class MambaEvidenceSequenceEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "evidence_encoder.type=mamba requires mamba_ssm. "
                "Install mamba-ssm, or set evidence_encoder.type=transformer/none."
            ) from exc
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden_dim),
                        "mamba": Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand),
                        "dropout": nn.Dropout(dropout),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out = x
        if mask is not None:
            out = out * mask.to(out.device).unsqueeze(-1)
        for layer in self.layers:
            residual = out
            out = residual + layer["dropout"](layer["mamba"](layer["norm"](out)))
            if mask is not None:
                out = out * mask.to(out.device).unsqueeze(-1)
        return self.final_norm(out)


def build_evidence_sequence_encoder(config: dict, hidden_dim: int, dropout: float) -> nn.Module:
    cfg = config.get("evidence_encoder", {})
    model_cfg = config.get("model", {})
    encoder_type = str(cfg.get("type") or ("mamba" if model_cfg.get("use_mamba", False) else "none")).lower()
    if encoder_type in {"none", "identity", "mlp"}:
        return IdentityEvidenceSequenceEncoder()
    if encoder_type == "transformer":
        return TransformerEvidenceSequenceEncoder(
            hidden_dim=hidden_dim,
            num_layers=int(cfg.get("num_layers", 1)),
            num_heads=int(cfg.get("num_heads", 4)),
            dropout=float(cfg.get("dropout", dropout)),
            max_len=int(cfg.get("max_len", 256)),
        )
    if encoder_type == "mamba":
        return MambaEvidenceSequenceEncoder(
            hidden_dim=hidden_dim,
            num_layers=int(cfg.get("num_layers", 1)),
            d_state=int(cfg.get("d_state", 16)),
            d_conv=int(cfg.get("d_conv", 4)),
            expand=int(cfg.get("expand", 2)),
            dropout=float(cfg.get("dropout", dropout)),
        )
    raise ValueError(f"Unknown evidence_encoder.type: {encoder_type}")
