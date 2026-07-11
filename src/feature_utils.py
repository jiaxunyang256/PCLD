from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def to_numpy_feature(obj, default_dim: int | None = None) -> np.ndarray | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        if obj.get("all_zero") is True and "features" not in obj:
            return np.zeros((1, default_dim), dtype=np.float32) if default_dim else None
        if "features" in obj:
            return to_numpy_feature(obj["features"], default_dim=default_dim)
        return None
    if isinstance(obj, np.ndarray):
        arr = obj
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return None
        arr = np.asarray(obj)
    else:
        try:
            arr = np.asarray(obj)
        except Exception:
            return None
    if arr.size == 0:
        return None
    return arr.astype(np.float32, copy=False)


def ensure_2d_feature(feature, default_dim: int | None = None) -> np.ndarray | None:
    arr = to_numpy_feature(feature, default_dim=default_dim)
    if arr is None:
        return None
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if default_dim is not None:
        arr = pad_or_truncate_dim(arr, default_dim)
    return arr.astype(np.float32, copy=False)


def pad_or_truncate_dim(feature: np.ndarray | None, target_dim: int) -> np.ndarray | None:
    if feature is None:
        return None
    arr = np.asarray(feature, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    dim = arr.shape[-1]
    if dim == target_dim:
        return arr
    if dim > target_dim:
        return arr[..., :target_dim]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, target_dim - dim)
    return np.pad(arr, pad_width, mode="constant").astype(np.float32)


def pad_sequence_features(
    features: Iterable[np.ndarray | None],
    target_dim: int,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    arrays = [ensure_2d_feature(f, default_dim=target_dim) for f in features]
    lengths = [0 if a is None else int(a.shape[0]) for a in arrays]
    seq_len = max(lengths) if lengths else 1
    if max_len is not None:
        seq_len = min(seq_len, max_len)
    seq_len = max(seq_len, 1)
    batch = np.zeros((len(arrays), seq_len, target_dim), dtype=np.float32)
    mask = np.zeros((len(arrays), seq_len), dtype=np.float32)
    for i, arr in enumerate(arrays):
        if arr is None:
            continue
        n = min(arr.shape[0], seq_len)
        batch[i, :n] = arr[:n]
        mask[i, :n] = 1.0
    return torch.from_numpy(batch), torch.from_numpy(mask)


def build_modality_mask(values: Iterable[object]) -> torch.Tensor:
    flags = []
    for value in values:
        if value is None:
            flags.append(False)
        elif isinstance(value, (list, tuple, str)):
            flags.append(len(value) > 0)
        elif isinstance(value, np.ndarray):
            flags.append(value.size > 0)
        else:
            flags.append(bool(value))
    return torch.tensor(flags, dtype=torch.float32)
