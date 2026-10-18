"""Shared multi-vector helpers."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def resolve_torch_device(device: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def pack_multivecs(
    reps: Sequence[np.ndarray],
    *,
    dtype: np.dtype | type = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    dim = int(np.asarray(reps[0]).shape[1])
    lengths = np.asarray([max(1, int(np.asarray(r).shape[0])) for r in reps], dtype=np.int64)
    p_max = int(lengths.max())
    packed = np.zeros((len(reps), p_max, dim), dtype=dtype)
    for i, r in enumerate(reps):
        rr = np.asarray(r, dtype=dtype)
        if rr.size:
            packed[i, : rr.shape[0]] = rr
    return packed, lengths


def pad_truncate_multivec(x: np.ndarray, k_mu: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    kx, dim = x.shape
    if kx >= k_mu:
        if kx == k_mu:
            return x.copy()
        pick = rng.choice(kx, size=k_mu, replace=False)
        return x[np.sort(pick)].copy()
    out = np.zeros((k_mu, dim), dtype=np.float64)
    out[:kx] = x
    out[kx:] = x.mean(axis=0)
    return out


def prepare_reps(
    reps: Sequence[np.ndarray],
    *,
    spherical: bool,
    dtype: np.dtype | type,
) -> List[np.ndarray]:
    dim = int(np.asarray(reps[0]).shape[1])
    out: List[np.ndarray] = []
    for rep in reps:
        rr = np.asarray(rep, dtype=dtype)
        if rr.size == 0:
            rr = np.zeros((1, dim), dtype=dtype)
        out.append(normalize_rows(rr).astype(dtype, copy=False) if spherical else rr)
    return out
