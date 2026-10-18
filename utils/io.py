"""Embedding I/O and artifact paths.

Naming follows the production MVTC layout so that pooled/multivector files
are interchangeable with decision-table artifacts:

- multivector: ``embeddings/{ds}/{variant}/multivector_{dim}.pkl``
- pooled:       ``pooled/{ds}/{variant}/pooled_m{m}_tome_bipartite[_{variant_suffix}].pkl``
- labels:       ``embeddings/{ds}/{variant}/labels.npy``

where ``variant`` is ``multi`` (jina) or ``mpnet`` (sbert) and ``dim`` is the
per-token vector dimension (128 for jina, 768 for mpnet).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EMBEDDINGS = Path(os.environ.get("EMBEDDINGS_ROOT", str(DATA / "embeddings")))
POOLED = Path(os.environ.get("POOLED_ROOT", str(DATA / "pooled")))


def load_pooled(path: Path | str) -> List[np.ndarray]:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    return obj["reps"] if isinstance(obj, dict) else obj


def save_pooled(path: Path | str, reps: List[np.ndarray], meta: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"reps": reps}
    if meta:
        payload["meta"] = meta
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def labels_path(dataset: str, variant: str) -> Path:
    return EMBEDDINGS / dataset / variant / "labels.npy"


def multivector_path(dataset: str, variant: str, dim: int) -> Path:
    return EMBEDDINGS / dataset / variant / f"multivector_{dim}.pkl"


def pooled_path(dataset: str, variant: str, m: int, pooling: str = "tome_bipartite") -> Path:
    base = f"pooled_m{m}_{pooling}"
    if variant == "mpnet":
        return POOLED / dataset / variant / f"{base}__{variant}.pkl"
    return POOLED / dataset / variant / f"{base}.pkl"


def variant_for_rep(representation: str) -> str:
    return "multi" if representation in ("multi", "multi_128") else representation


def dim_for_rep(representation: str) -> int:
    return 128 if representation in ("multi", "multi_128") else 768
