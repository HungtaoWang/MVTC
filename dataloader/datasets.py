"""Canonical text loaders for paper-8 datasets.

Each dataset's authoritative encode input lives at ``data/cache/{dataset}/records.jsonl``
(one JSON per line: ``{"id", "text", "label"}``). Download the canonical archive
(see README) and place it under ``data/cache/`` before running ``prepare_embeddings.py``.

This loader reads the provided canonical ``records.jsonl`` files.
It does not download datasets or preprocess raw input files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"

DATASETS: tuple[str, ...] = (
    "trec",
    "reuters52",
    "20news",
    "mag",
    "mesh",
    "locov1",
    "dbpedia",
    "yahoo",
)


def _records_path(dataset: str) -> Path:
    return CACHE / dataset / "records.jsonl"


def load_dataset(name: str) -> Tuple[List[str], np.ndarray, dict]:
    """Load canonical texts + integer labels for ``name``.

    Returns ``(texts, y, meta)`` where ``texts`` is the exact list of strings
    passed to the encoder, ``y`` is an int64 array of cluster ids, and ``meta``
    is the dataset's ``meta.json`` (or a minimal stub if absent).
    """
    key = name.strip().lower()
    if key not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    p = _records_path(key)
    if not p.is_file():
        raise FileNotFoundError(
            f"canonical records not found: {p}\n"
            f"download the canonical archive (see README) and extract under data/cache/{key}/"
        )
    texts: List[str] = []
    labels: List[int] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(str(obj["text"]))
            labels.append(int(obj["label"]))
    y = np.asarray(labels, dtype=np.int64)
    meta_path = CACHE / key / "meta.json"
    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("dataset", key)
    meta.setdefault("n", len(texts))
    meta.setdefault("k", len(np.unique(y)))
    return texts, y, meta
