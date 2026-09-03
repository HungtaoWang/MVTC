"""Jina and SBERT encoding helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np


def encode_jina(texts: List[str], *, model: str = "jinaai/jina-embeddings-v4", dim: int = 128, device: str = "cuda", batch_size: int = 64) -> List[np.ndarray]:
    import torch
    from transformers import AutoModel

    net = AutoModel.from_pretrained(model, trust_remote_code=True).to(device).eval()
    reps: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            out = net.encode_text(batch, task="text-matching", truncate_dim=dim if dim < 2048 else None)
        if isinstance(out, torch.Tensor):
            arr = out.detach().cpu().numpy()
            for j in range(arr.shape[0]):
                row = arr[j : j + 1] if arr.ndim == 3 else arr[j][None]
                reps.append(row / np.maximum(np.linalg.norm(row, axis=1, keepdims=True), 1e-12))
        else:
            for item in out:
                if hasattr(item, "detach"):
                    row = item.detach().cpu().numpy()
                else:
                    row = np.asarray(item, dtype=np.float32)
                if row.ndim == 1:
                    row = row[None, :]
                reps.append(row / np.maximum(np.linalg.norm(row, axis=1, keepdims=True), 1e-12))
    return reps


def encode_sbert(texts: List[str], *, model: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cuda", batch_size: int = 32) -> List[np.ndarray]:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model, device=device)
    reps: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        emb = st.encode(texts[i : i + batch_size], convert_to_numpy=True, normalize_embeddings=True)
        for row in emb:
            reps.append(np.asarray(row, dtype=np.float32)[None, :])
    return reps


def save_multivector(path: Path | str, reps: List[np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"reps": reps}, f, protocol=pickle.HIGHEST_PROTOCOL)
