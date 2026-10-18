"""Jina v4 and MPNet token-multivector encoders.

Ported from the production MVTC pipeline
(``bundles/cpu_timing_trec_reuters52/repo/src/encoding/jina_hf.py``) so that
``prepare_embeddings.py`` reproduces decision-table multivector embeddings.

Key points (must match production to reproduce):
- Jina: ``encode_text(..., return_multivector=True, task="text-matching")``,
  no extra L2, per-token 128-d vectors.
- MPNet: ``AutoModel`` ``last_hidden_state`` masked by ``attention_mask``,
  per-token L2 via ``torch.nn.functional.normalize``, keep CLS/SEP,
  ``max_length=384``, ``batch_size=16``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

JINA_MODEL = "jinaai/jina-embeddings-v4"
JINA_TASK = "text-matching"
MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
MPNET_MAX_LENGTH = 384
MPNET_BATCH = 16


def _get_last_hidden_state(outputs) -> "torch.Tensor":  # type: ignore[name-defined]
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if hasattr(outputs, "vlm_last_hidden_states") and outputs.vlm_last_hidden_states is not None:
        return outputs.vlm_last_hidden_states
    if isinstance(outputs, tuple) and len(outputs) > 0:
        return outputs[0]
    raise ValueError(f"unable to extract hidden state from {type(outputs)}")


def encode_jina(
    texts: List[str],
    *,
    max_length: int = 512,
    batch_size: int = 64,
    device: str = "cuda",
    model: str = JINA_MODEL,
) -> List[np.ndarray]:
    """Jina v4 native multivector (per-token 128-d, no extra L2)."""
    import torch
    from transformers import AutoModel

    net = AutoModel.from_pretrained(model, trust_remote_code=True).to(device).eval()
    results: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            multi = net.encode_text(
                texts=batch,
                task=JINA_TASK,
                return_multivector=True,
                max_length=int(max_length),
                batch_size=int(batch_size),
            )
            if isinstance(multi, torch.Tensor):
                arr = multi.detach().cpu().to(torch.float32).numpy()
                for b in range(arr.shape[0]):
                    results.append(np.asarray(arr[b], dtype=np.float32))
            elif isinstance(multi, list):
                for emb in multi:
                    if isinstance(emb, torch.Tensor):
                        emb = emb.detach().cpu().to(torch.float32).numpy()
                    results.append(np.asarray(emb, dtype=np.float32))
            else:
                arr = np.asarray(multi, dtype=np.float32)
                if arr.ndim == 3:
                    for b in range(arr.shape[0]):
                        results.append(arr[b].astype(np.float32))
                else:
                    raise ValueError(f"unexpected jina output: {arr.shape}")
    return results


def encode_sbert(
    texts: List[str],
    *,
    batch_size: int = MPNET_BATCH,
    max_length: int = MPNET_MAX_LENGTH,
    device: str = "cuda",
    model: str = MPNET_MODEL,
) -> List[np.ndarray]:
    """MPNet token multivector: last_hidden_state masked + per-token L2.

    NOTE: this is *not* SentenceTransformer sentence embedding. It extracts
    per-token hidden states from the underlying HF model, matching the
    production encoder. Keep CLS/SEP (only padding is removed via
    attention_mask). Output shape per doc: (L_tokens, 768).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModel.from_pretrained(model).to(device).eval()
    results: List[np.ndarray] = []
    bs = max(1, int(batch_size))
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = net(**enc)
            hidden = _get_last_hidden_state(outputs).detach().float()
            valid = enc["attention_mask"].bool()
            for emb, mask in zip(hidden, valid):
                row = emb[mask]
                row = torch.nn.functional.normalize(row, p=2, dim=-1)
                results.append(row.cpu().numpy().astype(np.float32))
    return results


def save_multivector(path: Path | str, reps: List[np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"reps": reps}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_multivector(path: Path | str) -> List[np.ndarray]:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    return obj["reps"] if isinstance(obj, dict) else obj
