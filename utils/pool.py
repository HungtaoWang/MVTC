"""ToMe bipartite pooling."""

from __future__ import annotations

from typing import List

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _merge_once(x: np.ndarray, sizes: np.ndarray, remove: int) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    r = min(max(0, remove), n // 2)
    if r == 0:
        return x.copy(), sizes.copy()
    m = _l2(x)
    scores = m[::2] @ m[1::2].T
    dst = np.argmax(scores, axis=1)
    ranked = np.argsort(-scores[np.arange(scores.shape[0]), dst], kind="stable")
    merged, kept = ranked[:r], ranked[r:]
    sx, dx = x[::2], x[1::2]
    ss, ds = sizes[::2], sizes[1::2]
    odx = dx * ds[:, None]
    ods = ds.copy()
    for i in merged:
        j = int(dst[i])
        odx[j] += sx[i] * ss[i]
        ods[j] += ss[i]
    odx /= ods[:, None]
    out_x = np.concatenate([sx[kept], odx], axis=0)
    out_s = np.concatenate([ss[kept], ods], axis=0)
    return out_x, out_s


def pool_tome(tokens: np.ndarray, target_m: int) -> np.ndarray:
    x = _l2(np.asarray(tokens, dtype=np.float64))
    target = max(1, min(int(target_m), x.shape[0]))
    sizes = np.ones(x.shape[0])
    while x.shape[0] > target:
        x, sizes = _merge_once(x, sizes, min(x.shape[0] - target, x.shape[0] // 2))
    return _l2(x)


def pool_corpus(reps: List[np.ndarray], target_m: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for r in reps:
        rr = np.asarray(r, dtype=np.float64)
        if rr.size == 0:
            out.append(np.zeros((1, 1)))
            continue
        m = max(1, min(int(target_m), rr.shape[0]))
        out.append(pool_tome(rr, m))
    return out
