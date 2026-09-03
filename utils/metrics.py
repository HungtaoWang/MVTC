"""Clustering metrics."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score


def _purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct = 0
    for c in np.unique(y_pred):
        mask = y_pred == c
        if mask.any():
            correct += np.bincount(y_true[mask]).max()
    return correct / len(y_true)


def _inv_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct = 0
    for c in np.unique(y_true):
        mask = y_true == c
        if mask.any():
            correct += np.bincount(y_pred[mask]).max()
    return correct / len(y_true)


def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for a, b in zip(y_pred, y_true):
        w[a, b] += 1
    r, c = linear_sum_assignment(w.max() - w)
    return w[r, c].sum() / len(y_true)


def evaluate(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    p, ip = _purity(yt, yp), _inv_purity(yt, yp)
    fp = 0.0 if p + ip == 0 else 2 * p * ip / (p + ip)
    return {
        "nmi": float(normalized_mutual_info_score(yt, yp)),
        "acc": float(_acc(yt, yp)),
        "fp_measure": float(fp),
    }
