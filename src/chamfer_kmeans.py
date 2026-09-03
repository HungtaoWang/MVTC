"""Chamfer-KMeans: exact SNCD Lloyd iteration with random init."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.constants import MAX_ITER, TOL
from src.ops import normalize_rows, pack_multivecs, pad_truncate_multivec, resolve_torch_device


def run_chamfer_kmeans(
    reps: List[np.ndarray],
    num_clusters: int,
    *,
    n_center_rows: Optional[int] = None,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
    random_state: int = 42,
    device: str = "cuda",
    assign_batch_size: int = 512,
    update_batch_size: int = 1024,
    return_stats: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float]]:
    """Run exact Chamfer-KMeans (SNCD Lloyd) with random initialization."""
    t0 = time.time()
    n = len(reps)
    if n == 0:
        out: Dict[str, float] = {"total_time": 0.0, "actual_iters": 0.0}
        return (np.array([], dtype=np.int32), out) if return_stats else np.array([], dtype=np.int32)

    import torch

    dim = int(reps[0].shape[1])
    proc = [normalize_rows(np.asarray(r if np.asarray(r).size else np.zeros((1, dim)), dtype=np.float64)) for r in reps]
    torch_dev = torch.device(resolve_torch_device(device))
    packed_np, lengths_np = pack_multivecs(proc)
    packed_t = torch.as_tensor(packed_np, dtype=torch.float32, device=torch_dev)
    lengths_t = torch.as_tensor(lengths_np, dtype=torch.int64, device=torch_dev)

    rng = np.random.default_rng(random_state)
    k_mu = max(1, int(n_center_rows if n_center_rows is not None else max(r.shape[0] for r in proc)))
    centers = [
        normalize_rows(pad_truncate_multivec(proc[int(i)], k_mu, rng))
        for i in rng.choice(n, size=num_clusters, replace=False)
    ]

    labels_prev: Optional[np.ndarray] = None
    labels = np.zeros(n, dtype=np.int32)
    actual_iters = 0
    row_positions = torch.arange(int(packed_t.shape[1]), device=torch_dev)

    for it in range(max_iter):
        actual_iters = it + 1
        centers_t = torch.as_tensor(np.stack(centers, axis=0), dtype=torch.float32, device=torch_dev)
        labels = _assign(packed_t, lengths_t, centers_t, row_positions, assign_batch_size)
        if labels_prev is not None and np.array_equal(labels, labels_prev):
            break
        labels_prev = labels.copy()

        max_shift = 0.0
        new_centers: List[np.ndarray] = []
        for c in range(num_clusters):
            idxs = np.flatnonzero(labels == c).tolist()
            old_c = centers[c]
            if not idxs:
                nc = normalize_rows(pad_truncate_multivec(proc[int(rng.integers(0, n))], k_mu, rng))
                new_centers.append(nc)
                max_shift = max(max_shift, float(np.linalg.norm(nc - old_c)))
                continue
            updated = normalize_rows(_update(packed_t, lengths_t, idxs, old_c, k_mu, update_batch_size))
            new_centers.append(updated)
            max_shift = max(max_shift, float(np.linalg.norm(updated - old_c)))
        centers = new_centers
        if max_shift < tol:
            break

    stats = {"total_time": time.time() - t0, "actual_iters": float(actual_iters)}
    return (labels, stats) if return_stats else labels


def _assign(
    packed: "object",
    lengths: "object",
    centers_t: "object",
    row_positions: "object",
    batch_size: int,
) -> np.ndarray:
    import torch

    n = int(packed.shape[0])
    k = int(centers_t.shape[0])
    labels = np.zeros(n, dtype=np.int32)
    k_mu = int(centers_t.shape[1])
    bs = max(1, int(batch_size))

    with torch.no_grad():
        for start in range(0, n, bs):
            end = min(start + bs, n)
            x = packed[start:end]
            lens = lengths[start:end]
            valid = row_positions.unsqueeze(0) < lens.unsqueeze(1)
            bsz, p_max, dim = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
            k_center_rows = int(centers_t.shape[1])
            x_pairs = x[:, None, :, :].expand(bsz, k, p_max, dim).reshape(bsz * k, p_max, dim)
            center_pairs = centers_t[None, :, :, :].expand(bsz, k, k_center_rows, dim).reshape(bsz * k, k_center_rows, dim)
            d2 = torch.cdist(x_pairs, center_pairs, p=2).square().reshape(bsz, k, p_max, k_center_rows)
            forward = d2.amin(dim=3)
            forward = (forward * valid[:, None, :].to(forward.dtype)).sum(dim=2) / lens.to(forward.dtype)[:, None]
            backward = d2.masked_fill(~valid[:, None, :, None], float("inf")).amin(dim=2).sum(dim=2) / float(k_mu)
            labels[start:end] = (forward + backward).argmin(dim=1).detach().cpu().numpy()
    return labels


def _update(
    packed: "object",
    lengths: "object",
    idxs: List[int],
    mu: np.ndarray,
    k_mu: int,
    batch_size: int,
) -> np.ndarray:
    import torch

    device = packed.device
    mu_t = torch.as_tensor(np.asarray(mu, dtype=np.float32), device=device)
    dim = int(mu_t.shape[1])
    num = torch.zeros((k_mu, dim), dtype=torch.float32, device=device)
    den = torch.zeros((k_mu,), dtype=torch.float32, device=device)
    row_positions = torch.arange(int(packed.shape[1]), device=device)
    bs = max(1, int(batch_size))

    with torch.no_grad():
        for start in range(0, len(idxs), bs):
            sub_t = torch.as_tensor(idxs[start : start + bs], dtype=torch.long, device=device)
            x = packed.index_select(0, sub_t)
            x_lengths = lengths.index_select(0, sub_t)
            valid = row_positions.unsqueeze(0) < x_lengths.unsqueeze(1)
            inv_len = 1.0 / x_lengths.clamp(min=1).to(torch.float32)
            d2 = torch.cdist(x, mu_t.unsqueeze(0).expand(int(x.shape[0]), -1, -1), p=2).square()
            f_idx = d2.argmin(dim=2)
            d2_valid = d2.masked_fill(~valid[:, :, None], float("inf"))
            h_idx = d2_valid.argmin(dim=1)
            h_vec = x.gather(1, h_idx.unsqueeze(-1).expand(-1, -1, dim))
            weights = valid.to(torch.float32) * inv_len[:, None]
            x_weighted = x * weights[:, :, None]
            b = int(x.shape[0])
            num_f = torch.zeros((b, k_mu, dim), dtype=torch.float32, device=device)
            num_f.scatter_add_(1, f_idx.unsqueeze(-1).expand(-1, -1, dim), x_weighted)
            den_f = torch.zeros((b, k_mu), dtype=torch.float32, device=device)
            den_f.scatter_add_(1, f_idx, weights)
            num += (h_vec / float(k_mu) + num_f).sum(dim=0)
            den += (1.0 / float(k_mu) + den_f).sum(dim=0)

    out = mu_t.clone()
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok].unsqueeze(1)
    return out.detach().cpu().numpy().astype(np.float64)
