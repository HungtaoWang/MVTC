"""Chamfer-KMeans+ (FDE init, lazy top-k assignment, exact centroid update)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.constants import CANDIDATE_TOPK, MAX_ITER, TOL
from src.fde import RawBlockwiseFDEConfig, RawBlockwiseFDEEmbeddings, build_raw_blockwise_fde_embeddings
from src.ops import normalize_rows, pack_multivecs, pad_truncate_multivec, prepare_reps, resolve_torch_device

_LAZY_REFRESH_MIN_STABLE_RATIO = 0.30
_FDE_INIT_LOCAL_TRIALS = 0
_FDE_INIT_POWER = 1.0


def _center_sncd_movement_bounds(
    reference_centers: List[np.ndarray],
    centers: List[np.ndarray],
) -> np.ndarray:
    bounds = np.zeros(len(centers), dtype=np.float64)
    for j, (old, new) in enumerate(zip(reference_centers, centers)):
        old_np = np.asarray(old, dtype=np.float64)
        new_np = np.asarray(new, dtype=np.float64)
        delta = np.linalg.norm(new_np - old_np, axis=1)
        row_bound = 2.0 * delta
        bounds[j] = float(np.max(row_bound) + np.mean(row_bound))
    return bounds


@dataclass(frozen=True)
class ChamferKMeansPlusConfig:
    """Chamfer-KMeans+ configuration (FDE dual-proposal init, lazy top-k assignment)."""

    fde_dim: int = 256
    max_iter: int = MAX_ITER
    tol: float = TOL
    random_state: int = 42
    device: str = "cuda"
    assign_batch_size: int = 512
    update_batch_size: int = 1024


def _build_fde_config(
    reps: Sequence[np.ndarray],
    cfg: ChamferKMeansPlusConfig,
) -> RawBlockwiseFDEConfig:
    if not reps:
        raise ValueError("cannot build FDE config for an empty dataset")
    n = len(reps)
    # Raw-blockwise FDE on GPU builds a second (n, max_len, dim) tensor alongside
    # packed reps.
    fde_device = str(cfg.device)
    if n > 100_000:
        fde_device = "cpu"
    return RawBlockwiseFDEConfig(
        input_dimension=int(np.asarray(reps[0]).shape[1]),
        num_repetitions=8,
        num_simhash_projections=4,
        projection_dimension=32,
        final_component_dimension=int(cfg.fde_dim),
        seed=int(cfg.random_state),
        count_sketch_device=fde_device,
    )


def _weighted_seed_candidates(
    weights: np.ndarray,
    seeds: Sequence[int],
    rng: np.random.Generator,
    *,
    trials: int,
    power: float,
) -> np.ndarray:
    """Default symmetric FDE-KMeans++ local-trial sampling."""
    n = int(weights.shape[0])
    probs = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if float(power) != 1.0:
        probs = np.power(probs, float(power))
    if seeds:
        probs[np.asarray(seeds, dtype=np.int64)] = 0.0
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 1e-12:
        remaining = np.setdiff1d(np.arange(n, dtype=np.int64), np.asarray(seeds, dtype=np.int64))
        if remaining.size == 0:
            return np.zeros(0, dtype=np.int64)
        return rng.choice(
            remaining,
            size=min(max(1, int(trials)), int(remaining.size)),
            replace=False,
        ).astype(np.int64, copy=False)
    return rng.choice(
        n,
        size=max(1, int(trials)),
        replace=True,
        p=probs / total,
    ).astype(np.int64, copy=False)


def _sncd_multivec_np(x: np.ndarray, y: np.ndarray) -> float:
    """Exact SNCD: forward mean squared L2 plus backward mean squared L2."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.size == 0 or yy.size == 0:
        return float("inf")
    if xx.ndim != 2 or yy.ndim != 2 or xx.shape[1] != yy.shape[1]:
        return float("inf")
    x_norm = np.sum(xx * xx, axis=1)
    y_norm = np.sum(yy * yy, axis=1)
    d2 = x_norm[:, None] + y_norm[None, :] - 2.0 * (xx @ yy.T)
    np.maximum(d2, 0.0, out=d2)
    return float(d2.min(axis=1).mean() + d2.min(axis=0).mean())


def _apply_fde_clip_torch(x: "object") -> "object":
    import torch

    return torch.clamp_min(x, 0.0)


def _fde_distance_to_centers_torch(
    left_t: "object",
    right_t: "object",
    center_left_t: "object",
    center_right_t: "object",
) -> "object":
    """FDE SNCD surrogate using torch GEMM/dot products."""
    forward = 4.0 - 2.0 * (left_t @ center_right_t.T)
    backward = 4.0 - 2.0 * (right_t @ center_left_t.T)
    forward = _apply_fde_clip_torch(forward)
    backward = _apply_fde_clip_torch(backward)
    return 0.5 * (forward + backward)


def _fde_directional_split_torch(
    left_t: "object",
    right_t: "object",
    center_left_t: "object",
    center_right_t: "object",
) -> Tuple["object", "object"]:
    """Split-half directional Chamfer terms F and B in FDE space."""
    import torch

    h = int(left_t.shape[1] // 2)

    q0_x = left_t[:, :h]
    d1_x = left_t[:, h:]
    d0_x = right_t[:, :h]
    q1_x = right_t[:, h:]

    q0_m = center_left_t[:, :h]
    d1_m = center_left_t[:, h:]
    d0_m = center_right_t[:, :h]
    q1_m = center_right_t[:, h:]

    sim_xm = 0.5 * (q0_x @ d0_m.T + q1_x @ d1_m.T)
    sim_mx = 0.5 * (d0_x @ q0_m.T + d1_x @ q1_m.T)

    f_dir = 2.0 - 2.0 * sim_xm
    b_dir = 2.0 - 2.0 * sim_mx
    f_dir = torch.clamp_min(f_dir, 0.0)
    b_dir = torch.clamp_min(b_dir, 0.0)
    return f_dir, b_dir


def _fde_coverage_distance_torch(
    left_t: "object",
    right_t: "object",
    center_left_t: "object",
    center_right_t: "object",
) -> "object":
    """Coverage proposal distance (hard min of directional terms)."""
    import torch

    f_dir, b_dir = _fde_directional_split_torch(
        left_t,
        right_t,
        center_left_t,
        center_right_t,
    )
    return 2.0 * torch.minimum(f_dir, b_dir)


def _fde_topk_excluding_old_torch(
    left_t: "object",
    right_t: "object",
    center_left: np.ndarray,
    center_right: np.ndarray,
    row_indices: np.ndarray,
    old_labels: np.ndarray,
    topk: int,
) -> np.ndarray:
    """Recall FDE nearest centroids with torch dot kernels."""
    import torch

    rows_np = np.asarray(row_indices, dtype=np.int64)
    old_np = np.asarray(old_labels, dtype=np.int64)
    if rows_np.size == 0:
        return np.zeros((0, int(topk)), dtype=np.int64)

    with torch.no_grad():
        rows_t = torch.as_tensor(rows_np, dtype=torch.long, device=left_t.device)
        old_t = torch.as_tensor(old_np, dtype=torch.long, device=left_t.device)
        center_left_t = torch.as_tensor(center_left, dtype=torch.float32, device=left_t.device)
        center_right_t = torch.as_tensor(center_right, dtype=torch.float32, device=left_t.device)
        dist = _fde_distance_to_centers_torch(
            left_t.index_select(0, rows_t),
            right_t.index_select(0, rows_t),
            center_left_t,
            center_right_t,
        )
        dist.scatter_(1, old_t[:, None], float("inf"))
        candidates = torch.topk(
            dist,
            k=max(1, min(int(topk), int(dist.shape[1]) - 1)),
            dim=1,
            largest=False,
            sorted=True,
        ).indices
        return candidates.detach().cpu().numpy().astype(np.int64, copy=False)


def _torch_pairwise_sq_l2_dot(x_pairs: "object", center_pairs: "object") -> "object":
    """Squared L2 via ||x||^2 + ||y||^2 - 2 x·y (batched bmm)."""
    import torch

    dots = torch.bmm(x_pairs, center_pairs.transpose(1, 2))
    x_norm = torch.sum(x_pairs * x_pairs, dim=2)
    center_norm = torch.sum(center_pairs * center_pairs, dim=2)
    return torch.clamp(x_norm[:, :, None] + center_norm[:, None, :] - 2.0 * dots, min=0.0)


def _torch_pairwise_sq_l2(x_pairs: "object", center_pairs: "object") -> "object":
    return _torch_pairwise_sq_l2_dot(x_pairs, center_pairs)


def _exact_multivec_distances_torch_packed(
    packed: "object",
    lengths: "object",
    centers_t: "object",
    *,
    batch_size: int,
    doc_indices: Optional[np.ndarray] = None,
    candidate_indices: Optional[np.ndarray] = None,
    valid_mask: Optional["object"] = None,
) -> np.ndarray:
    """Return exact SNCD distances (dot-kernel or cdist)."""
    import torch

    doc_idx = None if doc_indices is None else np.asarray(doc_indices, dtype=np.int64)
    n_docs = int(packed.shape[0]) if doc_idx is None else int(doc_idx.size)
    k = int(centers_t.shape[0])
    if n_docs == 0 or k == 0:
        cols = 0 if candidate_indices is None else int(np.asarray(candidate_indices).shape[1])
        return np.zeros((n_docs, cols), dtype=np.float32)

    cand = None
    if candidate_indices is not None:
        cand = np.asarray(candidate_indices, dtype=np.int64)
        if cand.ndim != 2 or cand.shape[0] != n_docs:
            raise ValueError(
                "candidate_indices must have shape (len(doc_indices), C), "
                f"got {cand.shape}, len(doc_indices)={n_docs}"
            )
        cand = np.clip(cand, 0, k - 1)
        out_cols = int(cand.shape[1])
    else:
        out_cols = k

    bs = max(1, int(batch_size))
    k_center_rows = int(centers_t.shape[1])
    row_positions = torch.arange(int(packed.shape[1]), device=packed.device)
    out_t = torch.empty((n_docs, out_cols), dtype=torch.float32, device=packed.device)

    with torch.no_grad():
        for start in range(0, n_docs, bs):
            end = min(start + bs, n_docs)
            if doc_idx is None:
                x = packed[start:end]
                lens = lengths[start:end]
                valid = (
                    valid_mask[start:end]
                    if valid_mask is not None
                    else row_positions.unsqueeze(0) < lens.unsqueeze(1)
                )
            else:
                rows_np = doc_idx[start:end]
                rows_t = torch.as_tensor(rows_np, dtype=torch.long, device=packed.device)
                x = packed.index_select(0, rows_t)
                lens = lengths.index_select(0, rows_t)
                valid = (
                    valid_mask.index_select(0, rows_t)
                    if valid_mask is not None
                    else row_positions.unsqueeze(0) < lens.unsqueeze(1)
                )
            bsz = int(x.shape[0])
            p_max = int(x.shape[1])
            dim = int(x.shape[2])

            if cand is None:
                c_per_doc = k
                x_pairs = (
                    x[:, None, :, :]
                    .expand(bsz, k, p_max, dim)
                    .reshape(bsz * k, p_max, dim)
                )
                center_pairs = (
                    centers_t[None, :, :, :]
                    .expand(bsz, k, k_center_rows, dim)
                    .reshape(bsz * k, k_center_rows, dim)
                )
            else:
                cand_chunk = cand[start:end]
                cand_t = torch.as_tensor(cand_chunk, dtype=torch.long, device=packed.device)
                centers_sel = centers_t[cand_t]
                c_per_doc = int(cand_chunk.shape[1])
                x_pairs = (
                    x[:, None, :, :]
                    .expand(bsz, c_per_doc, p_max, dim)
                    .reshape(bsz * c_per_doc, p_max, dim)
                )
                center_pairs = centers_sel.reshape(bsz * c_per_doc, k_center_rows, dim)

            d2 = _torch_pairwise_sq_l2(x_pairs, center_pairs).reshape(
                bsz,
                c_per_doc,
                p_max,
                k_center_rows,
            )
            forward = d2.amin(dim=3)
            forward = (forward * valid[:, None, :].to(forward.dtype)).sum(dim=2)
            forward = forward / lens.to(forward.dtype)[:, None]

            backward = d2.masked_fill(~valid[:, None, :, None], float("inf"))
            backward = backward.amin(dim=2).sum(dim=2) / float(k_center_rows)
            out_t[start:end] = (forward + backward).to(torch.float32)
    return out_t.detach().cpu().numpy().astype(np.float32, copy=False)


def _exact_assigned_center_distances_torch_packed(
    packed: "object",
    lengths: "object",
    centers_t: "object",
    labels: np.ndarray,
    *,
    batch_size: int,
    valid_mask: Optional["object"] = None,
) -> np.ndarray:
    """Return exact assigned-center SNCD."""
    import torch

    labels_np = np.asarray(labels, dtype=np.int64)
    n_docs = int(labels_np.size)
    if n_docs == 0:
        return np.empty(0, dtype=np.float32)

    bs = max(1, int(batch_size))
    k_center_rows = int(centers_t.shape[1])
    row_positions = torch.arange(int(packed.shape[1]), device=packed.device)
    out_t = torch.empty(n_docs, dtype=torch.float32, device=packed.device)

    with torch.no_grad():
        for start in range(0, n_docs, bs):
            end = min(start + bs, n_docs)
            x = packed[start:end]
            lens = lengths[start:end]
            valid = (
                valid_mask[start:end]
                if valid_mask is not None
                else row_positions.unsqueeze(0) < lens.unsqueeze(1)
            )
            label_t = torch.as_tensor(labels_np[start:end], dtype=torch.long, device=packed.device)
            centers_sel = centers_t.index_select(0, label_t)

            d2 = _torch_pairwise_sq_l2(x, centers_sel)
            forward = d2.amin(dim=2)
            forward = (forward * valid.to(forward.dtype)).sum(dim=1)
            forward = forward / lens.to(forward.dtype)

            backward = d2.masked_fill(~valid[:, :, None], float("inf"))
            backward = backward.amin(dim=1).sum(dim=1) / float(k_center_rows)
            out_t[start:end] = (forward + backward).to(torch.float32)
    return out_t.detach().cpu().numpy().astype(np.float32, copy=False)


def _mvkmeans_update_mu_rows_torch_packed(
    packed: "object",
    lengths: "object",
    idxs: List[int],
    mu: np.ndarray,
    k_mu: int,
    *,
    batch_size: int,
) -> np.ndarray:
    """Exact Chamfer centroid update."""
    import torch

    if not idxs:
        return np.asarray(mu, dtype=np.float64)

    device = packed.device
    mu_t = torch.as_tensor(np.asarray(mu, dtype=np.float32), dtype=torch.float32, device=device)
    dim = int(mu_t.shape[1])
    num = torch.zeros((k_mu, dim), dtype=torch.float32, device=device)
    den = torch.zeros((k_mu,), dtype=torch.float32, device=device)
    bs = max(1, int(batch_size))
    row_positions = torch.arange(int(packed.shape[1]), device=device)

    with torch.no_grad():
        for start in range(0, len(idxs), bs):
            sub = idxs[start : start + bs]
            sub_t = torch.as_tensor(sub, dtype=torch.long, device=device)
            x = packed.index_select(0, sub_t)
            x_lengths = lengths.index_select(0, sub_t)
            valid = row_positions.unsqueeze(0) < x_lengths.unsqueeze(1)
            inv_len = 1.0 / x_lengths.clamp(min=1).to(torch.float32)

            mu_batch = mu_t.unsqueeze(0).expand(int(x.shape[0]), -1, -1)
            d2 = _torch_pairwise_sq_l2(x, mu_batch)
            f_idx = d2.argmin(dim=2)

            d2_valid = d2.masked_fill(~valid[:, :, None], float("inf"))
            h_idx = d2_valid.argmin(dim=1)
            h_vec = x.gather(1, h_idx.unsqueeze(-1).expand(-1, -1, dim))

            weights = valid.to(torch.float32) * inv_len[:, None]
            x_weighted = x * weights[:, :, None]

            b = int(x.shape[0])
            num_f = torch.zeros((b, k_mu, dim), dtype=torch.float32, device=device)
            num_f.scatter_add_(
                1,
                f_idx.unsqueeze(-1).expand(-1, -1, dim),
                x_weighted,
            )

            den_f = torch.zeros((b, k_mu), dtype=torch.float32, device=device)
            den_f.scatter_add_(1, f_idx, weights)

            num += (h_vec / float(k_mu) + num_f).sum(dim=0)
            den += (1.0 / float(k_mu) + den_f).sum(dim=0)

    out = mu_t.clone()
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok].unsqueeze(1)
    return out.detach().cpu().numpy().astype(np.float64)


def _resolve_local_trials(local_trials: int, num_clusters: int) -> int:
    trials = int(local_trials)
    if trials <= 0:
        trials = 2 + int(np.log(max(2, int(num_clusters))))
    return max(1, trials)


def _select_dual_proposal_seeds(
    left_t: "object",
    right_t: "object",
    num_clusters: int,
    rng: np.random.Generator,
    cov_rng: np.random.Generator,
    *,
    local_trials: int,
    power: float,
) -> List[int]:
    """Dual proposal init: hard coverage proposes, FDE SNCD surrogate selects."""
    import torch

    n = int(left_t.shape[0])
    k = int(num_clusters)
    if k <= 0 or k > n:
        raise ValueError(f"need 0 < num_clusters <= n, got k={k}, n={n}")

    trials = _resolve_local_trials(local_trials, k)
    power = max(0.0, float(power))

    def _obj_dist(center_left_t: "object", center_right_t: "object") -> "object":
        return _fde_distance_to_centers_torch(left_t, right_t, center_left_t, center_right_t)

    def _cov_dist(center_left_t: "object", center_right_t: "object") -> "object":
        return _fde_coverage_distance_torch(left_t, right_t, center_left_t, center_right_t)

    seeds: List[int] = [int(rng.integers(0, n))]
    with torch.no_grad():
        first_t = torch.as_tensor(seeds, dtype=torch.long, device=left_t.device)
        min_dist_obj = _obj_dist(
            left_t.index_select(0, first_t),
            right_t.index_select(0, first_t),
        )[:, 0]
        min_dist_cov = _cov_dist(
            left_t.index_select(0, first_t),
            right_t.index_select(0, first_t),
        )[:, 0]
        min_dist_obj[seeds[0]] = 0.0
        min_dist_cov[seeds[0]] = 0.0

    for _ in range(1, k):
        weights_obj = torch.clamp_min(min_dist_obj, 0.0).detach().cpu().numpy()
        weights_cov = torch.clamp_min(min_dist_cov, 0.0).detach().cpu().numpy()

        cand_obj = _weighted_seed_candidates(weights_obj, seeds, rng, trials=trials, power=power)
        cand_cov = _weighted_seed_candidates(weights_cov, seeds, cov_rng, trials=trials, power=power)
        if cand_obj.size == 0 or cand_cov.size == 0:
            remaining = np.asarray([i for i in range(n) if i not in set(seeds)], dtype=np.int64)
            if remaining.size == 0:
                break
            fallback = rng.choice(remaining, size=1, replace=False).astype(np.int64, copy=False)
            if cand_obj.size == 0:
                cand_obj = fallback
            if cand_cov.size == 0:
                cand_cov = fallback

        obj_set = set(int(x) for x in np.unique(cand_obj.astype(np.int64, copy=False)).tolist())
        cov_set = set(int(x) for x in np.unique(cand_cov.astype(np.int64, copy=False)).tolist())
        merged = np.asarray(sorted(obj_set | cov_set), dtype=np.int64)

        with torch.no_grad():
            cand_t = torch.as_tensor(merged, dtype=torch.long, device=left_t.device)
            cand_dist_obj = _obj_dist(
                left_t.index_select(0, cand_t),
                right_t.index_select(0, cand_t),
            )
            potentials = torch.minimum(min_dist_obj[:, None], cand_dist_obj).sum(dim=0)
            best_pos = int(torch.argmin(potentials).item())
            best_seed = int(merged[best_pos])

            cand_dist_cov = _cov_dist(
                left_t.index_select(0, cand_t),
                right_t.index_select(0, cand_t),
            )
            min_dist_obj = torch.minimum(min_dist_obj, cand_dist_obj[:, best_pos])
            min_dist_cov = torch.minimum(min_dist_cov, cand_dist_cov[:, best_pos])

        seeds.append(best_seed)
        seed_t = torch.as_tensor(seeds, dtype=torch.long, device=left_t.device)
        min_dist_obj[seed_t] = 0.0
        min_dist_cov[seed_t] = 0.0

    return seeds


def _init_optimized_centers(
    reps: List[np.ndarray],
    num_clusters: int,
    k_mu: int,
    rng: np.random.Generator,
    *,
    cfg: ChamferKMeansPlusConfig,
) -> Tuple[List[np.ndarray], RawBlockwiseFDEEmbeddings, RawBlockwiseFDEConfig]:
    fde_cfg = _build_fde_config(reps, cfg)
    emb = build_raw_blockwise_fde_embeddings(reps, fde_cfg, spherical=False)

    import torch

    torch_dev = torch.device(resolve_torch_device(str(cfg.device)))
    left_t = torch.as_tensor(emb.left, dtype=torch.float32, device=torch_dev)
    right_t = torch.as_tensor(emb.right, dtype=torch.float32, device=torch_dev)

    cov_rng = np.random.default_rng(int(cfg.random_state) + 1_000_003)
    seeds = _select_dual_proposal_seeds(
        left_t,
        right_t,
        int(num_clusters),
        rng,
        cov_rng,
        local_trials=_FDE_INIT_LOCAL_TRIALS,
        power=_FDE_INIT_POWER,
    )
    centers = [normalize_rows(pad_truncate_multivec(reps[int(i)], int(k_mu), rng)) for i in seeds]
    return centers, emb, fde_cfg


def run_chamfer_kmeans_plus(
    reps: List[np.ndarray],
    num_clusters: int,
    *,
    config: Optional[ChamferKMeansPlusConfig] = None,
    return_stats: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float]]:
    """Run Chamfer-KMeans+ (FDE init + lazy top-k assignment)."""
    t0 = time.time()
    cfg = config or ChamferKMeansPlusConfig()
    n = len(reps)
    if n == 0:
        labels = np.array([], dtype=np.int32)
        out: Dict[str, float] = {"total_time": 0.0, "actual_iters": 0.0}
        return (labels, out) if return_stats else labels
    if int(num_clusters) <= 0 or int(num_clusters) > n:
        raise ValueError(f"need 0 < num_clusters <= n, got k={num_clusters}, n={n}")

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Chamfer-KMeans+ requires torch") from exc

    reps = prepare_reps(reps, spherical=True, dtype=np.float64)
    k_mu = max(1, max(int(rep.shape[0]) for rep in reps))
    rng = np.random.default_rng(int(cfg.random_state))

    torch_dev = torch.device(resolve_torch_device(str(cfg.device)))
    packed_np, lengths_np = pack_multivecs(reps, dtype=np.float32)
    packed_t = torch.as_tensor(packed_np, dtype=torch.float32, device=torch_dev)
    lengths_t = torch.as_tensor(lengths_np, dtype=torch.int64, device=torch_dev)
    valid_mask_t = (
        torch.arange(int(packed_t.shape[1]), device=torch_dev).unsqueeze(0)
        < lengths_t.unsqueeze(1)
    )

    centers, fde_emb, fde_cfg = _init_optimized_centers(
        reps,
        int(num_clusters),
        k_mu,
        rng,
        cfg=cfg,
    )

    fde_left_t = torch.as_tensor(fde_emb.left, dtype=torch.float32, device=torch_dev)
    fde_right_t = torch.as_tensor(fde_emb.right, dtype=torch.float32, device=torch_dev)

    lazy_ref_centers: Optional[List[np.ndarray]] = None
    lazy_ref_dist: Optional[np.ndarray] = None
    lazy_prev_labels: Optional[np.ndarray] = None
    lazy_force_refresh_next = True
    prev_center_shift = float("inf")

    labels = np.zeros(n, dtype=np.int32)
    actual_iters = 0

    for it in range(int(cfg.max_iter)):
        actual_iters = it + 1
        centers_t = torch.as_tensor(np.stack(centers, axis=0), dtype=torch.float32, device=torch_dev)
        do_refresh = (
            lazy_force_refresh_next
            or lazy_ref_centers is None
            or lazy_ref_dist is None
            or lazy_prev_labels is None
        )

        if do_refresh:
            dist = _exact_multivec_distances_torch_packed(
                packed_t,
                lengths_t,
                centers_t,
                batch_size=int(cfg.assign_batch_size),
                valid_mask=valid_mask_t,
            )
            labels = dist.argmin(axis=1).astype(np.int32, copy=False)
            lazy_ref_dist = dist
            lazy_ref_centers = [np.asarray(center, dtype=np.float64).copy() for center in centers]
            lazy_force_refresh_next = False
        else:
            assert lazy_ref_dist is not None
            assert lazy_ref_centers is not None
            assert lazy_prev_labels is not None

            movement_bounds = _center_sncd_movement_bounds(lazy_ref_centers, centers)
            old_labels = lazy_prev_labels.astype(np.int64, copy=False)
            old_label_dist = _exact_assigned_center_distances_torch_packed(
                packed_t,
                lengths_t,
                centers_t,
                old_labels,
                batch_size=int(cfg.assign_batch_size),
                valid_mask=valid_mask_t,
            )
            lower_bounds = lazy_ref_dist - movement_bounds[None, :]
            lower_bounds[np.arange(n), old_labels] = np.inf
            stable = old_label_dist <= (lower_bounds.min(axis=1) + 1e-12)
            unstable_idx = np.flatnonzero(~stable).astype(np.int64, copy=False)

            labels = old_labels.astype(np.int32, copy=True)

            if unstable_idx.size and int(num_clusters) > 1:
                unstable_old = old_labels[unstable_idx]
                center_emb = build_raw_blockwise_fde_embeddings(
                    centers,
                    fde_cfg,
                    spherical=False,
                )
                topk = max(1, min(CANDIDATE_TOPK, int(num_clusters) - 1))
                other_candidates = _fde_topk_excluding_old_torch(
                    fde_left_t,
                    fde_right_t,
                    center_emb.left,
                    center_emb.right,
                    unstable_idx,
                    unstable_old,
                    topk,
                )
                unstable_dist = _exact_multivec_distances_torch_packed(
                    packed_t,
                    lengths_t,
                    centers_t,
                    batch_size=int(cfg.assign_batch_size),
                    doc_indices=unstable_idx,
                    candidate_indices=other_candidates,
                    valid_mask=valid_mask_t,
                )
                best_other_pos = unstable_dist.argmin(axis=1)
                best_other_dist = unstable_dist[np.arange(int(unstable_idx.size)), best_other_pos]
                best_other_label = other_candidates[
                    np.arange(int(unstable_idx.size)),
                    best_other_pos,
                ]
                keep_old = old_label_dist[unstable_idx] <= (best_other_dist + 1e-12)
                labels[unstable_idx] = np.where(
                    keep_old,
                    unstable_old,
                    best_other_label,
                ).astype(np.int32, copy=False)

            stable_ratio = float(np.count_nonzero(stable)) / float(max(1, n))
            if stable_ratio < _LAZY_REFRESH_MIN_STABLE_RATIO:
                lazy_force_refresh_next = True

        lazy_prev_labels = labels.copy()

        max_center_shift = 0.0
        new_centers: List[np.ndarray] = []
        for c in range(int(num_clusters)):
            idxs = np.flatnonzero(labels == c).tolist()
            old_c = centers[c]
            if not idxs:
                ri = int(rng.integers(0, n))
                nc = normalize_rows(pad_truncate_multivec(reps[ri], k_mu, rng))
                new_centers.append(nc)
                max_center_shift = max(max_center_shift, _sncd_multivec_np(nc, old_c))
                continue

            updated = _mvkmeans_update_mu_rows_torch_packed(
                packed_t,
                lengths_t,
                idxs,
                old_c,
                k_mu,
                batch_size=int(cfg.update_batch_size),
            )
            updated = normalize_rows(updated)
            new_centers.append(updated)
            max_center_shift = max(max_center_shift, _sncd_multivec_np(updated, old_c))
        centers = new_centers
        if max_center_shift < float(cfg.tol) and prev_center_shift < float(cfg.tol):
            break
        prev_center_shift = float(max_center_shift)

    stats = {"total_time": time.time() - t0, "actual_iters": float(actual_iters)}
    if return_stats:
        return labels, stats
    return labels
