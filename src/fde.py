"""Raw blockwise FDE embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from src.ops import normalize_rows, prepare_reps, resolve_torch_device


@dataclass(frozen=True)
class RawBlockwiseFDEConfig:
    input_dimension: int
    num_repetitions: int = 8
    num_simhash_projections: int = 4
    projection_dimension: int = 32
    final_component_dimension: int = 512
    seed: int = 42
    count_sketch_device: str = "auto"

    @property
    def num_partitions(self) -> int:
        return 1 << int(self.num_simhash_projections)

    @property
    def raw_component_dimension(self) -> int:
        return int(self.num_repetitions) * int(self.num_partitions) * int(self.projection_dimension)


@dataclass
class RawBlockwiseFDEEmbeddings:
    left: np.ndarray
    right: np.ndarray
    lengths: np.ndarray
    config: RawBlockwiseFDEConfig


def _simhash_matrix(dimension: int, num_projections: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(dimension, num_projections)).astype(np.float32)


def _ams_projection_matrix(dimension: int, projection_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((dimension, projection_dim), dtype=np.float32)
    indices = rng.integers(0, projection_dim, size=dimension)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=dimension)
    out[np.arange(dimension), indices] = signs
    return out


def _partition_indices_from_sketches(sketches: np.ndarray) -> np.ndarray:
    bits = (sketches > 0).astype(np.uint32, copy=False)
    partition_indices = np.zeros(sketches.shape[0], dtype=np.uint32)
    for bit_idx in range(sketches.shape[1]):
        partition_indices = (partition_indices << 1) + (bits[:, bit_idx] ^ (partition_indices & 1))
    return partition_indices.astype(np.int64, copy=False)


def _partition_indices_from_sketches_torch(sketches: "object") -> "object":
    import torch

    bits = (sketches > 0).to(torch.int64)
    partition_indices = torch.zeros(sketches.shape[:-1], dtype=torch.int64, device=sketches.device)
    for bit_idx in range(int(sketches.shape[-1])):
        partition_indices = (partition_indices << 1) + torch.bitwise_xor(
            bits[..., bit_idx],
            partition_indices & 1,
        )
    return partition_indices


def _gray_code_to_binary(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.uint32).copy()
    mask = out >> 1
    while np.any(mask):
        out ^= mask
        mask >>= 1
    return out


def _binary_bits_for_partitions(partitions: np.ndarray, num_bits: int) -> np.ndarray:
    binary = _gray_code_to_binary(np.asarray(partitions, dtype=np.uint32))
    shifts = np.arange(num_bits - 1, -1, -1, dtype=np.uint32)
    return ((binary[:, None] >> shifts[None, :]) & 1).astype(bool, copy=False)


def _partition_hamming_table(num_bits: int) -> np.ndarray:
    bits = np.asarray(
        np.meshgrid(*([[False, True]] * int(num_bits)), indexing="ij"),
        dtype=bool,
    ).reshape(int(num_bits), -1).T
    sketches = np.where(bits, 1.0, -1.0).astype(np.float32, copy=False)
    partition_indices = _partition_indices_from_sketches(sketches)
    token_bits = np.zeros_like(bits)
    token_bits[partition_indices] = bits
    target_bits = _binary_bits_for_partitions(np.arange(1 << int(num_bits), dtype=np.uint32), int(num_bits))
    return np.count_nonzero(token_bits[:, None, :] != target_bits[None, :, :], axis=2).astype(np.uint8)


def _aggregate_projected_by_partition(
    projected: np.ndarray,
    partitions: np.ndarray,
    lengths: np.ndarray,
    boundaries: np.ndarray,
    *,
    num_partitions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(lengths.size)
    projection_dim = int(projected.shape[1])
    sums = np.zeros((n, num_partitions, projection_dim), dtype=np.float32)
    counts = np.zeros((n, num_partitions), dtype=np.int32)
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == int(length)).astype(np.int64, copy=False)
        offsets = boundaries[rows]
        for pos in range(int(length)):
            point_indices = offsets + int(pos)
            part = partitions[point_indices]
            counts[rows, part] += 1
            sums[rows, part, :] += projected[point_indices, :]
    return sums, counts


def _fill_empty_partitions(
    doc_values: np.ndarray,
    counts: np.ndarray,
    projected: np.ndarray,
    partitions: np.ndarray,
    lengths: np.ndarray,
    boundaries: np.ndarray,
    num_bits: int,
) -> None:
    empty_mask = counts == 0
    if not np.any(empty_mask):
        return
    hamming = _partition_hamming_table(int(num_bits))
    for doc_idx in range(int(lengths.size)):
        empty_parts = np.flatnonzero(empty_mask[doc_idx])
        if empty_parts.size == 0:
            continue
        start, end = int(boundaries[doc_idx]), int(boundaries[doc_idx + 1])
        distances = hamming[partitions[start:end]][:, empty_parts]
        nearest = np.argmin(distances, axis=0)
        doc_values[doc_idx, empty_parts, :] = projected[start + nearest]


def _count_sketch_batch(matrix: np.ndarray, final_dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    src = np.asarray(matrix, dtype=np.float32, order="C")
    source_dim = int(src.shape[1])
    indices = rng.integers(0, int(final_dimension), size=source_dim)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=source_dim)
    out = np.zeros((src.shape[0], int(final_dimension)), dtype=np.float32)
    for bucket in range(int(final_dimension)):
        cols = np.flatnonzero(indices == bucket)
        if cols.size:
            out[:, bucket] = src[:, cols] @ signs[cols]
    return out


def _count_sketch_torch_many(src: "object", final_dimension: int, seeds: Sequence[int]) -> Tuple["object", ...]:
    import torch

    source_dim = int(src.shape[1])
    final_dim = int(final_dimension)
    outs = []
    with torch.no_grad():
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            indices = torch.from_numpy(rng.integers(0, final_dim, size=source_dim, dtype=np.int64)).to(src.device)
            signs = torch.from_numpy(
                rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=source_dim)
            ).to(src.device)
            indices = indices.view(1, -1).expand(src.shape[0], -1)
            signs = signs.view(1, -1)
            out = torch.zeros((src.shape[0], final_dim), device=src.device, dtype=torch.float32)
            out.scatter_add_(1, indices, src * signs)
            outs.append(out)
    return tuple(outs)


def _count_sketch_batch_many(
    matrix: np.ndarray,
    final_dimension: int,
    seeds: Sequence[int],
    *,
    device: str = "cpu",
) -> Tuple[Tuple[np.ndarray, ...], str]:
    use_cuda = device.startswith("cuda") and resolve_torch_device("cuda") == "cuda"
    if not use_cuda:
        return tuple(_count_sketch_batch(matrix, final_dimension, int(s)) for s in seeds), "cpu"

    import torch

    src = torch.from_numpy(np.asarray(matrix, dtype=np.float32, order="C")).cuda()
    outs = []
    with torch.no_grad():
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            source_dim = int(src.shape[1])
            final_dim = int(final_dimension)
            indices = torch.from_numpy(rng.integers(0, final_dim, size=source_dim, dtype=np.int64)).cuda()
            signs = torch.from_numpy(
                rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=source_dim)
            ).cuda()
            indices = indices.view(1, -1).expand(src.shape[0], -1)
            signs = signs.view(1, -1)
            out = torch.zeros((src.shape[0], final_dim), device=src.device, dtype=torch.float32)
            out.scatter_add_(1, indices, src * signs)
            outs.append(out.cpu().numpy())
    return tuple(outs), "cuda"


def _sketch_device(device: str) -> str:
    if str(device).lower() in ("auto", "cuda") or str(device).startswith("cuda"):
        return resolve_torch_device("cuda")
    return "cpu"


def _raw_fde_torch(
    reps: Sequence[np.ndarray],
    config: RawBlockwiseFDEConfig,
    *,
    spherical: bool,
) -> Tuple["object", "object", np.ndarray] | None:
    if _sketch_device(config.count_sketch_device) != "cuda":
        return None
    import torch

    proc = prepare_reps(reps, spherical=spherical, dtype=np.float32)
    n = len(proc)
    lengths = np.asarray([max(1, int(r.shape[0])) for r in proc], dtype=np.int64)
    max_len = int(lengths.max())
    dim = int(config.input_dimension)
    points_np = np.zeros((n, max_len, dim), dtype=np.float32)
    valid_np = np.zeros((n, max_len), dtype=bool)
    for doc_idx, rep in enumerate(proc):
        length = int(lengths[doc_idx])
        points_np[doc_idx, :length] = rep[:length]
        valid_np[doc_idx, :length] = True

    torch_dev = torch.device("cuda")
    hamming = torch.as_tensor(
        _partition_hamming_table(int(config.num_simhash_projections)),
        dtype=torch.int16,
        device=torch_dev,
    )
    rep_dim = int(config.num_partitions) * int(config.projection_dimension)
    num_partitions = int(config.num_partitions)
    projection_dim = int(config.projection_dimension)
    scale = float(np.sqrt(float(config.num_repetitions)))

    with torch.no_grad():
        points = torch.as_tensor(points_np, dtype=torch.float32, device=torch_dev)
        valid = torch.as_tensor(valid_np, dtype=torch.bool, device=torch_dev)
        valid_f = valid.to(torch.float32)
        raw_query = torch.empty((n, config.raw_component_dimension), dtype=torch.float32, device=torch_dev)
        raw_doc = torch.empty_like(raw_query)

        for rep in range(int(config.num_repetitions)):
            seed = int(config.seed) + rep
            simhash = torch.as_tensor(
                _simhash_matrix(dim, int(config.num_simhash_projections), seed),
                dtype=torch.float32,
                device=torch_dev,
            )
            ams = torch.as_tensor(
                _ams_projection_matrix(dim, projection_dim, seed),
                dtype=torch.float32,
                device=torch_dev,
            )
            sketches = (points.reshape(n * max_len, dim) @ simhash).reshape(n, max_len, -1)
            partitions = _partition_indices_from_sketches_torch(sketches)
            projected = (points.reshape(n * max_len, dim) @ ams).reshape(n, max_len, projection_dim)
            projected = projected * valid_f.unsqueeze(-1)

            sums = torch.zeros((n, num_partitions, projection_dim), dtype=torch.float32, device=torch_dev)
            part_index = partitions.unsqueeze(-1).expand(-1, -1, projection_dim)
            sums.scatter_add_(1, part_index, projected)
            counts = torch.zeros((n, num_partitions), dtype=torch.float32, device=torch_dev)
            counts.scatter_add_(1, partitions, valid_f)

            start_col = rep * rep_dim
            raw_query[:, start_col : start_col + rep_dim] = sums.reshape(n, rep_dim)
            doc_values = sums / torch.clamp(counts.unsqueeze(-1), min=1.0)
            empty_mask = counts == 0
            if bool(torch.any(empty_mask).item()):
                distances = hamming[partitions].to(torch.int16).masked_fill(~valid.unsqueeze(-1), 32767)
                nearest = torch.argmin(distances, dim=1)
                nearest_values = projected.gather(1, nearest.unsqueeze(-1).expand(-1, -1, projection_dim))
                doc_values = torch.where(empty_mask.unsqueeze(-1), nearest_values, doc_values)
            raw_doc[:, start_col : start_col + rep_dim] = doc_values.reshape(n, rep_dim)

        raw_query.div_(scale)
        raw_doc.div_(scale)
        torch.cuda.synchronize()

    return raw_query, raw_doc, lengths


def _raw_fde_numpy(
    reps: Sequence[np.ndarray],
    config: RawBlockwiseFDEConfig,
    *,
    spherical: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    proc = prepare_reps(reps, spherical=spherical, dtype=np.float32)
    n = len(proc)
    lengths = np.asarray([max(1, int(r.shape[0])) for r in proc], dtype=np.int64)
    boundaries = np.insert(np.cumsum(lengths), 0, 0).astype(np.int64, copy=False)
    all_points = np.vstack(proc).astype(np.float32, copy=False)
    raw_query = np.zeros((n, config.raw_component_dimension), dtype=np.float32)
    raw_doc = np.zeros_like(raw_query)
    rep_dim = int(config.num_partitions) * int(config.projection_dimension)

    for rep in range(int(config.num_repetitions)):
        seed = int(config.seed) + rep
        sketches = all_points @ _simhash_matrix(dim := int(config.input_dimension), int(config.num_simhash_projections), seed)
        partitions = _partition_indices_from_sketches(sketches)
        projected = all_points @ _ams_projection_matrix(dim, int(config.projection_dimension), seed)
        sums, counts = _aggregate_projected_by_partition(
            projected, partitions, lengths, boundaries, num_partitions=int(config.num_partitions)
        )
        start_col = rep * rep_dim
        raw_query[:, start_col : start_col + rep_dim] = sums.reshape(n, rep_dim)
        doc_values = sums.copy()
        np.divide(doc_values, counts[:, :, None], out=doc_values, where=counts[:, :, None] > 0)
        _fill_empty_partitions(
            doc_values, counts, projected, partitions, lengths, boundaries, int(config.num_simhash_projections)
        )
        raw_doc[:, start_col : start_col + rep_dim] = doc_values.reshape(n, rep_dim)

    scale = np.float32(np.sqrt(float(config.num_repetitions)))
    raw_query /= scale
    raw_doc /= scale
    return raw_query, raw_doc, lengths


def build_raw_blockwise_fde_embeddings(
    reps: Sequence[np.ndarray],
    config: RawBlockwiseFDEConfig,
    *,
    spherical: bool = True,
) -> RawBlockwiseFDEEmbeddings:
    """Build blockwise FDE left/right embeddings."""
    gpu = _raw_fde_torch(reps, config, spherical=spherical)
    dim = int(config.final_component_dimension)
    seed = int(config.seed) + dim

    if gpu is not None:
        import torch

        raw_query_t, raw_doc_t, lengths = gpu
        inv = torch.as_tensor((1.0 / lengths.astype(np.float32))[:, None], dtype=torch.float32, device=raw_query_t.device)
        raw_query_t = raw_query_t * inv
        q_left_t, q_right_t = _count_sketch_torch_many(raw_query_t, dim, (seed, seed + 1))
        d_right_t, d_left_t = _count_sketch_torch_many(raw_doc_t, dim, (seed, seed + 1))
        left = torch.cat([q_left_t, d_left_t], dim=1).detach().cpu().numpy()
        right = torch.cat([d_right_t, q_right_t], dim=1).detach().cpu().numpy()
    else:
        raw_query, raw_doc, lengths = _raw_fde_numpy(reps, config, spherical=spherical)
        raw_query = raw_query * (1.0 / lengths.astype(np.float32))[:, None]
        (q_left, q_right), _ = _count_sketch_batch_many(
            raw_query, dim, (seed, seed + 1), device=config.count_sketch_device
        )
        (d_right, d_left), _ = _count_sketch_batch_many(
            raw_doc, dim, (seed, seed + 1), device=config.count_sketch_device
        )
        left = np.concatenate([q_left, d_left], axis=1).astype(np.float32, copy=False)
        right = np.concatenate([d_right, q_right], axis=1).astype(np.float32, copy=False)

    return RawBlockwiseFDEEmbeddings(left=left, right=right, lengths=lengths, config=config)
