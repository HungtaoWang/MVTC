# Multi-Vector Text Clustering

**Chamfer-KMeans** and **Chamfer-KMeans+** for multi-vector document clustering.

## Install

```bash
pip install -r requirements.txt
```

Encoders used by this repo:
- **Jina v4** (`jinaai/jina-embeddings-v4`, `trust_remote_code=True`) — for `trec`, `locov1`
- **MPNet** (`sentence-transformers/all-mpnet-base-v2`, via HF `AutoModel`) — for the other 6 datasets

The models download automatically on first encode run via HuggingFace.

## Data

This repo reads **canonical texts** from `data/cache/{dataset}/records.jsonl`
(one JSON per line: `{"id", "text", "label"}`). These are the exact strings
fed to the encoder to reproduce the paper's embeddings.

### Download canonical texts

Download the canonical archive: **https://www.dropbox.com/scl/fi/imd3t2eres9mksmre86j8/data.zip?rlkey=1oyub8mn583t00volsjzh9msa&st=wg707qdi&dl=0**

Extract under the repo root so that you have:

```
data/cache/trec/records.jsonl
data/cache/trec/meta.json
data/cache/reuters52/records.jsonl
...
```

## Usage

```bash
# encode canonical texts -> multivector -> ToMe pool -> pooled
python prepare_embeddings.py --dataset trec --device cuda

# or skip encoding and pool a precomputed multivector
python prepare_embeddings.py --dataset trec --from-multivector data/embeddings/trec/multi/multivector_128.pkl

# cluster (needs pooled .pkl + labels.npy under data/)
python run_cluster.py --dataset trec --method optimized --seed 1234 --fde-dim 256 --device cuda
```

Public API: `from src import run_chamfer_kmeans, run_chamfer_kmeans_plus`

## Layout

| Path | Role |
|------|------|
| `src/` | Chamfer-KMeans, Chamfer-KMeans+, FDE |
| `dataloader/` | Canonical `records.jsonl` loader |
| `utils/` | Encoding (Jina/MPNet multivector), ToMe pooling, metrics, I/O |
| `configs/datasets.json` | Per-dataset encoder / m / max_length / batch_size |
| `configs/protocol.json` | Shared algorithm settings |
| `data/cache/` | Canonical texts (download — not tracked) |
| `data/embeddings/` | Multivector + labels (generated) |
| `data/pooled/` | Pooled embeddings (generated) |

## Pipeline

```
data/cache/{ds}/records.jsonl
    -> prepare_embeddings.py (encode + ToMe pool)
    -> data/embeddings/{ds}/{variant}/multivector_{dim}.pkl + labels.npy
    -> data/pooled/{ds}/{variant}/pooled_m{m}_tome_bipartite[_{variant}].pkl
    -> run_cluster.py
    -> NMI / FP / ACC
```

Encoder settings per dataset are in `configs/datasets.json` (`encoder`, `max_length`,
`batch_size`, `m`). Variant naming: `multi` (jina, dim=128) or `mpnet` (sbert, dim=768).
