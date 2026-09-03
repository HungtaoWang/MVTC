# Multi-Vector Text Clustering

**Chamfer-KMeans** and **Chamfer-KMeans+** for multi-vector document clustering.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# encode + ToMe pool
python prepare_embeddings.py --dataset trec --device cuda

# cluster (needs pooled .pkl + labels.npy)
export POOLED_ROOT=./data/pooled EMBEDDINGS_ROOT=./data/embeddings
python run_cluster.py --dataset trec --method optimized --seed 1234 --fde-dim 256
```

Public API: `from src import run_chamfer_kmeans, run_chamfer_kmeans_plus`

## Layout

| Path | Role |
|------|------|
| `src/` | Chamfer-KMeans, Chamfer-KMeans+, FDE |
| `dataloader/` | Dataset loaders |
| `utils/` | Encoding, pooling, metrics, I/O |
| `configs/datasets.json` | Per-dataset encoder / m / source URL |
| `configs/protocol.json` | Shared algorithm settings |

## Data

Loaders live in `dataloader/datasets.py`. Raw files cache under `data/raw/` (created on first download). Sources and expected N/K are in `configs/datasets.json`.

| Dataset | Loader | Raw cache |
|---------|--------|-----------|
| trec | auto-download | `data/raw/trec/` |
| reuters52 | Hendrycks r52-*.txt | `data/raw/reuters52/` |
| 20news | sklearn (N=18211) | sklearn cache |
| mag, mesh | SciDocs | `data/raw/scidocs/` or `SCIDOCS_ROOT` |
| locov1, dbpedia | HuggingFace | HF cache |
| yahoo | Renovamen CSV | `data/raw/yahoo_answers_csv/` |
