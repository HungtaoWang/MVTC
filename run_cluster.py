#!/usr/bin/env python3
"""Cluster pre-pooled multivectors with paper configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chamfer_kmeans import run_chamfer_kmeans
from src.chamfer_kmeans_plus import ChamferKMeansPlusConfig, run_chamfer_kmeans_plus
from utils.io import labels_path, load_pooled, pooled_path, variant_for_rep
from utils.metrics import evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", choices=["naive", "optimized"], required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fde-dim", type=int, default=256)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "datasets.json")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())[args.dataset]
    var = variant_for_rep(cfg["representation"])
    m = int(cfg["m"])
    pooled = pooled_path(args.dataset, var, m)
    reps = load_pooled(pooled)
    y = np.load(labels_path(args.dataset, var)).tolist()
    k = len(set(y))

    if args.method == "naive":
        pred = run_chamfer_kmeans(reps, k, n_center_rows=m, random_state=args.seed, device=args.device)
    else:
        ocfg = ChamferKMeansPlusConfig(
            fde_dim=args.fde_dim,
            random_state=args.seed,
            device=args.device,
        )
        pred = run_chamfer_kmeans_plus(reps, k, config=ocfg)

    met = evaluate(y, pred.tolist())
    print(json.dumps(met, indent=2))


if __name__ == "__main__":
    main()
