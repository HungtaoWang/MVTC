#!/usr/bin/env python3
"""Encode canonical texts then apply ToMe pooling, or pool a precomputed multivector.

Usage:
  # full pipeline: canonical -> encode -> multivector -> pool -> pooled
  python prepare_embeddings.py --dataset trec --device cuda

  # skip encoding: load an existing multivector pkl -> pool -> pooled
  python prepare_embeddings.py --dataset trec --from-multivector path/to/multivector_128.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dataloader import load_dataset
from utils.encode import encode_jina, encode_sbert, load_multivector, save_multivector
from utils.io import dim_for_rep, multivector_path, pooled_path, save_pooled, variant_for_rep
from utils.pool import pool_corpus


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "datasets.json")
    p.add_argument(
        "--from-multivector",
        type=Path,
        default=None,
        help="skip encoding; pool this precomputed multivector pkl instead",
    )
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())[args.dataset]
    variant = variant_for_rep(cfg["representation"])
    dim = dim_for_rep(cfg["representation"])
    m = int(cfg["m"])

    if args.from_multivector is not None:
        print(f"[skip encode] loading {args.from_multivector}")
        reps = load_multivector(args.from_multivector)
        texts, labels, meta = load_dataset(args.dataset)
    else:
        texts, labels, meta = load_dataset(args.dataset)
        if cfg["encoder"] == "jina":
            max_length = int(cfg.get("max_length", 512))
            batch_size = int(cfg.get("batch_size", 64))
            reps = encode_jina(texts, max_length=max_length, batch_size=batch_size, device=args.device)
        else:
            reps = encode_sbert(texts, batch_size=int(cfg.get("batch_size", 16)), device=args.device)
        mv_out = multivector_path(args.dataset, variant, dim)
        save_multivector(mv_out, reps)
        np.save(mv_out.parent / "labels.npy", labels)
        print(f"[multivector] {mv_out} ({len(reps)} docs)")

    pooled = pool_corpus(reps, m)
    out = pooled_path(args.dataset, variant, m)
    save_pooled(out, pooled, meta)
    print(f"[pooled] {out}")


if __name__ == "__main__":
    main()
