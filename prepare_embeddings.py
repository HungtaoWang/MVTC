#!/usr/bin/env python3
"""Encode raw texts then apply ToMe pooling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dataloader import load_dataset
from utils.encode import encode_jina, encode_sbert, save_multivector
from utils.io import EMBEDDINGS, pooled_path, save_pooled, variant_for_rep
from utils.pool import pool_corpus


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "datasets.json")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())[args.dataset]
    texts, labels, meta = load_dataset(args.dataset)
    if cfg["encoder"] == "jina":
        reps = encode_jina(texts, device=args.device)
        sub = "multi"
    else:
        reps = encode_sbert(texts, device=args.device)
        sub = "mpnet"

    raw_dir = EMBEDDINGS / args.dataset / sub
    save_multivector(raw_dir / "multivector.pkl", reps)
    np.save(raw_dir / "labels.npy", labels)

    m = int(cfg["m"])
    pooled = pool_corpus(reps, m)
    out = pooled_path(args.dataset, variant_for_rep(cfg["representation"]), m)
    save_pooled(out, pooled, meta)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
