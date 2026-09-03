"""Dataset loaders (official HuggingFace / GitHub sources)."""

from __future__ import annotations

import csv
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

HENDRYCKS_CAT = "https://github.com/hendrycks/error-detection/tree/master/NLP/Categorization"
HENDRYCKS_DATA = "https://raw.githubusercontent.com/hendrycks/error-detection/master/NLP/Categorization/data"

SOURCES: dict[str, dict[str, Any]] = {
    "trec": {
        "name": "TREC-50",
        "urls": [
            "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
            "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
        ],
    },
    "reuters52": {
        "name": "Reuters-52",
        "repo": HENDRYCKS_CAT,
        "files": ["r52-train.txt", "r52-test.txt"],
    },
    "20news": {
        "name": "20 Newsgroups",
        "sklearn": "20newsgroups",
        "related_repo": HENDRYCKS_CAT,
        "related_files": ["20ng-train.txt", "20ng-test.txt"],
        "notes": "sklearn fetch_20newsgroups + min 21 chars.",
    },
    "mag": {"name": "SciDocs-MAG", "repo": "https://github.com/allenai/scidocs"},
    "mesh": {"name": "SciDocs-MESH", "repo": "https://github.com/allenai/scidocs"},
    "locov1": {"name": "LoCoV1", "hf": "hazyresearch/LoCoV1-Documents"},
    "dbpedia": {"name": "DBpedia", "hf": "fancyzhx/dbpedia_14"},
    "yahoo": {"name": "Yahoo Answers", "repo": "https://github.com/Renovamen/Text-Classification"},
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return
    urllib.request.urlretrieve(url, dest)


def _download_hendrycks(filename: str, dest: Path) -> None:
    _download(f"{HENDRYCKS_DATA}/{filename}", dest)


def _avg_tokens(texts: List[str]) -> float:
    if not texts:
        return 0.0
    return float(sum(len(t.split()) for t in texts) / len(texts))


def _meta(dataset: str, texts: List[str], labels: List[Any]) -> dict:
    y = np.asarray(labels)
    return {
        "dataset": dataset,
        "n": len(texts),
        "k": len(np.unique(y)),
        "avg_tokens": round(_avg_tokens(texts), 1),
    }


def load_trec() -> Tuple[List[str], np.ndarray, dict]:
    d = RAW / "trec"
    _download(SOURCES["trec"]["urls"][0], d / "train_5500.label")
    _download(SOURCES["trec"]["urls"][1], d / "TREC_10.label")
    texts, labels = [], []
    pat = re.compile(r"^(\w+):(\w+)\s+(.*)$")
    for p in (d / "train_5500.label", d / "TREC_10.label"):
        for line in p.read_text(encoding="latin-1").splitlines():
            m = pat.match(line.strip())
            if m:
                labels.append(m.group(2).lower())
                texts.append(m.group(3))
    classes = sorted(set(labels))
    y = np.asarray([classes.index(c) for c in labels], dtype=np.int64)
    meta = _meta("TREC-50", texts, labels)
    meta["k"] = len(classes)
    return texts, y, meta


def load_hendrycks_r52(data_dir: Path | None = None) -> Tuple[List[str], np.ndarray, dict]:
    """Reuters-52 from Hendrycks error-detection (tab-separated label\\ttext)."""
    d = data_dir or RAW / "reuters52"
    for fn in SOURCES["reuters52"]["files"]:
        _download_hendrycks(fn, d / fn)
    texts, labels = [], []
    for fn in SOURCES["reuters52"]["files"]:
        for line in (d / fn).read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            lab, text = line.split("\t", 1)
            labels.append(lab.strip())
            texts.append(text.strip())
    classes = sorted(set(labels))
    y = np.asarray([classes.index(c) for c in labels], dtype=np.int64)
    return texts, y, _meta("Reuters-52", texts, labels)


def load_20news() -> Tuple[List[str], np.ndarray, dict]:
    """20 Newsgroups.
    1. sklearn.datasets.fetch_20newsgroups(subset='all', remove headers/footers/quotes) → 18,846 docs
    2. Drop documents with len(text.strip()) < 21 characters → 18,211 docs
    """
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts, labels = [], []
    for text, lab in zip(data.data, data.target):
        t = text.strip()
        if len(t) >= 21:
            texts.append(t)
            labels.append(int(lab))
    y = np.asarray(labels, dtype=np.int64)
    return texts, y, _meta("20 Newsgroups", texts, labels)


def load_hf_text_label(name: str, *, text_key: str = "text", label_key: str = "label", split: str = "train") -> Tuple[List[str], np.ndarray, dict]:
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    texts = [str(x[text_key]) for x in ds]
    labels = [int(x[label_key]) for x in ds]
    classes = sorted(set(labels))
    remap = {c: i for i, c in enumerate(classes)}
    y = np.asarray([remap[v] for v in labels], dtype=np.int64)
    return texts, y, _meta(name, texts, labels)


def load_scidocs(label_type: str, scidocs_dir: Path | None = None) -> Tuple[List[str], np.ndarray, dict]:
    root = scidocs_dir or Path(os.environ.get("SCIDOCS_ROOT", str(RAW / "scidocs")))
    meta_path = root / "paper_metadata_mag_mesh.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"SciDocs metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pid_text = {}
    for pid, obj in meta.items():
        if isinstance(obj, dict):
            title = (obj.get("title") or "").strip()
            abstract = (obj.get("abstract") or "").strip()
            t = f"{title} {abstract}".strip()
            if pid and t:
                pid_text[pid] = t
    texts, labels = [], []
    for split in ("train", "val", "test"):
        p = root / label_type / f"{split}.csv"
        if not p.is_file():
            continue
        with p.open(encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0] in pid_text:
                    texts.append(pid_text[row[0]])
                    labels.append(int(row[1]))
    y = np.asarray(labels, dtype=np.int64)
    return texts, y, _meta(f"SciDocs-{label_type.upper()}", texts, labels)


def load_locov1() -> Tuple[List[str], np.ndarray, dict]:
    from datasets import load_dataset

    ds = load_dataset(SOURCES["locov1"]["hf"], split="test")
    names = sorted(set(ds["dataset"]))
    texts, labels = [], []
    for i, name in enumerate(names):
        part = ds.filter(lambda ex, n=name: ex["dataset"] == n)
        docs = [ex["passage"] for ex in part if ex.get("passage")]
        texts.extend(docs)
        labels.extend([i] * len(docs))
    y = np.asarray(labels, dtype=np.int64)
    return texts, y, _meta("LoCoV1", texts, labels)


def load_dbpedia() -> Tuple[List[str], np.ndarray, dict]:
    return load_hf_text_label("fancyzhx/dbpedia_14", text_key="content", label_key="label")


def load_renovamen_csv(name: str, csv_dir: Path) -> Tuple[List[str], np.ndarray, dict]:
    texts, labels = [], []
    for split in ("train.csv", "test.csv"):
        p = csv_dir / split
        if not p.is_file():
            continue
        with p.open(encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                try:
                    lab = int(row[0])
                except ValueError:
                    continue
                text = row[1].strip() if name != "yahoo" else " ".join(x.strip() for x in row[1:4] if x.strip())
                if len(text) >= 5:
                    texts.append(text)
                    labels.append(lab)
    classes = sorted(set(labels))
    remap = {c: i for i, c in enumerate(classes)}
    y = np.asarray([remap[v] for v in labels], dtype=np.int64)
    return texts, y, _meta(name, texts, labels)


def load_yahoo() -> Tuple[List[str], np.ndarray, dict]:
    csv_dir = RAW / "yahoo_answers_csv"
    if not (csv_dir / "train.csv").is_file():
        raise FileNotFoundError(f"Yahoo CSV not found under {csv_dir}")
    return load_renovamen_csv("yahoo", csv_dir)


_LOADERS: dict[str, Callable[[], Tuple[List[str], np.ndarray, dict]]] = {
    "trec": load_trec,
    "reuters52": load_hendrycks_r52,
    "20news": load_20news,
    "mag": lambda: load_scidocs("mag"),
    "mesh": lambda: load_scidocs("mesh"),
    "locov1": load_locov1,
    "dbpedia": load_dbpedia,
    "yahoo": load_yahoo,
}


def load_dataset(name: str) -> Tuple[List[str], np.ndarray, dict]:
    key = name.strip().lower()
    if key not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r}")
    return _LOADERS[key]()
