"""Shared helpers for the pipeline scripts."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def load_split(name: str, with_text: bool = False):
    """train/test features; with_text joins the L1 text encoder score as an extra feature."""
    import pandas as pd

    from spamsentinel_ml.config import PROCESSED_DIR
    from spamsentinel_ml.features import FEATURE_NAMES

    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv", index_col="user_id")
    if not with_text:
        return df, list(FEATURE_NAMES)
    scores = pd.read_csv(PROCESSED_DIR / "text_scores.csv", index_col="user_id")
    scores = scores[scores["split"] == name]
    if len(scores) != len(df) or not scores.index.isin(df.index).all():
        raise RuntimeError("text_scores.csv does not cover this split; run train_text_encoder.py")
    df["text_spam_score"] = scores["text_spam_score"].reindex(df.index)
    return df, [*FEATURE_NAMES, "text_spam_score"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
