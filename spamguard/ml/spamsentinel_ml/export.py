"""Model bundle exported for the FastAPI back end (FR10).

A bundle is a plain dict saved with joblib:
  model            fitted sklearn Pipeline (imputer + classifier)
  features         ordered input features the model expects
  feature_meta     name, category, unit, description, valid and typical ranges
  reference        per-class quartiles of each feature in the training data (FR12)
  review_band      spam-probability range labelled "needs review" (FR13)
  threshold, model_name, feature_set, metrics, versions, trained_at
"""

from __future__ import annotations

import datetime as dt
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .config import REVIEW_BAND
from .features import FEATURE_INFO

LABELS = ("likely_genuine", "needs_review", "likely_spam")


def feature_meta(features: list[str], train: pd.DataFrame) -> list[dict]:
    meta = []
    for f in features:
        info = FEATURE_INFO[f]
        col = train[f].dropna()
        if info.binary:
            valid = {"min": 0, "max": 1}
        elif info.unit == "fraction":
            valid = {"min": 0.0, "max": 1.0}
        else:
            valid = {"min": 0.0, "max": None}
        meta.append({
            "name": f,
            "category": info.category,
            "unit": info.unit,
            "description": info.description,
            "integer": info.integer,
            "binary": info.binary,
            "valid_range": valid,
            "typical_range": {"p01": float(col.quantile(0.01)), "p99": float(col.quantile(0.99))},
            "train_median": float(col.median()),
        })
    return meta


def reference_stats(train: pd.DataFrame, features: list[str]) -> dict:
    """{feature: {"genuine": {...}, "spam": {...}}} quartiles used in explanations."""
    out = {}
    for f in features:
        out[f] = {}
        for label, name in [(0, "genuine"), (1, "spam")]:
            col = train.loc[train["label"] == label, f].dropna()
            out[f][name] = {
                "p10": float(col.quantile(0.10)), "q1": float(col.quantile(0.25)),
                "median": float(col.median()), "q3": float(col.quantile(0.75)),
                "p90": float(col.quantile(0.90)), "mean": float(col.mean()),
            }
    return out


def build_bundle(model, model_name: str, feature_set: str, features: list[str], train: pd.DataFrame,
                 metrics: dict, review_band: tuple[float, float] = REVIEW_BAND) -> dict:
    return {
        "model": model,
        "model_name": model_name,
        "feature_set": feature_set,
        "features": list(features),
        "feature_meta": feature_meta(features, train),
        "reference": reference_stats(train, features),
        "review_band": list(review_band),
        "threshold": 0.5,
        "labels": list(LABELS),
        "metrics": metrics,
        "dataset": "cresci-2017 (genuine vs social + traditional spambots)",
        "train_accounts": int(len(train)),
        "versions": {
            "spamsentinel_ml": __version__,
            "sklearn": sklearn.__version__,
            "python": platform.python_version(),
        },
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: Path) -> dict:
    return joblib.load(path)


def label_from_prob(prob: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    prob = np.asarray(prob)
    return np.where(prob > hi, "likely_spam", np.where(prob < lo, "likely_genuine", "needs_review"))


def predict(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Spam probability, label and review flag for each row (columns = bundle features)."""
    X = rows[bundle["features"]].astype(float)
    prob = bundle["model"].predict_proba(X)[:, 1]
    label = label_from_prob(prob, tuple(bundle["review_band"]))
    return pd.DataFrame({"spam_score": prob, "label": label, "needs_review": label == "needs_review"},
                        index=rows.index)
