"""Metrics, stratified k-fold cross-validation and held-out testing (FR8, FR9)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .config import CV_FOLDS, SEED
from .models import make_model

METRICS = ["accuracy", "tpr", "tnr", "precision", "f1_spam", "f1_macro", "roc_auc", "fit_ms"]


def classification_metrics(y_true, y_pred, y_prob=None) -> dict[str, float]:
    """Spam (label 1) is the positive class: TPR = spammer detection rate."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "tpr": tp / (tp + fn) if tp + fn else 0.0,
        "tnr": tn / (tn + fp) if tn + fp else 0.0,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1_spam": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": np.nan,
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    return {k: float(v) for k, v in out.items()}


def _fit_score(model_name: str, X: pd.DataFrame, y: np.ndarray, train_idx, val_idx, fold: int) -> dict:
    model = make_model(model_name)
    t0 = time.perf_counter()
    model.fit(X.iloc[train_idx], y[train_idx])
    fit_ms = (time.perf_counter() - t0) * 1000
    prob = model.predict_proba(X.iloc[val_idx])[:, 1]
    pred = (prob >= 0.5).astype(int)
    m = classification_metrics(y[val_idx], pred, prob)
    m.update(fold=fold, fit_ms=fit_ms)
    return m


def cross_validate(
    model_name: str,
    X: pd.DataFrame,
    y,
    folds: int = CV_FOLDS,
    seed: int = SEED,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Stratified k-fold CV; returns one row of metrics per fold.

    Folds run in parallel; each model is single-threaded, so fit_ms is a
    comparable per-model build time (PRISMO's "model build time").
    """
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score)(model_name, X, y, tr, va, i) for i, (tr, va) in enumerate(skf.split(X, y))
    )
    return pd.DataFrame(rows)


def summarise_cv(fold_df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for m in METRICS:
        out[f"{m}_mean"] = float(fold_df[m].mean())
        out[f"{m}_sd"] = float(fold_df[m].std(ddof=1))
    return out


def review_band_stats(y_true, prob, band: tuple[float, float]) -> dict[str, float]:
    """How many predictions fall in the review band, and accuracy outside it (FR13)."""
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    lo, hi = band
    in_band = (prob >= lo) & (prob <= hi)
    outside = ~in_band
    pred = (prob >= 0.5).astype(int)
    return {
        "band_low": lo,
        "band_high": hi,
        "share_in_band": float(in_band.mean()),
        "n_in_band": int(in_band.sum()),
        "accuracy_in_band": float((pred[in_band] == y_true[in_band]).mean()) if in_band.any() else float("nan"),
        "accuracy_outside_band": float((pred[outside] == y_true[outside]).mean()) if outside.any() else float("nan"),
        "errors_caught_by_band": int(((pred != y_true) & in_band).sum()),
        "errors_total": int((pred != y_true).sum()),
    }
