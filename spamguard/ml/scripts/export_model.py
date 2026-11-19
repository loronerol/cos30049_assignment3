"""Phase 3b: refit the chosen classifier on the training split, test it once on
the held-out split, and export bundles for the API (FR10).

Two bundles are written so study mode can compare input forms (FR22):
  models/spamsentinel_model.joblib       chosen model on FS4 (reduced input)
  models/spamsentinel_model_full.joblib  same classifier on all features
Also: models/metrics.json, models/sample_accounts.json,
      results/test_results.json, results/test_results.md

Usage:  python ml/scripts/export_model.py [--quick]
"""

from __future__ import annotations

import argparse
import time

import _common  # noqa: F401
import numpy as np
import pandas as pd

from _common import load_split, log, read_json, write_json
from spamsentinel_ml.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, REVIEW_BAND, SEED
from spamsentinel_ml.evaluate import classification_metrics, review_band_stats
from spamsentinel_ml.export import build_bundle, label_from_prob, load_bundle, predict, save_bundle
from spamsentinel_ml.features import FEATURE_NAMES
from spamsentinel_ml.models import make_model
from spamsentinel_ml.preprocess import xy

PRISMO_TPR = 0.724  # Agrawal et al. (2018), Random Forest on FS4


def fit_and_test(model_name: str, features: list[str], train: pd.DataFrame, test: pd.DataFrame):
    X_tr, y_tr = xy(train, features)
    X_te, y_te = xy(test, features)
    model = make_model(model_name)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_ms = (time.perf_counter() - t0) * 1000
    prob = model.predict_proba(X_te)[:, 1]
    m = classification_metrics(y_te, (prob >= 0.5).astype(int), prob)
    m["fit_ms"] = fit_ms
    m["review_band"] = review_band_stats(y_te, prob, REVIEW_BAND)
    per_group = {}
    for g, idx in test.groupby("group").groups.items():
        mask = test.index.isin(idx)
        pred = (prob[mask] >= 0.5).astype(int)
        per_group[g] = {"n": int(mask.sum()), "accuracy": float((pred == y_te[mask]).mean())}
    m["accuracy_by_group"] = per_group
    return model, prob, m


def global_importance(model, features: list[str]) -> list[dict]:
    est = model.named_steps["model"]
    if not hasattr(est, "feature_importances_"):
        return []
    imp = pd.Series(est.feature_importances_, index=features).sort_values(ascending=False)
    return [{"feature": f, "importance": float(v)} for f, v in imp.items()]


def pick_samples(test: pd.DataFrame, prob: np.ndarray, n_each: int = 3) -> list[dict]:
    """Sample accounts for the UI: confident spam/genuine, uncertain, and model errors (RQ4)."""
    rng = np.random.default_rng(SEED)
    df = test.copy()
    df["prob"] = prob
    df["pred"] = (prob >= 0.5).astype(int)
    lo, hi = REVIEW_BAND
    buckets = {
        "confident_spam": df[(df["prob"] > hi) & (df["label"] == 1)],
        "confident_genuine": df[(df["prob"] < lo) & (df["label"] == 0)],
        "uncertain": df[(df["prob"] >= lo) & (df["prob"] <= hi)],
        "model_error": df[df["pred"] != df["label"]],
    }
    chosen, seen = [], set()
    for kind, sub in buckets.items():
        sub = sub[~sub.index.isin(seen)]
        take = sub.iloc[rng.permutation(len(sub))[:n_each]]
        for uid, row in take.iterrows():
            seen.add(uid)
            chosen.append({
                "kind": kind,
                "true_label": "spam" if row["label"] == 1 else "genuine",
                "group": row["group"],
                "model_spam_score": float(row["prob"]),
                "features": {f: (None if pd.isna(row[f]) else float(row[f])) for f in FEATURE_NAMES},
            })
    for i, s in enumerate(chosen, 1):
        s["sample_id"] = f"S{i:02d}"
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--with-text", action="store_true", help="export the FS4 + text_spam_score variant")
    args = ap.parse_args()
    suffix = ("_text" if args.with_text else "") + ("_quick" if args.quick else "")

    choice = read_json(RESULTS_DIR / f"model_choice{suffix}.json")
    train, all_names = load_split("train", args.with_text)
    test, _ = load_split("test", args.with_text)
    name, fs4 = choice["model"], choice["features"]

    reduced, prob_r, m_r = fit_and_test(name, fs4, train, test)
    full, prob_f, m_f = fit_and_test(name, all_names, train, test)
    targets = {
        "f1_macro_at_least_0.85": m_r["f1_macro"] >= 0.85,
        "tpr_above_prismo_0.724": m_r["tpr"] > PRISMO_TPR,
        "build_time_reduced_vs_full_ms": [m_r["fit_ms"], m_f["fit_ms"]],
    }
    log(f"TEST {name} FS4 (k={len(fs4)}): F1m={m_r['f1_macro']:.4f} TPR={m_r['tpr']:.4f} TNR={m_r['tnr']:.4f} "
        f"AUC={m_r['roc_auc']:.4f} | Full: F1m={m_f['f1_macro']:.4f}")
    log(f"review band {REVIEW_BAND}: {m_r['review_band']['share_in_band']:.1%} of test accounts; "
        f"caught {m_r['review_band']['errors_caught_by_band']}/{m_r['review_band']['errors_total']} errors")

    metrics = {
        "model": name,
        "cv": {k: v for k, v in choice.items() if k.startswith("cv_") or k.startswith("f1_gap") or k.startswith("build")},
        "test_reduced": m_r,
        "test_full": m_f,
        "targets": targets,
        "global_importance": global_importance(reduced, fs4),
        "test_accounts": int(len(test)),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_r = MODELS_DIR / f"spamsentinel_model{suffix}.joblib"
    out_f = MODELS_DIR / f"spamsentinel_model_full{suffix}.joblib"
    save_bundle(build_bundle(reduced, name, "FS4", fs4, train, m_r), out_r)
    save_bundle(build_bundle(full, name, "Full", all_names, train, m_f), out_f)
    write_json(MODELS_DIR / f"metrics{suffix}.json", metrics)
    if not args.with_text:  # the UI samples come from the numeric model
        write_json(MODELS_DIR / f"sample_accounts{suffix}.json", pick_samples(test, prob_r))
    write_json(RESULTS_DIR / f"test_results{suffix}.json", metrics)
    (RESULTS_DIR / f"test_results{suffix}.md").write_text(markdown(metrics, fs4, len(all_names)), encoding="utf-8")

    # Round-trip check + NFR3 timing: reload and score 1,000 accounts.
    bundle = load_bundle(out_r)
    batch = test.sample(1000, replace=len(test) < 1000, random_state=SEED)
    t0 = time.perf_counter()
    res = predict(bundle, batch)
    secs = time.perf_counter() - t0
    assert np.allclose(res["spam_score"].to_numpy(), bundle["model"].predict_proba(batch[fs4])[:, 1])
    log(f"reloaded bundle scored 1,000 accounts in {secs:.2f}s (NFR3 budget 30s); "
        f"labels: {res['label'].value_counts().to_dict()}")
    log(f"exported {out_r.name} and {out_f.name}")


def markdown(m: dict, fs4: list[str], n_all: int) -> str:
    rows = []
    for key, label in [("test_reduced", f"FS4 ({len(fs4)} features)"), ("test_full", f"Full ({n_all} features)")]:
        r = m[key]
        rows.append(f"| {label} | {r['accuracy']:.4f} | {r['tpr']:.4f} | {r['tnr']:.4f} | {r['precision']:.4f} | "
                    f"{r['f1_spam']:.4f} | {r['f1_macro']:.4f} | {r['roc_auc']:.4f} | {r['fit_ms']:.0f} |")
    rb = m["test_reduced"]["review_band"]
    grp = "\n".join(f"| {g} | {v['n']} | {v['accuracy']:.4f} |" for g, v in m["test_reduced"]["accuracy_by_group"].items())
    return (
        f"# Held-out test results — {m['model']}\n\n"
        "| Feature set | Accuracy | TPR | TNR | Precision | F1 (spam) | F1 (macro) | ROC-AUC | Build ms |\n"
        "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        f"Targets: macro-F1 ≥ 0.85 → **{m['targets']['f1_macro_at_least_0.85']}**; "
        f"TPR > PRISMO's 72.4% → **{m['targets']['tpr_above_prismo_0.724']}**\n\n"
        f"Review band {rb['band_low']}–{rb['band_high']}: {rb['share_in_band']:.1%} of test accounts "
        f"({rb['n_in_band']}); accuracy outside band {rb['accuracy_outside_band']:.4f}; "
        f"{rb['errors_caught_by_band']} of {rb['errors_total']} errors fall inside the band.\n\n"
        "## Accuracy by account group (FS4 model)\n\n| Group | n | Accuracy |\n|---|---|---|\n" + grp + "\n"
    )


if __name__ == "__main__":
    main()
