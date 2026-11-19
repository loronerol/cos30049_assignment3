"""Phase 3: stratified 10-fold CV of every classifier on every feature set (FR8, FR9).

Feature sets: Full, FS1 (GA), FS2 (PSO), FS3 (SA), FS4 (PRISMO).
Usage:  python ml/scripts/run_experiments.py [--quick] [--no-optional]
Outputs: results/cv_folds.csv, results/cv_results.csv, results/cv_results.md,
         results/figures/cv_*.png, results/model_choice.json
"""

from __future__ import annotations

import argparse

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import load_split, log, read_json, write_json
from spamsentinel_ml.config import CV_FOLDS, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR
from spamsentinel_ml.evaluate import cross_validate, summarise_cv
from spamsentinel_ml.models import MODEL_FACTORIES, OPTIONAL_FACTORIES, model_params
from spamsentinel_ml.preprocess import xy

SET_ORDER = ["Full", "FS1", "FS2", "FS3", "FS4"]
SET_LABEL = {"Full": "Full", "FS1": "FS1 (GA)", "FS2": "FS2 (PSO)", "FS3": "FS3 (SA)", "FS4": "FS4 (PRISMO)"}


def load_sets(tag: str) -> dict[str, list[str]]:
    sel = read_json(RESULTS_DIR / f"selection{tag}.json")
    return {"Full": sel["full"]["features"], **{k: sel[k]["features"] for k in ["FS1", "FS2", "FS3", "FS4"]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-optional", action="store_true", help="skip HistGradientBoosting")
    ap.add_argument("--with-text", action="store_true", help="feature sets that include text_spam_score")
    args = ap.parse_args()
    suffix = ("_text" if args.with_text else "") + ("_quick" if args.quick else "")

    train, _ = load_split("train", args.with_text)
    sets = load_sets(suffix)
    models = list(MODEL_FACTORIES) + ([] if args.no_optional else list(OPTIONAL_FACTORIES))
    folds = 3 if args.quick else CV_FOLDS

    fold_rows, summary_rows = [], []
    for set_name in SET_ORDER:
        X, y = xy(train, sets[set_name])
        for model in models:
            fd = cross_validate(model, X, y, folds=folds)
            fd.insert(0, "feature_set", set_name)
            fd.insert(1, "model", model)
            fold_rows.append(fd)
            s = summarise_cv(fd)
            summary_rows.append({"feature_set": set_name, "model": model, "n_features": len(sets[set_name]), **s})
            log(f"{set_name:4s} {model:20s} F1m={s['f1_macro_mean']:.4f}+/-{s['f1_macro_sd']:.4f} "
                f"TPR={s['tpr_mean']:.4f} TNR={s['tnr_mean']:.4f} AUC={s['roc_auc_mean']:.4f} "
                f"fit={s['fit_ms_mean']:.0f}ms")

    folds_df = pd.concat(fold_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    folds_df.to_csv(RESULTS_DIR / f"cv_folds{suffix}.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"cv_results{suffix}.csv", index=False)
    (RESULTS_DIR / f"cv_results{suffix}.md").write_text(markdown_table(summary, folds), encoding="utf-8")

    # Model choice: best macro-F1 on FS4 among the four PRISMO classifiers (the optional
    # gradient-boosting model is reported for comparison only); report the gap to Full.
    fs4 = summary[(summary["feature_set"] == "FS4") & summary["model"].isin(list(MODEL_FACTORIES))]
    fs4 = fs4.sort_values("f1_macro_mean", ascending=False)
    best = fs4.iloc[0]
    full_same = summary[(summary["feature_set"] == "Full") & (summary["model"] == best["model"])].iloc[0]
    gap = float(full_same["f1_macro_mean"] - best["f1_macro_mean"])
    choice = {
        "model": best["model"],
        "feature_set": "FS4",
        "features": sets["FS4"],
        "cv_f1_macro_fs4": float(best["f1_macro_mean"]),
        "cv_f1_macro_full_same_model": float(full_same["f1_macro_mean"]),
        "f1_gap_full_minus_fs4": gap,
        "gap_exceeds_1_point": gap > 0.01,
        "build_time_ms_fs4": float(best["fit_ms_mean"]),
        "build_time_ms_full": float(full_same["fit_ms_mean"]),
        "params": model_params(best["model"]),
        "cv_folds": folds,
        "with_text": args.with_text,
        "text_feature_in_fs4": "text_spam_score" in sets["FS4"],
    }
    write_json(RESULTS_DIR / f"model_choice{suffix}.json", choice)
    log(f"chosen: {best['model']} on FS4 (CV macro-F1 {best['f1_macro_mean']:.4f}; Full {full_same['f1_macro_mean']:.4f})")
    if not args.quick:
        plot(summary, models, suffix)


def markdown_table(summary: pd.DataFrame, folds: int) -> str:
    lines = [
        f"# Stratified {folds}-fold cross-validation (training split)\n",
        "Mean ± standard deviation over folds. Spam is the positive class (TPR = spammer detection rate). "
        "Build time = single-threaded fit time per fold.\n",
        "| Feature set | k | Model | Accuracy | TPR | TNR | Precision | F1 (spam) | F1 (macro) | ROC-AUC | Build ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        cell = lambda m: f"{r[m + '_mean']:.4f} ± {r[m + '_sd']:.4f}"
        lines.append(
            f"| {SET_LABEL[r['feature_set']]} | {r['n_features']} | {r['model']} | {cell('accuracy')} | {cell('tpr')} | "
            f"{cell('tnr')} | {cell('precision')} | {cell('f1_spam')} | {cell('f1_macro')} | {cell('roc_auc')} | "
            f"{r['fit_ms_mean']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def plot(summary: pd.DataFrame, models: list[str], suffix: str = "") -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    palette = ["#1F2A44", "#0F766E", "#D97706", "#6D28D9", "#D50032"]
    for metric, label, fname in [
        ("f1_macro", "Macro F1-score", "cv_f1_macro.png"),
        ("tpr", "True positive rate (spam detection)", "cv_tpr.png"),
        ("fit_ms", "Model build time (ms)", "cv_build_time.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.8 / len(SET_ORDER)
        x = np.arange(len(models))
        for i, s in enumerate(SET_ORDER):
            sub = summary[summary["feature_set"] == s].set_index("model").reindex(models)
            ax.bar(x + i * width, sub[f"{metric}_mean"], width, yerr=sub[f"{metric}_sd"],
                   label=SET_LABEL[s], color=palette[i], capsize=2)
        ax.set_xticks(x + width * (len(SET_ORDER) - 1) / 2)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylabel(label)
        if metric != "fit_ms":
            lo = max(0.0, float(summary[f"{metric}_mean"].min()) - 0.05)
            ax.set_ylim(lo, 1.0)
        else:
            ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / fname.replace(".png", f"{suffix}.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
