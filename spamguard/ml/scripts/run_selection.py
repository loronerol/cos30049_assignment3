"""Phase 2: PRISMO two-layer feature subset selection on the training split.

Layer 1: GA -> FS1, PSO -> FS2, SA -> FS3 (one shared fitness with fixed folds).
Layer 2: priority combination -> FS4.

Usage:  python ml/scripts/run_selection.py [--quick]
Output: results/selection.json, results/figures/selection_*.png
"""

from __future__ import annotations

import argparse

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import load_split, log, write_json
from spamsentinel_ml.config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, SEED, SelectionConfig, quick_selection_config
from spamsentinel_ml.features import FEATURE_INFO, FEATURE_NAMES
from spamsentinel_ml.preprocess import xy
from spamsentinel_ml.selection import FitnessEvaluator, priority_combine, run_ga, run_pso, run_sa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny budgets for a smoke test")
    ap.add_argument("--with-text", action="store_true", help="add the L1 text_spam_score feature")
    args = ap.parse_args()
    cfg = quick_selection_config() if args.quick else SelectionConfig()

    train, names = load_split("train", args.with_text)
    X, y = xy(train, names)
    # The fitness RF cannot take NaN in older sklearn builds; impute with training medians.
    X = X.fillna(X.median())
    evaluator = FitnessEvaluator(X.to_numpy(), y, cfg.fitness, seed=SEED)
    n = len(names)
    full_fit, full_f1 = evaluator.score(np.ones(n, dtype=bool))
    log(f"full feature set: macro-F1={full_f1:.4f} fitness={full_fit:.4f} (N={n})")

    results = {
        "GA": run_ga(evaluator.score, n, cfg.ga, cfg.seed_ga, log),
        "PSO": run_pso(evaluator.score, n, cfg.pso, cfg.seed_pso, log),
        "SA": run_sa(evaluator.score, n, cfg.sa, cfg.seed_sa, log),
    }
    for name, r in results.items():
        log(f"{name}: k={r.n_selected} macro-F1={r.macro_f1:.4f} fitness={r.fitness:.4f} "
            f"evals={r.n_evals} {r.seconds:.0f}s")

    combo = priority_combine({k: r.mask for k, r in results.items()}, names)
    fs4_fit, fs4_f1 = evaluator.score(combo["fs4_mask"])
    log(f"FS4: k={len(combo['fs4'])} macro-F1={fs4_f1:.4f} (priority 1: {len(combo['priority_1'])}, "
        f"priority 2: {len(combo['priority_2'])})")

    out = {
        "quick": args.quick,
        "n_features": n,
        "train_accounts": int(len(train)),
        "parameters": cfg.to_dict(),
        "fitness_seed": SEED,
        "fitness_definition": "mean stratified k-fold macro-F1 of a Random Forest on the subset "
                              "minus alpha * |S| / N (training split only)",
        "with_text": args.with_text,
        "full": {"features": names, "n_selected": n, "fitness": full_fit, "macro_f1": full_f1},
        "FS1": results["GA"].to_dict(names),
        "FS2": results["PSO"].to_dict(names),
        "FS3": results["SA"].to_dict(names),
        "FS4": {
            "algorithm": "PRISMO priority combination",
            "features": combo["fs4"],
            "n_selected": len(combo["fs4"]),
            "priority_1": combo["priority_1"],
            "priority_2": combo["priority_2"],
            "discarded_single_vote": combo["discarded_single_vote"],
            "fitness": fs4_fit,
            "macro_f1": fs4_f1,
        },
        "votes": combo["votes"],
        "fitness_evaluations": evaluator.n_evals,
        "fitness_cache_hits": evaluator.cache_hits,
    }
    tag = ("_text" if args.with_text else "") + ("_quick" if args.quick else "")
    write_json(RESULTS_DIR / f"selection{tag}.json", out)
    if not args.quick:
        plot(results, combo, tag)
    log("selection written")


def plot(results, combo, tag: str = "") -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, r in results.items():
        xs = np.linspace(0, 1, len(r.history))
        ax.plot(xs, r.history, label=f"{name} (k={r.n_selected})")
    ax.set_xlabel("Search progress (fraction of budget)")
    ax.set_ylabel("Best fitness so far")
    ax.set_title("Layer 1 feature selection convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"selection_convergence{tag}.png", dpi=150)
    plt.close(fig)

    votes = pd.Series(combo["votes"]).sort_values(ascending=True)
    votes = votes[votes > 0]
    colours = {"content": "#1F2A44", "behaviour": "#0F766E", "profile": "#D97706"}
    fig, ax = plt.subplots(figsize=(7, max(4, 0.22 * len(votes))))
    ax.barh(votes.index, votes.values, color=[colours[FEATURE_INFO[f].category] for f in votes.index])
    ax.axvline(1.5, color="#D50032", ls="--", lw=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Number of algorithms selecting the feature (FS4 keeps >= 2)")
    ax.set_title("PRISMO Layer 2 votes (colour = category)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"selection_votes{tag}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
