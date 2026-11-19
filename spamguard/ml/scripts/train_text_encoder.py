"""L1: train the text encoder with 5-fold out-of-fold (OOF) scoring.

* training accounts: text_spam_score from the fold model that did not see them
* test accounts:     text_spam_score from a model trained on all training accounts
The final model is saved to models/text_encoder/ for the API.

Usage:  python ml/scripts/train_text_encoder.py [--smoke N] [--epochs E]
Outputs: data/processed/text_scores.csv, results/text_encoder.json, models/text_encoder/
"""

from __future__ import annotations

import argparse
import dataclasses

import _common  # noqa: F401
import numpy as np
import pandas as pd
import torch

from _common import log, read_json, write_json
from spamsentinel_ml.config import INTERIM_DIR, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from spamsentinel_ml.evaluate import classification_metrics
from spamsentinel_ml.text.encoder import EncoderConfig, oof_plan, predict_proba, save_encoder, train_encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="use only N train / N/4 test accounts")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = EncoderConfig()
    if args.epochs:
        cfg = dataclasses.replace(cfg, epochs=args.epochs)

    df = pd.read_parquet(PROCESSED_DIR / "text_accounts.parquet")
    df.index = df.index.astype(str)
    folds = pd.read_csv(PROCESSED_DIR / "text_folds.csv", index_col="user_id")["fold"]
    folds.index = folds.index.astype(str)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    if args.smoke:
        rng = np.random.default_rng(SEED)
        train = train.iloc[rng.permutation(len(train))[: args.smoke]]
        test = test.iloc[rng.permutation(len(test))[: max(args.smoke // 4, 20)]]
    fold_of = {u: int(folds[u]) for u in train.index}

    scores = {}
    fold_info = {}
    tag = "_smoke" if args.smoke else ""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    for k, part in oof_plan(fold_of).items():
        assert not set(part["train"]) & set(part["score"])
        # Each finished fold is saved, so a crash only costs the fold in progress.
        fold_file = INTERIM_DIR / f"text_oof_fold{k}{tag}.csv"
        info_file = INTERIM_DIR / f"text_oof_fold{k}{tag}.json"
        if fold_file.exists() and info_file.exists():
            done = pd.read_csv(fold_file, index_col="user_id")["text_spam_score"]
            done.index = done.index.astype(str)
            if set(done.index) == set(part["score"]):
                scores.update(done.to_dict())
                fold_info[k] = read_json(info_file)
                log(f"fold {k}: loaded saved scores")
                continue
        log(f"fold {k}: train {len(part['train'])}, score {len(part['score'])}")
        tr = train.loc[part["train"]]
        model, tok, info = train_encoder(tr["text"].tolist(), tr["label"].to_numpy(), cfg, log)
        prob = predict_proba(model, tok, train.loc[part["score"], "text"].tolist(), cfg)
        scores.update(dict(zip(part["score"], prob)))
        fold_info[k] = info
        pd.DataFrame({"text_spam_score": prob}, index=pd.Index(part["score"], name="user_id")).to_csv(fold_file)
        write_json(info_file, info)
        del model
        torch.cuda.empty_cache()

    log("final model on all training accounts")
    model, tok, final_info = train_encoder(train["text"].tolist(), train["label"].to_numpy(), cfg, log)
    test_prob = predict_proba(model, tok, test["text"].tolist(), cfg)

    oof = pd.Series(scores).reindex(train.index)
    out = pd.concat([
        pd.DataFrame({"split": "train", "fold": [fold_of[u] for u in train.index], "text_spam_score": oof.values},
                     index=train.index),
        pd.DataFrame({"split": "test", "fold": -1, "text_spam_score": test_prob}, index=test.index),
    ])
    out.index.name = "user_id"
    m_oof = classification_metrics(train["label"], (oof.values >= 0.5).astype(int), oof.values)
    m_test = classification_metrics(test["label"], (test_prob >= 0.5).astype(int), test_prob)
    eng = test["has_english_text"].to_numpy()
    m_test_en = classification_metrics(test["label"][eng], (test_prob[eng] >= 0.5).astype(int), test_prob[eng])
    log(f"OOF (train) F1m={m_oof['f1_macro']:.4f} AUC={m_oof['roc_auc']:.4f} | "
        f"TEST F1m={m_test['f1_macro']:.4f} AUC={m_test['roc_auc']:.4f} TPR={m_test['tpr']:.4f}")

    suffix = "_smoke" if args.smoke else ""
    out.to_csv(PROCESSED_DIR / f"text_scores{suffix}.csv")
    report = {"config": cfg.to_dict(), "oof_metrics": m_oof, "test_metrics": m_test,
              "test_metrics_english_only": m_test_en, "folds": fold_info, "final": final_info,
              "n_train": len(train), "n_test": len(test)}
    write_json(RESULTS_DIR / f"text_encoder{suffix}.json", report)
    if not args.smoke:
        save_encoder(model, tok, MODELS_DIR / "text_encoder", {**final_info, "config": cfg.to_dict(),
                                                               "test_metrics": m_test})
    log("done")


if __name__ == "__main__":
    main()
