"""L2: QLoRA fine-tune Qwen2.5-3B-Instruct to label accounts, and score the test split.

Also scores the test split with the same model *before* fine-tuning (local
zero-shot), so the report can show what fine-tuning adds.

Usage:  python ml/scripts/train_llm_qlora.py [--smoke N] [--train-limit N] [--skip-zero-shot]
Outputs: models/llm_lora/ (adapters), data/processed/llm_scores.csv, results/llm_qlora.json
"""

from __future__ import annotations

import argparse
import dataclasses

import _common  # noqa: F401
import numpy as np
import pandas as pd
import torch

from _common import log, read_json, write_json
from spamsentinel_ml.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from spamsentinel_ml.evaluate import classification_metrics
from spamsentinel_ml.text.llm_classifier import LLMConfig, account_prompt, load_model, score, train_qlora


def build_prompts(df: pd.DataFrame, feats: pd.DataFrame, cfg: LLMConfig) -> list[str]:
    rows = feats.loc[df.index].to_dict(orient="index")
    return [account_prompt(rows[u], d, list(t), cfg) for u, d, t in zip(df.index, df["description"], df["tweets"])]


def metrics(y, prob, english_mask):
    m = classification_metrics(y, (prob >= 0.5).astype(int), prob)
    m_en = classification_metrics(y[english_mask], (prob[english_mask] >= 0.5).astype(int), prob[english_mask])
    return m, m_en


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--skip-zero-shot", action="store_true")
    args = ap.parse_args()
    cfg = LLMConfig(train_limit=args.train_limit)
    suffix = "_smoke" if args.smoke else ""

    df = pd.read_parquet(PROCESSED_DIR / "text_accounts.parquet")
    df.index = df.index.astype(str)
    feats = pd.concat([pd.read_csv(PROCESSED_DIR / f, index_col="user_id") for f in ("train.csv", "test.csv")])
    feats.index = feats.index.astype(str)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    rng = np.random.default_rng(SEED)
    if args.smoke:
        train = train.iloc[rng.permutation(len(train))[: args.smoke]]
        test = test.iloc[rng.permutation(len(test))[: max(args.smoke // 4, 20)]]
    elif cfg.train_limit:
        # stratified subsample by group keeps every spambot family represented
        frac = cfg.train_limit / len(train)
        train = train.groupby("group", group_keys=False).sample(frac=frac, random_state=SEED)
    y_te = test["label"].to_numpy()
    en_te = test["has_english_text"].to_numpy()
    p_train, p_test = build_prompts(train, feats, cfg), build_prompts(test, feats, cfg)
    log(f"train {len(p_train)} / test {len(p_test)} prompts; example:\n{p_test[0][:800]}")

    tok, model = load_model(cfg)
    report = {"config": cfg.to_dict(), "n_train": len(train), "n_test": len(test)}
    if not args.skip_zero_shot:
        zs_file = RESULTS_DIR / f"llm_zero_shot_local{suffix}.json"  # cached, so a retry after a GPU reset skips it
        if zs_file.exists():
            report["zero_shot_local"] = read_json(zs_file)
            log("zero-shot (base model): loaded cached result")
        else:
            prob0, ms0 = score(model, tok, p_test, cfg)
            m0, m0_en = metrics(y_te, prob0, en_te)
            report["zero_shot_local"] = {"test_metrics": m0, "test_metrics_english_only": m0_en, "inference_ms": ms0}
            write_json(zs_file, report["zero_shot_local"])
            log(f"zero-shot (base model): F1m={m0['f1_macro']:.4f} AUC={m0['roc_auc']:.4f} {ms0:.0f} ms/account")

    ckpt = None if args.smoke else MODELS_DIR / "llm_lora_checkpoint"
    model, info = train_qlora(model, tok, p_train, train["label"].to_numpy(), cfg, log, checkpoint_dir=ckpt)
    prob, ms = score(model, tok, p_test, cfg)
    m, m_en = metrics(y_te, prob, en_te)
    log(f"QLoRA: F1m={m['f1_macro']:.4f} TPR={m['tpr']:.4f} TNR={m['tnr']:.4f} AUC={m['roc_auc']:.4f} "
        f"| English-only F1m={m_en['f1_macro']:.4f} | {ms:.0f} ms/account | train {info['train_seconds'] / 60:.1f} min")
    report["qlora"] = {"test_metrics": m, "test_metrics_english_only": m_en, "inference_ms": ms, **info,
                       "gpu": torch.cuda.get_device_name(0)}

    pd.DataFrame({"llm_spam_score": prob}, index=test.index).rename_axis("user_id").to_csv(
        PROCESSED_DIR / f"llm_scores{suffix}.csv")
    write_json(RESULTS_DIR / f"llm_qlora{suffix}.json", report)
    if not args.smoke:
        model.save_pretrained(MODELS_DIR / "llm_lora")
        tok.save_pretrained(MODELS_DIR / "llm_lora")
    log("done")


if __name__ == "__main__":
    main()
