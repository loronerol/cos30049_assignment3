"""Final comparison table for the A2/A3 report: results/comparison.md (+ .json).

Reads the results each pipeline step already wrote; nothing is retrained. The two
Random Forest bundles are re-scored on the test split (cheap) to get their
English-only metrics and per-account inference time.

Rows: PRISMO RF on FS4 | RF on FS4 + text score | L1 text encoder | Qwen zero-shot
(local) | L2 QLoRA Qwen | L3 Claude zero-shot. Missing results show as "not run".

Usage:  python ml/scripts/compare_models.py
"""

from __future__ import annotations

import platform
import time

import _common  # noqa: F401
import numpy as np
import pandas as pd

from _common import load_split, log, read_json, write_json
from spamsentinel_ml.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from spamsentinel_ml.evaluate import classification_metrics
from spamsentinel_ml.export import load_bundle, predict

COLS = ["accuracy", "tpr", "tnr", "f1_macro", "roc_auc"]


def _maybe(path):
    return read_json(path) if path.exists() else None


def _cpu() -> str:
    if platform.system() == "Windows":
        import subprocess

        try:
            name = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                  capture_output=True, text=True, timeout=20).stdout.strip()
            if name:
                return name
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or "CPU"


def _gpu() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except ImportError:
        return "GPU"


def encoder_ms(n: int = 256) -> float:
    """Mean GPU inference time per account for the exported text encoder."""
    path = MODELS_DIR / "text_encoder"
    if not path.exists():
        return float("nan")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from spamsentinel_ml.text.encoder import EncoderConfig, predict_proba

    df = pd.read_parquet(PROCESSED_DIR / "text_accounts.parquet", columns=["split", "text"])
    texts = df.loc[df["split"] == "test", "text"].tolist()[:n]
    cfg = EncoderConfig()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    predict_proba(model, tok, texts[:16], cfg)  # warm-up
    t0 = time.perf_counter()
    predict_proba(model, tok, texts, cfg)
    return (time.perf_counter() - t0) * 1000 / len(texts)


def score_bundle(path, with_text: bool, english: pd.Series) -> dict | None:
    if not path.exists():
        return None
    bundle = load_bundle(path)
    test, _ = load_split("test", with_text)
    t0 = time.perf_counter()
    res = predict(bundle, test)
    ms = (time.perf_counter() - t0) * 1000 / len(test)
    prob = res["spam_score"].to_numpy()
    y = test["label"].to_numpy()
    en = english.reindex(test.index.astype(str)).fillna(False).to_numpy(dtype=bool)
    return {
        "test": classification_metrics(y, (prob >= 0.5).astype(int), prob),
        "test_en": classification_metrics(y[en], (prob[en] >= 0.5).astype(int), prob[en]),
        "inference_ms": ms,
        "train_seconds": bundle["metrics"].get("fit_ms", float("nan")) / 1000,
        "features": len(bundle["features"]),
        "uses_text": "text_spam_score" in bundle["features"],
    }


def main() -> None:
    texts = pd.read_parquet(PROCESSED_DIR / "text_accounts.parquet", columns=["has_english_text"])
    texts.index = texts.index.astype(str)
    english = texts["has_english_text"]
    cpu = _cpu()
    gpu = _gpu()

    rows: list[dict] = []

    def add(name, data, hardware, note=""):
        if data is None:
            rows.append({"model": name, "status": "not run", "note": note})
        else:
            rows.append({"model": name, "status": "ok", "hardware": hardware, "note": note, **data})

    rf = score_bundle(MODELS_DIR / "spamsentinel_model.joblib", False, english)
    add(f"PRISMO RF on FS4 ({rf['features'] if rf else '?'} features)", rf, cpu)
    rft = score_bundle(MODELS_DIR / "spamsentinel_model_text.joblib", True, english)
    note = "" if rft is None else ("FS4 kept text_spam_score" if rft["uses_text"] else "FS4 dropped text_spam_score")
    add(f"RF on FS4 + text score ({rft['features'] if rft else '?'} features)", rft, cpu, note)

    enc = _maybe(RESULTS_DIR / "text_encoder.json")
    add("L1 text encoder alone (Twitter RoBERTa)", None if enc is None else {
        "test": enc["test_metrics"], "test_en": enc["test_metrics_english_only"],
        "train_seconds": enc["final"]["train_seconds"], "inference_ms": encoder_ms(),
    }, gpu)

    llm = _maybe(RESULTS_DIR / "llm_qlora.json")
    zs = (llm or {}).get("zero_shot_local")
    add("Qwen2.5-3B zero-shot (local, no training)", None if zs is None else {
        "test": zs["test_metrics"], "test_en": zs["test_metrics_english_only"], "train_seconds": 0.0,
        "inference_ms": zs["inference_ms"],
    }, gpu)
    q = (llm or {}).get("qlora")
    add("L2 QLoRA Qwen2.5-3B", None if q is None else {
        "test": q["test_metrics"], "test_en": q["test_metrics_english_only"], "train_seconds": q["train_seconds"],
        "inference_ms": q["inference_ms"],
    }, gpu, "" if q is None else f"trained on {llm['n_train']} accounts")

    l3 = _maybe(RESULTS_DIR / "llm_zero_shot.json")
    add("L3 Claude zero-shot", None if l3 is None else {
        "test": l3["test_metrics"], "test_en": l3["test_metrics_english_only"], "train_seconds": 0.0,
        "inference_ms": float("nan"),
    }, "Claude API", "skipped (paid; needs an API key)" if l3 is None else l3["model"])

    adv = _maybe(RESULTS_DIR / "adversarial.json")
    write_json(RESULTS_DIR / "comparison.json", {"rows": rows, "adversarial": adv})
    (RESULTS_DIR / "comparison.md").write_text(markdown(rows, adv), encoding="utf-8")
    log(f"wrote results/comparison.md ({sum(r['status'] == 'ok' for r in rows)}/{len(rows)} models with results)")


def _fmt(v, nd=4):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def markdown(rows: list[dict], adv: dict | None) -> str:
    out = ["# Model comparison — held-out test split\n",
           "Same test accounts for every model (1,789; spam is the positive class). The English-only view "
           "keeps the test accounts with at least one English tweet or bio, since the text models see only "
           "English text.\n"]
    for key, title in [("test", "All test accounts"), ("test_en", "English-text test accounts only")]:
        out += [f"## {title}\n",
                "| Model | Accuracy | TPR | TNR | F1 (macro) | ROC-AUC | Training time | Inference ms/account | Hardware | Note |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["status"] != "ok":
                out.append(f"| {r['model']} | not run | | | | | | | | {r.get('note', '')} |")
                continue
            m = r[key]
            secs = r.get("train_seconds", float("nan"))
            train = "–" if np.isnan(secs) else (f"{secs:.1f} s" if secs < 120 else f"{secs / 60:.1f} min")
            out.append(f"| {r['model']} | {_fmt(m['accuracy'])} | {_fmt(m['tpr'])} | {_fmt(m['tnr'])} | "
                       f"{_fmt(m['f1_macro'])} | {_fmt(m['roc_auc'])} | {train} | "
                       f"{_fmt(r.get('inference_ms'), 2)} | {r['hardware']} | {r.get('note', '')} |")
        out.append("")
    if adv:
        out += ["## L4 — adversarial rewrite of spam tweets\n",
                f"{adv['n_accounts']} held-out spam accounts; tweets rewritten by the {adv['rewriter']} LLM to sound "
                "human (links, mentions and hashtags kept). All accounts are spam, so TPR = share still flagged.\n",
                "| Model | TPR before | TPR after | Drop |", "|---|---|---|---|"]
        for k, v in adv.items():
            if isinstance(v, dict) and "tpr_before" in v:
                out.append(f"| {k} | {v['tpr_before']:.3f} | {v['tpr_after']:.3f} | {v['tpr_drop']:.3f} |")
        out.append(f"\nFeatures changed by the rewrite: {', '.join(adv.get('features_changed_by_rewrite', [])) or 'none'}\n")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
