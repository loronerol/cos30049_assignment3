"""L3 (optional, paid): zero-shot baseline with a Claude model on the held-out test split.

Same account prompt as the L2 QLoRA model, no training. Runs through the Message
Batches API (50% of standard prices). The Claude API exposes no token logprobs, so
the model returns a stated spam probability (0-100) with its label via a JSON schema;
that probability is used for ROC-AUC and the review band.

Spending guard: by default the script only *estimates* the cost (token counting on a
sample). It submits the batch only with --confirm, and refuses if the estimate is
above --max-usd. Raw outputs are cached in results/llm_zero_shot.jsonl, so a rerun
never pays twice for an account.

Usage:
  python ml/scripts/zero_shot_claude.py                    # estimate only
  python ml/scripts/zero_shot_claude.py --confirm          # submit, wait, score
  python ml/scripts/zero_shot_claude.py --collect BATCH_ID # fetch an earlier batch
"""

from __future__ import annotations

import argparse
import json
import time

import _common  # noqa: F401
import anthropic
import numpy as np
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from _common import log, write_json
from spamsentinel_ml.config import PROCESSED_DIR, RESULTS_DIR
from spamsentinel_ml.evaluate import classification_metrics
from spamsentinel_ml.text.llm_classifier import SYSTEM_PROMPT, LLMConfig, account_prompt

# $ per million tokens (standard API); the Batches API bills half of this.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
CACHE = RESULTS_DIR / "llm_zero_shot.jsonl"
EST_OUTPUT_TOKENS = 400  # low-effort thinking + a short JSON answer, per account

ZS_SYSTEM = SYSTEM_PROMPT.replace(
    "Answer with exactly one word: spam or genuine.",
    "Give your label, your probability (0-100) that the account is spam, and a one-sentence reason.",
)
SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["spam", "genuine"]},
        "spam_probability": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["label", "spam_probability", "reason"],
    "additionalProperties": False,
}


def build(model: str) -> tuple[pd.DataFrame, dict[str, dict]]:
    df = pd.read_parquet(PROCESSED_DIR / "text_accounts.parquet")
    df.index = df.index.astype(str)
    test = df[df["split"] == "test"]
    feats = pd.read_csv(PROCESSED_DIR / "test.csv", index_col="user_id")
    feats.index = feats.index.astype(str)
    rows = feats.loc[test.index].to_dict(orient="index")
    cfg = LLMConfig()
    params = {}
    for uid, desc, tweets in zip(test.index, test["description"], test["tweets"]):
        params[f"u{uid}"] = {
            "model": model,
            "max_tokens": 2048,
            "system": ZS_SYSTEM,
            "messages": [{"role": "user", "content": account_prompt(rows[uid], desc, list(tweets), cfg)}],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        }
    return test, params


def cached() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        out[rec["custom_id"]] = rec
    return out


def estimate(client: anthropic.Anthropic, params: dict[str, dict], model: str, sample: int = 25) -> float:
    keys = list(params)[:sample]
    toks = [client.messages.count_tokens(model=model, system=params[k]["system"], messages=params[k]["messages"])
            .input_tokens for k in keys]
    mean_in = float(np.mean(toks))
    p_in, p_out = PRICES[model]
    usd = len(params) * (mean_in * p_in + EST_OUTPUT_TOKENS * p_out) / 1e6 * 0.5
    log(f"{len(params)} requests x ~{mean_in:.0f} input + ~{EST_OUTPUT_TOKENS} output tokens "
        f"on {model} (batch, 50% off): estimated ${usd:.2f}")
    return usd


def collect(client: anthropic.Anthropic, batch_id: str) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        log(f"batch {batch_id}: {batch.processing_status}, processing={batch.request_counts.processing}")
        time.sleep(60)
    with CACHE.open("a", encoding="utf-8") as f:
        for res in client.messages.batches.results(batch_id):
            rec = {"custom_id": res.custom_id, "type": res.result.type}
            if res.result.type == "succeeded":
                msg = res.result.message
                rec["stop_reason"] = msg.stop_reason
                rec["usage"] = {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}
                text = next((b.text for b in msg.content if b.type == "text"), "")
                if msg.stop_reason == "refusal":
                    rec["answer"] = None
                else:
                    try:
                        rec["answer"] = json.loads(text)
                    except json.JSONDecodeError:
                        rec["answer"] = None
            f.write(json.dumps(rec) + "\n")


def score(test: pd.DataFrame, model: str) -> None:
    recs = cached()
    prob = []
    for uid in test.index:
        ans = (recs.get(f"u{uid}") or {}).get("answer")
        prob.append(np.nan if not ans else min(max(ans["spam_probability"], 0), 100) / 100)
    prob = np.asarray(prob)
    ok = ~np.isnan(prob)
    y = test["label"].to_numpy()
    en = test["has_english_text"].to_numpy()
    m = classification_metrics(y[ok], (prob[ok] >= 0.5).astype(int), prob[ok])
    m_en = classification_metrics(y[ok & en], (prob[ok & en] >= 0.5).astype(int), prob[ok & en])
    usage = [r["usage"] for r in recs.values() if "usage" in r]
    p_in, p_out = PRICES[model]
    cost = sum(u["input"] * p_in + u["output"] * p_out for u in usage) / 1e6 * 0.5
    write_json(RESULTS_DIR / "llm_zero_shot.json", {
        "model": model, "test_metrics": m, "test_metrics_english_only": m_en,
        "answered": int(ok.sum()), "unanswered": int((~ok).sum()), "cost_usd": cost,
        "probability_source": "model-stated spam probability (0-100) via JSON schema",
    })
    log(f"zero-shot {model}: F1m={m['f1_macro']:.4f} AUC={m['roc_auc']:.4f} TPR={m['tpr']:.4f} "
        f"on {ok.sum()} accounts (unanswered {int((~ok).sum())}); cost ${cost:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5", choices=sorted(PRICES))
    ap.add_argument("--max-usd", type=float, default=15.0)
    ap.add_argument("--confirm", action="store_true", help="actually submit the paid batch")
    ap.add_argument("--collect", metavar="BATCH_ID", help="fetch results of an earlier batch")
    args = ap.parse_args()

    client = anthropic.Anthropic()
    test, params = build(args.model)
    if args.collect:
        collect(client, args.collect)
        score(test, args.model)
        return
    done = cached()
    todo = {k: v for k, v in params.items() if k not in done or done[k].get("answer") is None}
    if not todo:
        score(test, args.model)
        return
    usd = estimate(client, todo, args.model)
    if usd > args.max_usd:
        raise SystemExit(f"estimate ${usd:.2f} exceeds --max-usd {args.max_usd}; aborting")
    if not args.confirm:
        log("estimate only; rerun with --confirm to submit")
        return
    batch = client.messages.batches.create(requests=[
        Request(custom_id=k, params=MessageCreateParamsNonStreaming(**v)) for k, v in todo.items()
    ])
    log(f"submitted batch {batch.id} with {len(todo)} requests")
    collect(client, batch.id)
    score(test, args.model)


if __name__ == "__main__":
    main()
