"""L4: LLM adversarial robustness test (after Feng et al. 2024).

An LLM rewrites the English tweets of ~200 held-out spam accounts so they read as if
written by an ordinary person, keeping every link, @mention and #hashtag (a spammer
still needs those). Content features are then recomputed from the rewritten tweets,
and each model is rescored:
  * PRISMO Random Forest on FS4 (numeric features)
  * RF on FS4 + text_spam_score (if that variant was exported)
  * L1 text encoder alone
  * L2 QLoRA LLM
Only the text changes; profile and behaviour features are what the attacker cannot
cheaply fake, so they stay as they were.

Usage:  python ml/scripts/adversarial_rewrite.py [--n 200] [--rewriter local|claude]
Outputs: results/adversarial.json, results/adversarial_rewrites.jsonl
"""

from __future__ import annotations

import argparse
import json

import _common  # noqa: F401
import numpy as np
import pandas as pd
import torch

from _common import log, write_json
from spamsentinel_ml.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from spamsentinel_ml.data.cresci import load_tweets, load_users
from spamsentinel_ml.export import load_bundle, predict
from spamsentinel_ml.features import build_account_features
from spamsentinel_ml.text.llm_classifier import LLMConfig, account_prompt
from spamsentinel_ml.text.prepare import MAX_TWEETS, account_texts, detect_english, format_account

REWRITE_INSTRUCTIONS = (
    "Rewrite each tweet below so it sounds like it was written by an ordinary person on Twitter: casual, "
    "natural and varied in wording. Keep the meaning. Keep every URL, @mention and #hashtag exactly as it is. "
    "Return only a JSON array of strings with exactly {n} items, in the same order."
)


def pick_accounts(n: int) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "text_accounts.parquet")
    df.index = df.index.astype(str)
    spam = df[(df["split"] == "test") & (df["label"] == 1) & (df["n_english_tweets"] > 0)]
    frac = min(1.0, n / len(spam))
    return spam.groupby("group", group_keys=False).sample(frac=frac, random_state=SEED)


SINGLE_INSTRUCTIONS = (
    "Rewrite this tweet so it sounds like it was written by an ordinary person on Twitter: casual and "
    "natural. Write it in English. Keep the meaning. Keep every URL, @mention and #hashtag exactly as it is. "
    "Reply with the rewritten tweet only."
)


def rewrite_local(batches: list[list[str]], batch_size: int = 16) -> list[list[str] | None]:
    """One tweet per prompt (a 3B model does not reliably return long JSON arrays).

    A tweet whose rewrite comes back empty keeps its original text; an account counts
    as rewritten when at least half of its tweets were rewritten.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    name = LLMConfig().model_name
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map={"": 0})
    flat = [(a, t) for a, tweets in enumerate(batches) for t in tweets]
    new: list[str | None] = []
    for i in range(0, len(flat), batch_size):
        chunk = flat[i: i + batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": f"{SINGLE_INSTRUCTIONS}\n\nTweet: {t}"}],
                                           tokenize=False, add_generation_prompt=True) for _, t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=120, do_sample=True, temperature=0.8, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        for j in range(len(chunk)):
            text = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            text = text.strip('"').strip()
            new.append(text if text else None)
        if (i // batch_size) % 25 == 0:
            log(f"rewrote {min(i + batch_size, len(flat))}/{len(flat)} tweets")
    del model
    torch.cuda.empty_cache()
    out: list[list[str] | None] = []
    k = 0
    for tweets in batches:
        got = new[k: k + len(tweets)]
        k += len(tweets)
        ok = sum(g is not None for g in got)
        out.append([g if g is not None else t for g, t in zip(got, tweets)] if ok * 2 >= len(tweets) else None)
    return out


def rewrite_claude(batches: list[list[str]], model: str = "claude-opus-5-5") -> list[list[str] | None]:
    import anthropic

    client = anthropic.Anthropic()
    schema = {"type": "object", "properties": {"tweets": {"type": "array", "items": {"type": "string"}}},
              "required": ["tweets"], "additionalProperties": False}
    out = []
    for k, tweets in enumerate(batches, 1):
        resp = client.messages.create(
            model=model, max_tokens=8000,
            messages=[{"role": "user", "content": REWRITE_INSTRUCTIONS.format(n=len(tweets)).replace(
                "Return only a JSON array of strings", "Return the rewritten tweets") + "\n\n"
                + json.dumps(tweets, ensure_ascii=False)}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            got = json.loads(text)["tweets"] if resp.stop_reason != "refusal" else None
        except (json.JSONDecodeError, KeyError):
            got = None
        out.append(got if got and len(got) == len(tweets) else None)
        if k % 20 == 0:
            log(f"rewrote {k}/{len(batches)} accounts")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rewriter", choices=["local", "claude"], default="local")
    args = ap.parse_args()

    chosen = pick_accounts(args.n)
    log(f"{len(chosen)} spam test accounts: {chosen['group'].value_counts().to_dict()}")

    # Re-derive exactly which tweets the text models saw (newest first, first 20 English).
    users_all, tweets_all, targets = [], [], {}
    for group in chosen["group"].unique():
        ids = set(chosen.index[chosen["group"] == group])
        u = load_users(group)
        u = u[u["id"].astype(str).isin(ids)]
        t = load_tweets(group)
        t = t[t["user_id"].isin(ids)].copy()
        t["english"] = detect_english(t["text"].fillna("").astype(str).tolist())
        sel = t[t["english"]].groupby("user_id", sort=False).head(MAX_TWEETS)
        for uid, g in sel.groupby("user_id", sort=False):
            targets[uid] = list(g.index)
        users_all.append(u)
        tweets_all.append(t.drop(columns="english"))
    users = pd.concat(users_all)
    tweets = pd.concat(tweets_all)
    order = [u for u in chosen.index if u in targets]
    originals = [tweets.loc[targets[u], "text"].fillna("").astype(str).tolist() for u in order]

    rewritten = rewrite_local(originals) if args.rewriter == "local" else rewrite_claude(originals)
    failed = sum(r is None for r in rewritten)
    adv = tweets.copy()
    with (RESULTS_DIR / "adversarial_rewrites.jsonl").open("w", encoding="utf-8") as f:
        for u, orig, new in zip(order, originals, rewritten):
            if new is not None:
                adv.loc[targets[u], "text"] = new
            f.write(json.dumps({"user_id": u, "original": orig, "rewritten": new}, ensure_ascii=False) + "\n")
    ok = [u for u, r in zip(order, rewritten) if r is not None]
    log(f"rewrites usable for {len(ok)}/{len(order)} accounts ({failed} not rewritten)")

    # Recompute numeric features and texts before/after for the rewritten accounts.
    f_orig = build_account_features(users, tweets).loc[ok]
    f_adv = build_account_features(users, adv).loc[ok]
    t_adv = account_texts(users[users["id"].astype(str).isin(ok)], adv[adv["user_id"].isin(ok)])
    text_orig = chosen.loc[ok, "text"].tolist()
    text_adv = [format_account(d, list(t)) for d, t in zip(t_adv.loc[ok, "description"], t_adv.loc[ok, "tweets"])]

    results = {"n_accounts": len(ok), "rewriter": args.rewriter, "parse_failures": failed}
    tpr = lambda p: float((np.asarray(p) >= 0.5).mean())  # all accounts are spam: TPR = share flagged

    rf = load_bundle(MODELS_DIR / "spamsentinel_model.joblib")
    results["prismo_rf_fs4"] = {"tpr_before": tpr(predict(rf, f_orig)["spam_score"]),
                                "tpr_after": tpr(predict(rf, f_adv)["spam_score"])}

    enc_dir = MODELS_DIR / "text_encoder"
    if enc_dir.exists():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from spamsentinel_ml.text.encoder import EncoderConfig, predict_proba

        cfg = EncoderConfig()
        tok = AutoTokenizer.from_pretrained(enc_dir)
        enc = AutoModelForSequenceClassification.from_pretrained(enc_dir)
        s_orig = predict_proba(enc, tok, text_orig, cfg)
        s_adv = predict_proba(enc, tok, text_adv, cfg)
        results["text_encoder"] = {"tpr_before": tpr(s_orig), "tpr_after": tpr(s_adv),
                                   "mean_score_before": float(s_orig.mean()), "mean_score_after": float(s_adv.mean())}
        text_bundle = MODELS_DIR / "spamsentinel_model_text.joblib"
        if text_bundle.exists():
            rt = load_bundle(text_bundle)
            results["rf_fs4_plus_text"] = {
                "tpr_before": tpr(predict(rt, f_orig.assign(text_spam_score=s_orig))["spam_score"]),
                "tpr_after": tpr(predict(rt, f_adv.assign(text_spam_score=s_adv))["spam_score"]),
                "text_feature_in_model": "text_spam_score" in rt["features"],
            }
        del enc
        torch.cuda.empty_cache()

    lora_dir = MODELS_DIR / "llm_lora"
    if lora_dir.exists():
        from spamsentinel_ml.text.llm_classifier import load_model, score

        cfg = LLMConfig()
        tok, model = load_model(cfg, adapter_path=lora_dir)
        p_orig = [account_prompt(r, d, list(t), cfg) for r, d, t in zip(
            f_orig.to_dict(orient="records"), chosen.loc[ok, "description"], chosen.loc[ok, "tweets"])]
        p_adv = [account_prompt(r, d, list(t), cfg) for r, d, t in zip(
            f_adv.to_dict(orient="records"), t_adv.loc[ok, "description"], t_adv.loc[ok, "tweets"])]
        results["qlora_llm"] = {"tpr_before": tpr(score(model, tok, p_orig, cfg)[0]),
                                "tpr_after": tpr(score(model, tok, p_adv, cfg)[0])}

    for k, v in results.items():
        if isinstance(v, dict):
            v["tpr_drop"] = v["tpr_before"] - v["tpr_after"]
            log(f"{k}: TPR {v['tpr_before']:.3f} -> {v['tpr_after']:.3f} (drop {v['tpr_drop']:.3f})")
    changed = (f_orig != f_adv).any().loc[lambda s: s].index.tolist()
    results["features_changed_by_rewrite"] = changed
    write_json(RESULTS_DIR / "adversarial.json", results)


if __name__ == "__main__":
    main()
