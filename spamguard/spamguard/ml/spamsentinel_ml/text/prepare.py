"""L0: per-account English text for the text encoder (L1) and the LLM (L2/L3).

For each account: the profile description (if English) and up to 20 of its most
recent English tweets. Only English text is kept, so that the text models cannot
learn the shortcut "Italian = spam" from social_spambots_1 (the Italian E13 and
TFP humans would make that shortcut wrong anyway, but filtering removes it).
Accounts with no English text get an explicit empty-text marker.

URLs are replaced with "http" and user mentions with "@user", following the
convention of the cardiffnlp Twitter models.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NO_TEXT = "[no English text]"
MAX_TWEETS = 20
MIN_WORDS = 3  # shorter texts carry too little signal to identify a language
LANG_THRESHOLD = 0.5


def normalise(text: str) -> str:
    text = URL_RE.sub("http", text)
    text = MENTION_RE.sub("@user", text)
    return re.sub(r"\s+", " ", text).strip()


def _lang_view(text: str) -> str:
    """Text used only for language detection: no URLs, mentions, RT marker or '#'."""
    t = URL_RE.sub(" ", text)
    t = MENTION_RE.sub(" ", t)
    t = re.sub(r"^RT\b:?", " ", t.strip())
    t = t.replace("#", " ")
    return re.sub(r"\s+", " ", t).strip()


def detect_english(texts: list[str]) -> np.ndarray:
    """Boolean per text: confidently English with at least MIN_WORDS words."""
    from fast_langdetect import detect

    out = np.zeros(len(texts), dtype=bool)
    for i, t in enumerate(texts):
        v = _lang_view(t or "")
        if len(v.split()) < MIN_WORDS:
            continue
        try:
            r = detect(v, model="lite")[0]
        except Exception:  # noqa: BLE001 - detector errors mean "not English"
            continue
        out[i] = r["lang"] == "en" and r["score"] >= LANG_THRESHOLD
    return out


def account_texts(users: pd.DataFrame, tweets: pd.DataFrame) -> pd.DataFrame:
    """One row per account: English description, English tweets and filter counts.

    `tweets` must already be sorted newest first per user (as load_tweets returns).
    """
    tw = tweets[["user_id", "text"]].copy()
    tw["text"] = tw["text"].fillna("").astype(str)
    tw["english"] = detect_english(tw["text"].tolist())
    eng = tw[tw["english"]].groupby("user_id", sort=False).head(MAX_TWEETS)
    eng = eng.assign(norm=eng["text"].map(normalise))
    tweets_en = eng.groupby("user_id", sort=False)["norm"].apply(list)
    counts = tw.groupby("user_id").agg(n_tweets_considered=("text", "size"), n_english_tweets=("english", "sum"))

    desc = users.set_index(users["id"].astype(str))["description"].fillna("").astype(str)
    desc_en_mask = detect_english(desc.tolist())
    desc_en = pd.Series(np.where(desc_en_mask, desc.map(normalise), ""), index=desc.index)

    out = counts.join(tweets_en.rename("tweets"), how="left")
    out["tweets"] = out["tweets"].apply(lambda v: v if isinstance(v, list) else [])
    out["description"] = desc_en.reindex(out.index).fillna("")
    out["has_english_text"] = (out["tweets"].str.len() > 0) | (out["description"] != "")
    out.index.name = "user_id"
    return out


def format_account(description: str, tweets: list[str]) -> str:
    """Flat input string for the text encoder."""
    if not description and not tweets:
        return NO_TEXT
    parts = []
    if description:
        parts.append(f"Bio: {description}")
    parts.extend(tweets)
    return " </s> ".join(parts)
