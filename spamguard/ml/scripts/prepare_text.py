"""L0: build English account texts and shared out-of-fold folds for L1-L3.

Uses the same accounts and train/test split as the core pipeline
(data/processed/train.csv, test.csv).

Usage:  python ml/scripts/prepare_text.py
Outputs (data/processed/):
  text_accounts.parquet   user_id, label, group, split, description, tweets, text
  text_folds.csv          user_id, fold (0-4) for training accounts
  text_report.json        how much text the English filter drops, by group
"""

from __future__ import annotations

import _common  # noqa: F401
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from _common import log, write_json
from spamsentinel_ml.config import GROUPS, PROCESSED_DIR, SEED
from spamsentinel_ml.data.cresci import load_tweets, load_users
from spamsentinel_ml.text.prepare import account_texts, format_account

N_FOLDS = 5


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train.csv", index_col="user_id", usecols=["user_id", "group", "label"])
    test = pd.read_csv(PROCESSED_DIR / "test.csv", index_col="user_id", usecols=["user_id", "group", "label"])
    accounts = pd.concat([train.assign(split="train"), test.assign(split="test")])
    accounts.index = accounts.index.astype(str)

    frames = []
    for group in GROUPS:
        users = load_users(group)
        users = users[users["id"].astype(str).isin(accounts.index)]
        tweets = load_tweets(group)
        tweets = tweets[tweets["user_id"].isin(users["id"].astype(str))]
        t = account_texts(users, tweets)
        log(f"{group}: {len(t):,} accounts, {t['n_english_tweets'].sum():,.0f}/{t['n_tweets_considered'].sum():,} "
            f"tweets English, {(~t['has_english_text']).sum():,} accounts without English text")
        frames.append(t)
    texts = pd.concat(frames)
    texts = texts[~texts.index.duplicated()]
    df = accounts.join(texts, how="left")
    df["tweets"] = df["tweets"].apply(lambda v: v if isinstance(v, list) else [])
    df["description"] = df["description"].fillna("")
    df["has_english_text"] = df["has_english_text"].fillna(False).astype(bool)
    df["text"] = [format_account(d, t) for d, t in zip(df["description"], df["tweets"])]

    tr = df[df["split"] == "train"]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = pd.Series(-1, index=tr.index, name="fold")
    for k, (_, val) in enumerate(skf.split(tr, tr["group"])):
        folds.iloc[val] = k

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "text_accounts.parquet")
    folds.to_csv(PROCESSED_DIR / "text_folds.csv")
    by_group = df.groupby("group").agg(
        accounts=("label", "size"),
        without_english_text=("has_english_text", lambda s: int((~s).sum())),
        english_tweets=("n_english_tweets", "sum"),
        tweets_considered=("n_tweets_considered", "sum"),
    )
    by_group["share_tweets_english"] = by_group["english_tweets"] / by_group["tweets_considered"]
    write_json(PROCESSED_DIR / "text_report.json", {
        "max_tweets_per_account": 20,
        "filter": "fast-langdetect lid.176 (lite), English with score >= 0.5 and >= 3 words",
        "n_folds": N_FOLDS,
        "by_group": by_group.reset_index().to_dict(orient="records"),
        "accounts_without_english_text": int((~df["has_english_text"]).sum()),
    })
    log(f"wrote {len(df):,} account texts; {int((~df['has_english_text']).sum()):,} without English text")


if __name__ == "__main__":
    main()
