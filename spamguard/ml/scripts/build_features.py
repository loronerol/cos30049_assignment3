"""Phase 1: Cresci-2017 (+ Cresci-2015 human groups) raw files -> cleaned account
features and stratified splits (D2).

Usage:  python ml/scripts/build_features.py
Outputs (data/processed/):
  features.csv          all cleaned accounts (user_id, group, label, 64 features)
  train.csv, test.csv   stratified 80/20 split
  data_dictionary.csv   feature category, unit, description and training range
  cleaning_report.json  what was removed and why
"""

from __future__ import annotations

import _common  # noqa: F401  (sets sys.path)
import pandas as pd

from _common import log, write_json
from spamsentinel_ml.config import GROUPS, PROCESSED_DIR, RAW_ROOT
from spamsentinel_ml.data.cresci import extract_nested, load_tweets, load_users
from spamsentinel_ml.features import FEATURE_NAMES, build_account_features, data_dictionary
from spamsentinel_ml.preprocess import clean, split


DISCLAIMER = """Derived from the Cresci-2017 and Cresci-2015 datasets ("The Fake Project"),
released by the Institute of Informatics and Telematics of the Italian National
Research Council (CNR), http://mib.projects.iit.cnr.it

Cresci et al. (2017) The paradigm-shift of social spambots: Evidence, theories and
tools for the arms race. WWW '17 Companion, 963-972.
Cresci et al. (2015) Fame for sale: efficient detection of fake Twitter followers.
Decision Support Systems, 80, 56-71.

LIABILITY DISCLAIMER (from the original archive)
The information included in the files stored in this archive are released FOR
RESEARCH PURPOSES ONLY. Any improper use or abuse of the released data can be
prosecuted in accordance with the laws in force. Any redistribution without this
disclaimer can be considered for prosecution in accordance with the laws in force.
"""


def main() -> None:
    for ds in {d for d, _ in GROUPS.values()}:
        extract_nested(RAW_ROOT / ds)
    frames, per_group = [], {}
    for group in GROUPS:
        users = load_users(group)
        tweets = load_tweets(group)
        log(f"{group}: {len(users):,} users, {len(tweets):,} tweets kept (<=200/account)")
        feats = build_account_features(users, tweets)
        per_group[group] = {"dataset": GROUPS[group][0], "label": GROUPS[group][1], "users": len(users),
                            "tweets_kept": len(tweets), "accounts_with_tweets": len(feats)}
        frames.append(feats)
    df = pd.concat(frames)
    df.index.name = "user_id"
    df = df.drop(columns=["screen_name"], errors="ignore")  # not needed; keeps outputs pseudonymous

    df, report = clean(df)
    report["per_group"] = per_group
    report["class_balance"] = df["label"].value_counts().rename({0: "genuine", 1: "spam"}).to_dict()
    report["group_counts"] = df["group"].value_counts().to_dict()
    train, test = split(df)
    report["train_size"], report["test_size"] = len(train), len(test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "features.csv")
    train.to_csv(PROCESSED_DIR / "train.csv")
    test.to_csv(PROCESSED_DIR / "test.csv")
    data_dictionary(train).to_csv(PROCESSED_DIR / "data_dictionary.csv", index=False)
    write_json(PROCESSED_DIR / "cleaning_report.json", report)
    (PROCESSED_DIR / "LICENSE_DISCLAIMER.txt").write_text(DISCLAIMER, encoding="utf-8")
    log(f"done: {len(df):,} accounts x {len(FEATURE_NAMES)} features; "
        f"train {len(train):,} / test {len(test):,}; balance {report['class_balance']}")


if __name__ == "__main__":
    main()
