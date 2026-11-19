"""Cleaning and stratified splitting (FR2).

Scaling note: every classifier used is insensitive to feature scale (tree
ensembles and rules split on thresholds, GaussianNB fits one Gaussian per
feature), so no scaler is added; missing values are imputed with training
medians inside each model Pipeline so CV folds never see validation data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import MIN_TWEETS_PER_ACCOUNT, SEED, TEST_SIZE
from .features import FEATURE_NAMES


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove duplicate/ambiguous accounts and accounts without usable tweets."""
    report: dict[str, int] = {"accounts_in": int(len(df))}
    dup = df.index.duplicated(keep=False)
    n_labels = df.groupby(level=0)["label"].transform("nunique")
    report["accounts_in_several_groups"] = int(df[dup].index.nunique())
    report["accounts_with_conflicting_labels"] = int(df[n_labels > 1].index.nunique())
    df = df[n_labels == 1]  # drop accounts labelled both spam and genuine
    df = df[~df.index.duplicated(keep="first")]  # same label in two groups: keep one copy
    few = df["n_tweets_collected"].fillna(0) < MIN_TWEETS_PER_ACCOUNT
    report["accounts_without_tweets"] = int(few.sum())
    df = df[~few]
    missing_core = df[["followers_count", "followees_count", "statuses_count"]].isna().any(axis=1)
    report["accounts_missing_core_counts"] = int(missing_core.sum())
    df = df[~missing_core]
    df[FEATURE_NAMES] = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    report["accounts_out"] = int(len(df))
    report["missing_values_remaining"] = int(df[FEATURE_NAMES].isna().sum().sum())
    return df, report


def split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified by group, so each spambot family appears in train and test in proportion."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["group"])
    return train.sort_index(), test.sort_index()


def xy(df: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    features = features or FEATURE_NAMES
    return df[features], df["label"].to_numpy(dtype=int)
