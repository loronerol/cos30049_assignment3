import math

import pandas as pd
import pytest

from spamsentinel_ml.config import PROCESSED_DIR
from spamsentinel_ml.text.encoder import oof_plan
from spamsentinel_ml.text.llm_classifier import LLMConfig, account_prompt
from spamsentinel_ml.text.prepare import NO_TEXT, format_account, normalise


def test_oof_plan_never_scores_training_accounts():
    folds = {f"u{i}": i % 5 for i in range(50)}
    plan = oof_plan(folds)
    assert sorted(plan) == [0, 1, 2, 3, 4]
    scored = []
    for k, part in plan.items():
        assert not set(part["train"]) & set(part["score"])
        assert all(folds[u] == k for u in part["score"])
        assert all(folds[u] != k for u in part["train"])
        scored += part["score"]
    assert sorted(scored) == sorted(folds)  # every account scored exactly once


@pytest.mark.skipif(not (PROCESSED_DIR / "text_scores.csv").exists(), reason="text encoder not trained yet")
def test_saved_oof_scores_follow_the_fold_plan():
    scores = pd.read_csv(PROCESSED_DIR / "text_scores.csv", index_col="user_id")
    folds = pd.read_csv(PROCESSED_DIR / "text_folds.csv", index_col="user_id")["fold"]
    train = scores[scores["split"] == "train"]
    assert (train["fold"] == folds.reindex(train.index)).all()
    assert (scores.loc[scores["split"] == "test", "fold"] == -1).all()
    assert scores["text_spam_score"].between(0, 1).all()


def test_normalise_and_format():
    assert normalise("Hi @bob see https://x.co/1  now") == "Hi @user see http now"
    assert format_account("", []) == NO_TEXT
    assert format_account("I love tea", ["one", "two"]) == "Bio: I love tea </s> one </s> two"


def test_account_prompt_handles_missing_values():
    cfg = LLMConfig()
    feats = {"followers_count": 1234.0, "frac_tweets_with_url": 0.5, "account_age_days": math.nan,
             "default_profile_image": 1.0}
    p = account_prompt(feats, "", ["t" * 500] + [f"tweet {i}" for i in range(20)], cfg)
    assert "Followers: 1,234" in p
    assert "Share of tweets with a link: 50%" in p
    assert "Account age (days): unknown" in p
    assert "Default profile image: yes" in p
    assert "(none or not in English)" in p
    assert "t" * cfg.max_tweet_chars in p and "t" * (cfg.max_tweet_chars + 1) not in p
    assert f"{cfg.max_tweets}. " in p and f"{cfg.max_tweets + 1}. " not in p
