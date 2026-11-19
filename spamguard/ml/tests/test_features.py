import numpy as np
import pandas as pd
import pytest

from spamsentinel_ml.data.cresci import parse_time
from spamsentinel_ml.features import FEATURE_NAMES, build_account_features, data_dictionary


@pytest.fixture()
def accounts():
    users = pd.DataFrame({
        "id": ["1", "2"],
        "name": ["Alice Smith", "Deals4U"],
        "screen_name": ["alice", "deals2024x"],
        "statuses_count": ["300", "5000"],
        "followers_count": ["150", "40"],
        "friends_count": ["100", "1850"],
        "favourites_count": ["20", "0"],
        "listed_count": ["3", "0"],
        "url": [None, "http://spam.example"],
        "default_profile": [None, "1"],
        "default_profile_image": [None, "1"],
        "geo_enabled": ["1", None],
        "verified": [None, None],
        "description": ["Coffee and books", ""],
        "created_at": ["Mon Jan 01 00:00:00 +0000 2018", "Sat Dec 01 00:00:00 +0000 2018"],
        "group": ["genuine_accounts", "social_spambots_1"],
        "label": [0, 1],
    })
    tweets = pd.DataFrame({
        "id": ["10", "11", "12", "20", "21", "22"],
        "user_id": ["1", "1", "1", "2", "2", "2"],
        "text": [
            "Reading a great book today #books",
            "@bob see you at lunch",
            "RT @carol: lovely weather",
            "WIN a FREE iPhone http://x.co/1 #giveaway",
            "WIN a FREE iPhone http://x.co/2 #giveaway",
            "Cheap deals http://x.co/3 @dave @erin",
        ],
        "source": [
            '<a href="http://twitter.com">Twitter Web Client</a>',
            '<a href="http://twitter.com">Twitter for iPhone</a>',
            '<a href="http://twitter.com">Twitter for iPhone</a>',
            '<a href="http://dlvr.it">dlvr.it</a>',
            '<a href="http://dlvr.it">dlvr.it</a>',
            '<a href="http://dlvr.it">dlvr.it</a>',
        ],
        "in_reply_to_status_id": ["0", "555", "0", "0", "0", "0"],
        "retweeted_status_id": ["0", "0", "777", "0", "0", "0"],
        "retweet_count": ["1", "0", "4", "0", "0", "0"],
        "created_at": [
            "Mon Jan 01 10:00:00 +0000 2019", "Tue Jan 02 10:00:00 +0000 2019", "Thu Jan 04 10:00:00 +0000 2019",
            "Sat Jan 01 12:00:00 +0000 2019", "Sat Jan 01 12:01:00 +0000 2019", "Sat Jan 01 12:02:00 +0000 2019",
        ],
    })
    tweets["time"] = parse_time(tweets["created_at"])
    return build_account_features(users, tweets)


def test_every_registered_feature_is_present(accounts):
    assert list(accounts.columns[-len(FEATURE_NAMES):]) == FEATURE_NAMES
    assert len(accounts) == 2


def test_content_features(accounts):
    a, s = accounts.loc["1"], accounts.loc["2"]
    assert a["frac_tweets_with_url"] == 0
    assert s["frac_tweets_with_url"] == 1
    assert s["urls_per_tweet_max"] == 1
    assert s["mentions_per_tweet_max"] == 2
    assert a["frac_replies"] == pytest.approx(1 / 3)
    assert a["frac_retweets"] == pytest.approx(1 / 3)
    assert s["frac_tweets_with_spam_word"] == 1
    assert s["frac_duplicate_tweets"] == pytest.approx(1 / 3)  # the second WIN tweet repeats the first
    assert s["frac_third_party_client"] == 1
    assert a["frac_third_party_client"] == 0
    assert a["n_distinct_sources"] == 2
    assert s["distinct_mentions_per_tweet"] == pytest.approx(2 / 3)
    assert a["retweets_received_per_tweet_max"] == 4


def test_behaviour_and_profile_features(accounts):
    a, s = accounts.loc["1"], accounts.loc["2"]
    assert s["follower_followee_ratio"] == pytest.approx(40 / 1850)
    assert a["inter_tweet_hours_min"] == pytest.approx(24)
    assert a["inter_tweet_hours_max"] == pytest.approx(48)
    assert s["inter_tweet_hours_median"] == pytest.approx(1 / 60)
    assert s["max_tweets_in_a_day"] == 3
    assert a["account_age_days"] == pytest.approx(368.417, rel=1e-3)
    assert s["default_profile_image"] == 1 and a["default_profile_image"] == 0
    assert s["screen_name_digits"] == 4
    assert s["has_url"] == 1 and a["has_url"] == 0
    assert a["has_description"] == 1 and s["has_description"] == 0
    assert a["geo_enabled"] == 1


def test_data_dictionary_has_ranges(accounts):
    dd = data_dictionary(accounts)
    assert set(dd["category"]) == {"content", "behaviour", "profile"}
    assert len(dd) == len(FEATURE_NAMES)
    assert np.isfinite(dd["train_median"]).all()
