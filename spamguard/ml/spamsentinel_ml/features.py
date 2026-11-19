"""Account-level features equivalent to Benevenuto et al. (2010), grouped into the
three categories of the project plan (Table 2): content, user behaviour, profile.

Content features describe an account's most recent tweets (statistics of
per-tweet counts and fractions of tweets with a property). Behaviour features
describe how the account acts and is connected. Profile features describe how
the account is set up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

URL_RE = r"https?://\S+|www\.\S+"
HASHTAG_RE = r"#\w+"
MENTION_RE = r"@\w+"
DIGIT_RE = r"\d"

# Words common in unsolicited advertising and scams (after Benevenuto et al. 2010,
# who counted tweets containing words from a list of popular spam words).
SPAM_WORDS = [
    "free", "win", "winner", "won", "prize", "cash", "money", "earn", "income", "profit",
    "click", "offer", "discount", "deal", "deals", "sale", "buy", "cheap", "bonus", "giveaway",
    "promo", "coupon", "limited", "act now", "loan", "credit", "casino", "bet", "viagra",
    "weight loss", "diet", "work from home", "make money", "followers", "follow back",
    "subscribe", "check out", "visit", "% off", "order now", "guaranteed", "risk free",
]
SPAM_RE = r"(?i)(?:" + "|".join(r"\b" + re.escape(w).replace(r"\ ", r"\s+") + r"\b" for w in SPAM_WORDS) + ")"

# Official first-party Twitter clients; anything else counts as a third-party
# client (schedulers, auto-posters, cross-posting tools, bots).
OFFICIAL_CLIENTS = [
    "twitter web client", "twitter for iphone", "twitter for android", "twitter for ipad",
    "mobile web", "twitter for windows phone", "twitter for blackberry", "twitter for mac",
    "twitter for windows", "tweetdeck", "twitter web app", "twitter lite", "iphone", "android",
    # legacy names used in 2009-2013 tweets
    "web", "txt", "mobile web (m2)", "mobile web (m5)", "twitter for blackberry®",
    "twitter for android tablets", "twitter for nokia s40", "ios", "os x",
]

PER_TWEET_COUNTS = {
    "hashtags": "hashtags",
    "urls": "URLs",
    "mentions": "mentions",
    "chars": "characters",
    "words": "words",
    "digits": "numeric characters",
    "retweets_received": "times the tweet was retweeted",
}
STATS = ["max", "min", "mean", "median"]


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    category: str  # content | behaviour | profile
    unit: str
    description: str
    integer: bool = False
    binary: bool = False


def _registry() -> list[FeatureInfo]:
    feats: list[FeatureInfo] = []
    for key, label in PER_TWEET_COUNTS.items():
        for s in STATS:
            feats.append(FeatureInfo(
                f"{key}_per_tweet_{s}", "content", "per tweet",
                f"{s.capitalize()} number of {label} per tweet",
            ))
    feats += [
        FeatureInfo("frac_tweets_with_url", "content", "fraction", "Fraction of tweets containing a URL"),
        FeatureInfo("frac_tweets_with_hashtag", "content", "fraction", "Fraction of tweets containing a hashtag"),
        FeatureInfo("frac_tweets_with_mention", "content", "fraction", "Fraction of tweets mentioning another user"),
        FeatureInfo("frac_replies", "content", "fraction", "Fraction of tweets that are replies"),
        FeatureInfo("frac_retweets", "content", "fraction", "Fraction of tweets that are retweets"),
        FeatureInfo("frac_tweets_with_spam_word", "content", "fraction", "Fraction of tweets containing a common spam word"),
        FeatureInfo("frac_duplicate_tweets", "content", "fraction", "Fraction of tweets that repeat another tweet (ignoring URLs, mentions and numbers)"),
        FeatureInfo("frac_third_party_client", "content", "fraction", "Fraction of tweets posted from a non-official (third-party or automated) client"),
        FeatureInfo("n_distinct_sources", "content", "count", "Number of different clients used to post", integer=True),
        FeatureInfo("n_tweets_collected", "content", "count", "Number of recent tweets analysed (up to 200)", integer=True),
        FeatureInfo("distinct_mentions_per_tweet", "content", "per tweet", "Distinct users mentioned, divided by tweets analysed"),
        # ---- user behaviour
        FeatureInfo("followers_count", "behaviour", "count", "Number of followers", integer=True),
        FeatureInfo("followees_count", "behaviour", "count", "Number of accounts followed", integer=True),
        FeatureInfo("follower_followee_ratio", "behaviour", "ratio", "Followers divided by followees"),
        FeatureInfo("statuses_count", "behaviour", "count", "Total tweets posted over the account's life", integer=True),
        FeatureInfo("favourites_count", "behaviour", "count", "Tweets the account has liked", integer=True),
        FeatureInfo("listed_count", "behaviour", "count", "Public lists that include the account", integer=True),
        FeatureInfo("account_age_days", "behaviour", "days", "Account age at its most recent analysed tweet"),
        FeatureInfo("tweets_per_day_lifetime", "behaviour", "per day", "Total tweets divided by account age"),
        FeatureInfo("tweets_per_day_recent", "behaviour", "per day", "Analysed tweets divided by the days they span"),
        FeatureInfo("max_tweets_in_a_day", "behaviour", "count", "Most tweets posted on a single day", integer=True),
        FeatureInfo("max_tweets_in_a_week", "behaviour", "count", "Most tweets posted in a single week", integer=True),
    ]
    for s in STATS:
        feats.append(FeatureInfo(
            f"inter_tweet_hours_{s}", "behaviour", "hours", f"{s.capitalize()} time between consecutive tweets",
        ))
    feats += [
        FeatureInfo("default_profile", "profile", "yes/no", "Profile theme never changed from the default", binary=True),
        FeatureInfo("default_profile_image", "profile", "yes/no", "Uses the default profile image", binary=True),
        FeatureInfo("has_description", "profile", "yes/no", "Profile has a description", binary=True),
        FeatureInfo("description_length", "profile", "characters", "Length of the profile description", integer=True),
        FeatureInfo("name_length", "profile", "characters", "Length of the display name", integer=True),
        FeatureInfo("screen_name_length", "profile", "characters", "Length of the username", integer=True),
        FeatureInfo("screen_name_digits", "profile", "count", "Digits in the username", integer=True),
        FeatureInfo("has_url", "profile", "yes/no", "Profile links to a website", binary=True),
        FeatureInfo("geo_enabled", "profile", "yes/no", "Location tagging enabled", binary=True),
        FeatureInfo("verified", "profile", "yes/no", "Account is verified", binary=True),
    ]
    return feats


FEATURES: list[FeatureInfo] = _registry()
FEATURE_NAMES: list[str] = [f.name for f in FEATURES]

# Optional 65th feature from the LLM extension (L1): out-of-fold spam probability of
# the account's English text from the fine-tuned text encoder.
TEXT_FEATURE = FeatureInfo(
    "text_spam_score", "content", "probability",
    "Spam probability of the account's English text, from the fine-tuned text encoder",
)
FEATURE_INFO: dict[str, FeatureInfo] = {f.name: f for f in [*FEATURES, TEXT_FEATURE]}


# ------------------------------------------------------------------ helpers
def to_bool(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower().isin({"1", "1.0", "true", "t", "yes"}).astype(int)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _nonzero_id(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").fillna(0)
    return (v > 0).astype(int)


def _normalise_text(text: pd.Series) -> pd.Series:
    t = text.str.lower().str.replace(URL_RE, "", regex=True)
    t = t.str.replace(MENTION_RE, "", regex=True).str.replace(r"\d+", "", regex=True)
    t = t.str.replace(r"^rt\s*:?\s*", "", regex=True)
    return t.str.replace(r"\s+", " ", regex=True).str.strip()


def _source_name(source: pd.Series) -> pd.Series:
    """'<a href=...>Twitter for iPhone</a>' -> 'twitter for iphone'."""
    return source.fillna("").astype(str).str.replace(r"<[^>]+>", "", regex=True).str.strip().str.lower()


# ------------------------------------------------------------------ tweets
def tweet_level(tweets: pd.DataFrame) -> pd.DataFrame:
    """Per-tweet counts and flags (vectorised)."""
    text = tweets["text"].fillna("").astype(str)
    out = pd.DataFrame({"user_id": tweets["user_id"].astype(str), "time": tweets["time"]})
    out["hashtags"] = text.str.count(HASHTAG_RE)
    out["urls"] = text.str.count(URL_RE)
    out["mentions"] = text.str.count(MENTION_RE)
    out["chars"] = text.str.len()
    out["words"] = text.str.split().str.len().fillna(0)
    out["digits"] = text.str.count(DIGIT_RE)
    out["has_url"] = (out["urls"] > 0).astype(int)
    out["has_hashtag"] = (out["hashtags"] > 0).astype(int)
    out["has_mention"] = (out["mentions"] > 0).astype(int)
    out["retweets_received"] = to_num(tweets.get("retweet_count", pd.Series(0, index=tweets.index))).fillna(0)
    is_rt_text = text.str.match(r"^RT @\w+")
    rt_id = tweets["retweeted_status_id"] if "retweeted_status_id" in tweets else pd.Series(0, index=tweets.index)
    out["is_retweet"] = (_nonzero_id(rt_id).astype(bool) | is_rt_text).astype(int)
    rp_id = tweets["in_reply_to_status_id"] if "in_reply_to_status_id" in tweets else pd.Series(0, index=tweets.index)
    out["is_reply"] = _nonzero_id(rp_id)
    out["spam_word"] = text.str.contains(SPAM_RE, regex=True).astype(int)
    src = _source_name(tweets["source"]) if "source" in tweets else pd.Series("", index=tweets.index)
    out["source"] = src
    out["third_party"] = (~src.isin(OFFICIAL_CLIENTS) & (src != "")).astype(int)
    out["norm_text"] = _normalise_text(text)
    out["mention_handles"] = text.str.lower().str.findall(MENTION_RE)
    return out


def content_and_activity(tweets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-tweet values into account-level content and activity features."""
    tl = tweet_level(tweets)
    g = tl.groupby("user_id", sort=False)
    parts = []
    for key in PER_TWEET_COUNTS:
        agg = g[key].agg(STATS)
        agg.columns = [f"{key}_per_tweet_{s}" for s in STATS]
        parts.append(agg)
    out = pd.concat(parts, axis=1)

    out["frac_tweets_with_url"] = g["has_url"].mean()
    out["frac_tweets_with_hashtag"] = g["has_hashtag"].mean()
    out["frac_tweets_with_mention"] = g["has_mention"].mean()
    out["frac_replies"] = g["is_reply"].mean()
    out["frac_retweets"] = g["is_retweet"].mean()
    out["frac_tweets_with_spam_word"] = g["spam_word"].mean()
    out["frac_third_party_client"] = g["third_party"].mean()
    out["n_distinct_sources"] = g["source"].nunique()
    n = g.size()
    out["n_tweets_collected"] = n
    nonempty = tl[tl["norm_text"] != ""]
    dup = nonempty.duplicated(subset=["user_id", "norm_text"], keep="first").groupby(nonempty["user_id"]).sum()
    out["frac_duplicate_tweets"] = (dup.reindex(out.index).fillna(0) / n).astype(float)
    mentions = tl[["user_id", "mention_handles"]].explode("mention_handles").dropna()
    distinct = mentions.groupby("user_id")["mention_handles"].nunique()
    out["distinct_mentions_per_tweet"] = (distinct.reindex(out.index).fillna(0) / n).astype(float)

    # ---- activity over time (behaviour)
    timed = tl.dropna(subset=["time"]).sort_values(["user_id", "time"])
    hours = timed.groupby("user_id")["time"].diff().dt.total_seconds() / 3600.0
    gaps = pd.DataFrame({"user_id": timed["user_id"], "h": hours}).dropna()
    gstats = gaps.groupby("user_id")["h"].agg(STATS)
    gstats.columns = [f"inter_tweet_hours_{s}" for s in STATS]
    out = out.join(gstats)
    span = timed.groupby("user_id")["time"].agg(["min", "max"])
    span_days = ((span["max"] - span["min"]).dt.total_seconds() / 86400.0).clip(lower=1.0)
    out["tweets_per_day_recent"] = (n / span_days.reindex(out.index)).astype(float)
    day = timed["time"].dt.floor("D")
    out["max_tweets_in_a_day"] = timed.groupby([timed["user_id"], day]).size().groupby(level=0).max()
    week = timed["time"].dt.tz_localize(None).dt.to_period("W")
    out["max_tweets_in_a_week"] = timed.groupby([timed["user_id"], week]).size().groupby(level=0).max()
    out["_last_tweet_time"] = span["max"]
    out.index.name = "user_id"
    return out


# ------------------------------------------------------------------ users
def user_level(users: pd.DataFrame) -> pd.DataFrame:
    u = users.copy()
    u["user_id"] = u["id"].astype(str)
    out = pd.DataFrame({"user_id": u["user_id"]})
    out["followers_count"] = to_num(u["followers_count"])
    out["followees_count"] = to_num(u["friends_count"])
    out["follower_followee_ratio"] = out["followers_count"] / out["followees_count"].clip(lower=1)
    out["statuses_count"] = to_num(u["statuses_count"])
    out["favourites_count"] = to_num(u.get("favourites_count"))
    out["listed_count"] = to_num(u.get("listed_count"))
    from .data.cresci import parse_time

    out["_created_at"] = parse_time(u["created_at"], u.get("timestamp"))
    desc = u.get("description", pd.Series("", index=u.index)).fillna("").astype(str)
    name = u.get("name", pd.Series("", index=u.index)).fillna("").astype(str)
    sname = u["screen_name"].fillna("").astype(str)
    out["default_profile"] = to_bool(u.get("default_profile", pd.Series("", index=u.index)))
    out["default_profile_image"] = to_bool(u.get("default_profile_image", pd.Series("", index=u.index)))
    out["has_description"] = (desc.str.strip() != "").astype(int)
    out["description_length"] = desc.str.len()
    out["name_length"] = name.str.len()
    out["screen_name_length"] = sname.str.len()
    out["screen_name_digits"] = sname.str.count(r"\d")
    url = u.get("url", pd.Series("", index=u.index)).fillna("").astype(str).str.strip()
    out["has_url"] = (~url.str.lower().isin({"", "nan", "null", "none"})).astype(int)
    out["geo_enabled"] = to_bool(u.get("geo_enabled", pd.Series("", index=u.index)))
    out["verified"] = to_bool(u.get("verified", pd.Series("", index=u.index)))
    for meta in ("group", "label", "screen_name"):
        if meta in u:
            out[meta] = u[meta].values
    return out.set_index("user_id")


def build_account_features(users: pd.DataFrame, tweets: pd.DataFrame) -> pd.DataFrame:
    """Join user-level and tweet-derived features; one row per account with tweets."""
    ul = user_level(users)
    ca = content_and_activity(tweets)
    df = ul.join(ca, how="inner")
    age = (df["_last_tweet_time"] - df["_created_at"]).dt.total_seconds() / 86400.0
    df["account_age_days"] = age.clip(lower=1.0)
    df["tweets_per_day_lifetime"] = df["statuses_count"] / df["account_age_days"]
    df = df.drop(columns=["_last_tweet_time", "_created_at"])
    meta = [c for c in ("group", "label", "screen_name") if c in df]
    df = df[meta + FEATURE_NAMES]
    df[FEATURE_NAMES] = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).astype(float)
    return df


def data_dictionary(train: pd.DataFrame | None = None) -> pd.DataFrame:
    """Name, category, unit, description and (from training data) observed range."""
    rows = []
    for f in FEATURES:
        row = {"feature": f.name, "category": f.category, "unit": f.unit, "description": f.description,
               "integer": f.integer, "binary": f.binary}
        if train is not None and f.name in train:
            col = train[f.name].dropna()
            row.update(
                train_min=float(col.min()), train_p01=float(col.quantile(0.01)),
                train_median=float(col.median()), train_p99=float(col.quantile(0.99)),
                train_max=float(col.max()), missing_rate=float(train[f.name].isna().mean()),
            )
        rows.append(row)
    return pd.DataFrame(rows)
