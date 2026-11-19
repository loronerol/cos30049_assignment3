"""Load Cresci-2017 / Cresci-2015 account groups (users.csv + tweets.csv per group)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from ..config import GROUP_LABELS, GROUPS, MAX_TWEETS_PER_ACCOUNT, RAW_DIR, RAW_ROOT

USER_COLS = [
    "id", "name", "screen_name", "statuses_count", "followers_count", "friends_count",
    "favourites_count", "listed_count", "url", "default_profile", "default_profile_image",
    "geo_enabled", "verified", "protected", "description", "created_at", "timestamp",
    "crawled_at", "updated",
]
TWEET_COLS = [
    "id", "text", "source", "user_id", "in_reply_to_status_id", "retweeted_status_id",
    "retweet_count", "timestamp", "created_at",
]


def extract_nested(raw_dir: Path = RAW_DIR) -> None:
    """The archive holds one zip per group; unpack any that are not yet extracted."""
    for z in raw_dir.rglob("*.csv.zip"):
        if z.name.startswith("cresci-"):
            continue  # the top-level archive itself
        target = z.with_suffix("")  # <group>.csv
        if (z.parent / target.name / "users.csv").exists() or (target / "users.csv").exists():
            continue
        with zipfile.ZipFile(z) as zf:
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
            # Cresci-2017 zips contain a <group>.csv/ folder; Cresci-2015 zips are flat.
            flat = all("/" not in m for m in members)
            zf.extractall(target if flat else z.parent, members=members)


def group_dir(group: str, raw_root: Path = RAW_ROOT) -> Path:
    base = raw_root / GROUPS[group][0]
    hits = [p.parent for p in base.rglob("users.csv") if p.parent.name in (group, group + ".csv")]
    if not hits:
        raise FileNotFoundError(
            f"users.csv for group {group!r} not found under {base}. "
            "Run ml/scripts/download_data.py first."
        )
    return hits[0]


def _read_csv(path: Path, usecols: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0, encoding="utf-8", encoding_errors="replace").columns
    cols = [c for c in usecols if c in header]
    return pd.read_csv(
        path,
        usecols=cols,
        dtype=str,
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="skip",
        na_values=["NULL"],  # Cresci-2017 writes missing values as the string NULL
        low_memory=False,
    )


def load_users(group: str, raw_root: Path = RAW_ROOT) -> pd.DataFrame:
    df = _read_csv(group_dir(group, raw_root) / "users.csv", USER_COLS)
    df["group"] = group
    df["label"] = GROUP_LABELS[group]
    return df


def parse_time(created_at: pd.Series, timestamp: pd.Series | None = None) -> pd.Series:
    """Twitter's 'Fri Feb 13 18:53:43 +0000 2009' format, or epoch milliseconds such as
    '1183552203000L' (traditional_spambots_1), falling back to the local `timestamp` column."""
    created_at = created_at.astype("string")
    t = pd.to_datetime(created_at, format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
    epoch = created_at.str.fullmatch(r"\d{10,13}L?", na=False)
    if epoch.any():
        ms = pd.to_numeric(created_at[epoch].str.rstrip("L"), errors="coerce")
        t[epoch] = pd.to_datetime(ms, unit="ms", utc=True)
    if timestamp is not None:
        alt = pd.to_datetime(timestamp, errors="coerce", utc=True)
        t = t.fillna(alt)
    return t


def load_tweets(group: str, raw_root: Path = RAW_ROOT, max_per_user: int = MAX_TWEETS_PER_ACCOUNT) -> pd.DataFrame:
    """All tweets of a group, keeping each account's `max_per_user` most recent."""
    path = group_dir(group, raw_root) / "tweets.csv"
    if not path.exists():
        return pd.DataFrame(columns=TWEET_COLS + ["time"])
    df = _read_csv(path, TWEET_COLS)
    df = df.dropna(subset=["user_id"])
    df = df[df["user_id"].str.fullmatch(r"\d+", na=False)]
    df["time"] = parse_time(df.get("created_at"), df.get("timestamp"))
    df = df.drop_duplicates(subset=["id"]) if "id" in df else df
    df = df.sort_values(["user_id", "time"], ascending=[True, False])
    df = df.groupby("user_id", sort=False).head(max_per_user)
    return df.reset_index(drop=True)
