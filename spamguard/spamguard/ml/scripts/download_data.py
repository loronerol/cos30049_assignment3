"""Download and unpack Cresci-2017 and Cresci-2015 from the IU Bot Repository
(research use only).

The server limits each connection to ~15 KB/s, so the file is fetched as
parallel byte ranges (about 20 minutes at 24 connections instead of ~8 hours).

Usage:  python ml/scripts/download_data.py [--workers 24]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import tarfile
import urllib.request
import zipfile

import _common  # noqa: F401

from _common import log
from spamsentinel_ml.config import DATASET_URLS, GROUPS, RAW_ROOT
from spamsentinel_ml.data.cresci import extract_nested

CHUNK = 4_000_000


def fetch_range(url: str, start: int, end: int, attempts: int = 6) -> bytes:
    want = end - start + 1
    for _ in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=900) as r:
                data = r.read()
            if len(data) == want:
                return data
        except OSError:
            pass
    raise RuntimeError(f"failed to fetch bytes {start}-{end}")


def download(url: str, out, workers: int) -> None:
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        size = int(r.headers["Content-Length"])
    ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    log(f"downloading {out.name}: {size / 1e6:.0f} MB in {len(ranges)} chunks with {workers} connections")
    parts: dict[int, bytes] = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(fetch_range, url, s, e): s for s, e in ranges}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            parts[futures[fut]] = fut.result()
            if i % 10 == 0:
                log(f"{i}/{len(ranges)} chunks")
    tmp = out.with_suffix(".part")
    with open(tmp, "wb") as f:
        for s, _ in ranges:
            f.write(parts[s])
    tmp.rename(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    for name, url in DATASET_URLS.items():
        folder = RAW_ROOT / name
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / url.rsplit("/", 1)[1]
        if not out.exists():
            download(url, out, args.workers)
        if out.suffix == ".zip":
            with zipfile.ZipFile(out) as zf:
                bad = zf.testzip()
                if bad:
                    raise RuntimeError(f"corrupt member in archive: {bad}")
                zf.extractall(folder)
        else:
            with tarfile.open(out) as tf:
                tf.extractall(folder, filter="data")
    for ds in {d for d, _ in GROUPS.values()}:
        extract_nested(RAW_ROOT / ds)
    log(f"datasets ready in {RAW_ROOT}")


if __name__ == "__main__":
    main()
