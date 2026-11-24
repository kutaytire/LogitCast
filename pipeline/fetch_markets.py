"""Download every closed Polymarket market whose own endDate falls in [start, end).

Markets are fetched individually, not through closed events: a market resolves on its own,
while its event can stay open (e.g. "by Aug 31" resolves NO while "by Dec 31" still trades).
Fetching only closed events silently drops those markets and biases the sample toward YES.

One JSONL file per UTC day under data/raw/markets/, written atomically; a rerun skips finished
days. Each market carries its tags (include_tag) and its event(s). Filtering is in pipeline/candidates.py.
"""
import argparse
import datetime as dt
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

GAMMA = "https://gamma-api.polymarket.com/markets/keyset"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/raw/markets")
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "jev-forecast-research/0.1"


def get(params):
    for attempt in range(8):
        try:
            r = SESSION.get(GAMMA, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after retries: {params}")


def fetch_day(day):
    path = os.path.join(OUT, f"{day.isoformat()}.jsonl")
    if os.path.exists(path):
        return day, None
    params = {
        "closed": "true",
        "include_tag": "true",
        "limit": 100,
        "end_date_min": f"{day.isoformat()}T00:00:00Z",
        "end_date_max": f"{(day + dt.timedelta(days=1)).isoformat()}T00:00:00Z",
    }
    markets = []
    while True:
        page = get(params)
        markets += page["markets"]
        if not page.get("next_cursor") or not page["markets"]:
            break
        params["after_cursor"] = page["next_cursor"]
    # end_date_max is inclusive on the API side; keep each market on exactly one day.
    markets = [m for m in markets if (m.get("endDate") or "")[:10] == day.isoformat()]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for m in markets:
            f.write(json.dumps(m) + "\n")
    os.replace(tmp, path)
    return day, len(markets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-02-27")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    days = [start + dt.timedelta(days=i) for i in range((end - start).days)]
    with ThreadPoolExecutor(a.workers) as ex:
        for day, n in ex.map(fetch_day, days):
            print(day, "skipped" if n is None else f"{n} markets", flush=True)


if __name__ == "__main__":
    main()
