"""Fetch the YES-token hourly price history for the 8 days up to each candidate's forecast time t.

Nothing after t is requested, so the stored series cannot leak the outcome.
Appends to data/raw/prices.jsonl; a rerun skips (market_id, horizon_days, t) keys already present.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data/raw/prices.jsonl")
CLOB = "https://clob.polymarket.com/prices-history"
WINDOW = pd.Timedelta(days=8)
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "jev-forecast-research/0.1"


def fetch(row):
    end = int(row.t.timestamp())
    params = {"market": row.yes_token, "startTs": int((row.t - WINDOW).timestamp()), "endTs": end, "fidelity": 60}
    for attempt in range(8):
        try:
            r = SESSION.get(CLOB, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            hist = [p for p in r.json()["history"] if p["t"] <= end]
            return {"market_id": row.market_id, "horizon_days": int(row.horizon_days), "t": end, "history": hist}
        time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after retries: {params}")


def main():
    cand = pd.read_parquet(os.path.join(ROOT, "data/candidates.parquet"))
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            d = json.loads(line)
            done.add((d["market_id"], d["horizon_days"], d["t"]))
    todo = [r for r in cand.itertuples() if (r.market_id, r.horizon_days, int(r.t.timestamp())) not in done]
    print(f"{len(done)} done, {len(todo)} to fetch", flush=True)
    with open(OUT, "a") as f, ThreadPoolExecutor(8) as ex:
        for i, d in enumerate(ex.map(fetch, todo), 1):
            f.write(json.dumps(d) + "\n")
            if i % 1000 == 0:
                f.flush()
                print(i, flush=True)


if __name__ == "__main__":
    main()
