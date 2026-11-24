"""Manifold binary markets: a cross-platform test set (split="manifold") plus train and val splits for tuning.

Leakage rules (Manifold differs from Polymarket):
  1. Forecast times are anchored on creation time, which never changes: t = created + h, h in HORIZONS.
     Manifold close times are editable by the creator and are moved when a market resolves early, so
     anchoring on the close would leak the outcome.
  2. Test rows have t >= TRAIN_LABEL_END (Aug 1 2026). The fine-tuned models learned Polymarket outcomes
     public before that date, and Manifold often asks about the same events. Train and val rows mirror the
     Polymarket training period: t >= CUTOFF (Feb 27 2026, the backbone's release) and resolved before
     TRAIN_LABEL_END; markets resolved in July are val (for calibration), earlier ones train.
  3. The market must be unresolved at t, and its crowd probability at t is taken from the last bet
     placed at or before t. Nothing after t is used.
  4. The context is the date and the question only. Descriptions and close dates are creator-editable
     and are not shown.
Filters: binary, resolved YES or NO, at least MIN_BETTORS unique bettors, a bet in the 7 days before t,
probability at t in (0.02, 0.98).

Writes data/manifold.parquet with splits train, val and manifold (test) and the columns logitcast/eval_forecast.py and
logitcast/train_forecast.py expect.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from pipeline.candidates import CUTOFF, ROOT

API = "https://api.manifold.markets/v0"
RAW = os.path.join(ROOT, "data/raw/manifold")
TRAIN_LABEL_END = pd.Timestamp("2026-08-01", tz="UTC")
VAL_START = pd.Timestamp("2026-07-01", tz="UTC")  # train markets resolve before this, val markets in July
SNAPSHOT = pd.Timestamp("2026-10-30", tz="UTC")
HORIZONS = (3, 10, 30, 90)  # days after creation
MIN_BETTORS = 20
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "jev-forecast-research/0.1"


def get(path, **params):
    for attempt in range(8):
        try:
            r = SESSION.get(f"{API}/{path}", params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        time.sleep(2 ** attempt)
    raise RuntimeError(f"manifold request failed: {path} {params}")


def ms(x):
    return pd.Timestamp(x, unit="ms", tz="UTC")


def list_markets():
    """All markets created after the earliest useful date, newest first (cached; an older cache is extended
    backwards from its oldest market)."""
    path = os.path.join(RAW, "markets.jsonl")
    earliest = CUTOFF - pd.Timedelta(days=max(HORIZONS))
    out = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    if out and ms(out[-1]["createdTime"]) < earliest:
        return out
    before = out[-1]["id"] if out else None
    while True:
        page = get("markets", limit=1000, **({"before": before} if before else {}))
        if not page:
            break
        out += page
        before = page[-1]["id"]
        if ms(page[-1]["createdTime"]) < earliest:
            break
    os.makedirs(RAW, exist_ok=True)
    with open(path + ".tmp", "w") as f:
        for m in out:
            f.write(json.dumps(m) + "\n")
    os.replace(path + ".tmp", path)
    return out


def bets_before(market_id, t):
    """Bets on a market placed at or before t, oldest first (cached per market and t)."""
    path = os.path.join(RAW, "bets", f"{market_id}_{int(t.timestamp())}.json")
    if os.path.exists(path):
        return json.load(open(path))
    bets, before = [], None
    while True:
        page = get("bets", contractId=market_id, limit=1000, **({"before": before} if before else {}))
        if not page:
            break
        bets += [b for b in page if b["createdTime"] <= t.timestamp() * 1000]
        before = page[-1]["id"]
        # pages are newest first; stop once they reach past t - 8 days (p_7d and activity_7d need that far)
        if page[-1]["createdTime"] < (t - pd.Timedelta(days=8)).timestamp() * 1000 or len(page) < 1000:
            break
    bets = sorted(({"t": b["createdTime"], "p": b["probAfter"]} for b in bets if "probAfter" in b),
                  key=lambda b: b["t"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(bets, open(path, "w"))
    return bets


def price_at(bets, t):
    pts = [b["p"] for b in bets if b["t"] <= t.timestamp() * 1000]
    return pts[-1] if pts else np.nan


def candidates(markets):
    rows = []
    for m in markets:
        if (m.get("outcomeType") != "BINARY" or m.get("resolution") not in ("YES", "NO")
                or (m.get("uniqueBettorCount") or 0) < MIN_BETTORS or not m.get("resolutionTime")):
            continue
        created, resolved = ms(m["createdTime"]), ms(m["resolutionTime"])
        for h in HORIZONS:
            t = (created + pd.Timedelta(days=h)).floor("h")
            if TRAIN_LABEL_END <= t < resolved <= SNAPSHOT:
                split = "manifold"
            elif CUTOFF <= t < resolved < TRAIN_LABEL_END:
                split = "val" if resolved >= VAL_START else "train"
            else:
                continue
            rows.append({"market_id": m["id"], "event_id": m["id"], "split": split, "horizon_days": h, "t": t,
                         "closed": resolved, "y": int(m["resolution"] == "YES"), "question": m["question"],
                         "volume": float(m.get("volume") or 0), "bettors": m["uniqueBettorCount"]})
    return pd.DataFrame(rows)


def main():
    markets = list_markets()
    c = candidates(markets)
    print(f"{len(markets)} markets listed, {len(c)} candidate rows from {c.market_id.nunique()} markets", flush=True)
    with ThreadPoolExecutor(4) as ex:
        bets = list(ex.map(lambda r: bets_before(r.market_id, r.t), c.itertuples()))
    c["p_t"] = [price_at(b, r.t) for b, r in zip(bets, c.itertuples())]
    c["p_1d"] = [price_at(b, r.t - pd.Timedelta(days=1)) for b, r in zip(bets, c.itertuples())]
    c["p_7d"] = [price_at(b, r.t - pd.Timedelta(days=7)) for b, r in zip(bets, c.itertuples())]
    c["activity_7d"] = [sum(1 for x in b if x["t"] >= (r.t - pd.Timedelta(days=7)).timestamp() * 1000)
                        for b, r in zip(bets, c.itertuples())]
    c = c[(c.activity_7d > 0) & c.p_t.between(0.02, 0.98)].copy()
    for col in ("p_1d", "p_7d"):
        c[col] = c[col].fillna(c.p_t)
    c["category"] = "manifold"
    c["state"] = [f"Today is {r.t:%A, %B %-d, %Y, %H:%M} UTC.\nEvent: {r.question}" for r in c.itertuples()]
    c["state_price"] = [s + f"\nMarket price of Yes now: {r.p_t:.2f}; one day ago: {r.p_1d:.2f}; "
                            f"seven days ago: {r.p_7d:.2f}." for s, r in zip(c.state, c.itertuples())]
    c["state_json_price"] = [json.dumps({"today": f"{r.t:%Y-%m-%d %H:%M} UTC", "event": r.question,
                                         "market": {"probability_yes_now": round(r.p_t, 2),
                                                    "probability_yes_one_day_ago": round(r.p_1d, 2),
                                                    "probability_yes_seven_days_ago": round(r.p_7d, 2)}},
                                        ensure_ascii=False) for r in c.itertuples()]
    c.reset_index(drop=True).to_parquet(os.path.join(ROOT, "data/manifold.parquet"))
    for split, g in c.groupby("split"):
        print(f"{split}: {len(g)} rows from {g.market_id.nunique()} markets | YES rate {g.y.mean():.3f} | "
              f"mean price {g.p_t.mean():.3f} | market Brier {np.mean((g.p_t - g.y) ** 2):.4f}")


if __name__ == "__main__":
    main()
