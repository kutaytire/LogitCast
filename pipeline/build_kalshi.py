"""Kalshi binary markets: a cross-platform test set (split="kalshi"), built like pipeline/build_manifold.py.

Leakage rules:
  1. Forecast times are anchored on the open time: t = open + h, h in HORIZONS. Kalshi's close time is moved to the
     moment a market resolves early (can_close_early), so anchoring on it would leak the outcome.
  2. t >= TRAIN_LABEL_END (Aug 1 2026), after every outcome the fine-tuned models trained on, and the market
     settled by SNAPSHOT.
  3. The market must still be open at t, and its probability at t is the bid/ask midpoint of the last hourly
     candle at or before t. Nothing after t is used.
  4. The context is the date, the market question and its primary rules (fixed at listing). The close and
     expiration times are not shown.
Filters: binary, settled yes or no, not a multi-leg combo, at least MIN_VOLUME contracts traded, a trade in the 7
days before t, a spread of at most MAX_SPREAD at t, probability at t in (0.02, 0.98). At most PER_EVENT markets per
event (the most traded), since many events are ladders of strikes on one quantity, and PER_SERIES markets per series
(a seeded sample), since most of the listing is a few sports series of game, spread and quarter markets.

Writes data/kalshi.parquet with the columns logitcast/eval_forecast.py expects.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from pipeline.candidates import ROOT

API = "https://api.elections.kalshi.com/trade-api/v2"
RAW = os.path.join(ROOT, "data/raw/kalshi")
TRAIN_LABEL_END = pd.Timestamp("2026-08-01", tz="UTC")
SNAPSHOT = pd.Timestamp("2026-11-01", tz="UTC")
HORIZONS = (3, 10, 30, 90)  # days after opening
MIN_VOLUME = 1000
MAX_SPREAD = 0.10
PER_EVENT = 3
PER_SERIES = 3
RULES_CHARS = 1500
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
    raise RuntimeError(f"kalshi request failed: {path} {params}")


def ts(x):
    return pd.Timestamp(x).tz_convert("UTC") if pd.Timestamp(x).tzinfo else pd.Timestamp(x, tz="UTC")


MARKET_FIELDS = ("ticker", "title", "market_type", "result", "mve_collection_ticker", "volume_fp", "open_time",
                 "close_time", "settlement_ts", "rules_primary")


def compact(e):
    """The fields this builder uses, or None for a multi-leg combo event (most of Kalshi's listing)."""
    if e["event_ticker"].startswith("KXMVE"):
        return None
    return {"event_ticker": e["event_ticker"], "series_ticker": e["series_ticker"], "category": e.get("category"),
            "markets": [{k: m.get(k) for k in MARKET_FIELDS} for m in e.get("markets", [])]}


def list_events():
    """Settled events (markets nested) that closed after TRAIN_LABEL_END, compacted (cached), streamed."""
    path = os.path.join(RAW, "events.jsonl")
    if os.path.exists(path):
        return (json.loads(l) for l in open(path))
    os.makedirs(RAW, exist_ok=True)
    cursor, n = None, 0
    with open(path + ".tmp", "w") as f:
        while True:
            page = get("events", status="settled", limit=200, with_nested_markets="true",
                       min_close_ts=int(TRAIN_LABEL_END.timestamp()), **({"cursor": cursor} if cursor else {}))
            for e in filter(None, map(compact, page["events"])):
                f.write(json.dumps(e) + "\n")
            n += len(page["events"])
            cursor = page.get("cursor")
            if not cursor or not page["events"]:
                break
    os.replace(path + ".tmp", path)
    print(f"listed {n} settled events", flush=True)
    return (json.loads(l) for l in open(path))


def candidates(events):
    rows = []
    for e in events:
        ms = [m for m in e.get("markets", [])
              if m.get("market_type") == "binary" and m.get("result") in ("yes", "no")
              and not m.get("mve_collection_ticker") and float(m.get("volume_fp") or 0) >= MIN_VOLUME]
        for m in sorted(ms, key=lambda m: -float(m["volume_fp"]))[:PER_EVENT]:
            opened, closed = ts(m["open_time"]), ts(m["close_time"])
            settled = ts(m["settlement_ts"]) if m.get("settlement_ts") else closed
            for h in HORIZONS:
                t = (opened + pd.Timedelta(days=h)).ceil("h")
                if TRAIN_LABEL_END <= t < closed and settled <= SNAPSHOT:
                    rows.append({"market_id": m["ticker"], "event_id": e["event_ticker"], "series": e["series_ticker"],
                                 "split": "kalshi", "horizon_days": h, "t": t, "closed": closed,
                                 "y": int(m["result"] == "yes"), "question": m["title"],
                                 "rules": " ".join((m.get("rules_primary") or "").split())[:RULES_CHARS],
                                 "category": e.get("category") or "", "volume": float(m["volume_fp"])})
    c = pd.DataFrame(rows)
    markets = c.drop_duplicates("market_id")[["market_id", "series"]]
    keep = markets.groupby("series").sample(frac=1, random_state=0).groupby("series").head(PER_SERIES).market_id
    return c[c.market_id.isin(set(keep))].reset_index(drop=True)


def candles(series, ticker, t):
    """Hourly candles of a market from t - 8 days to t (cached per market and t)."""
    path = os.path.join(RAW, "candles", f"{ticker}_{int(t.timestamp())}.json")
    if os.path.exists(path):
        return json.load(open(path))
    c = get(f"series/{series}/markets/{ticker}/candlesticks", period_interval=60,
            start_ts=int((t - pd.Timedelta(days=8)).timestamp()), end_ts=int(t.timestamp()))["candlesticks"]
    c = [{"t": x["end_period_ts"], "bid": float(x["yes_bid"]["close_dollars"]),
          "ask": float(x["yes_ask"]["close_dollars"]), "vol": float(x.get("volume_fp") or 0)} for x in c]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(c, open(path, "w"))
    return c


def quote_at(c, t):
    """(midpoint, spread) from the last candle at or before t."""
    pts = [x for x in c if x["t"] <= t.timestamp()]
    if not pts:
        return np.nan, np.nan
    x = pts[-1]
    return (x["bid"] + x["ask"]) / 2, x["ask"] - x["bid"]


def main():
    events = list_events()
    c = candidates(events)
    print(f"{len(c)} candidate rows from {c.market_id.nunique()} markets", flush=True)
    with ThreadPoolExecutor(4) as ex:
        cs = list(ex.map(lambda r: candles(r.series, r.market_id, r.t), c.itertuples()))
    q = [quote_at(x, r.t) for x, r in zip(cs, c.itertuples())]
    c["p_t"], c["spread"] = [a for a, _ in q], [b for _, b in q]
    c["p_1d"] = [quote_at(x, r.t - pd.Timedelta(days=1))[0] for x, r in zip(cs, c.itertuples())]
    c["p_7d"] = [quote_at(x, r.t - pd.Timedelta(days=7))[0] for x, r in zip(cs, c.itertuples())]
    c["activity_7d"] = [sum(1 for y in x if y["vol"] > 0 and y["t"] >= (r.t - pd.Timedelta(days=7)).timestamp())
                        for x, r in zip(cs, c.itertuples())]
    c = c[(c.activity_7d > 0) & (c.spread <= MAX_SPREAD) & c.p_t.between(0.02, 0.98)].copy()
    for col in ("p_1d", "p_7d"):
        c[col] = c[col].fillna(c.p_t)
    c["state"] = [f"Today is {r.t:%A, %B %-d, %Y, %H:%M} UTC.\nEvent: {r.question}\nResolution rules: {r.rules}"
                  for r in c.itertuples()]
    c["state_price"] = [s + f"\nMarket price of Yes now: {r.p_t:.2f}; one day ago: {r.p_1d:.2f}; "
                            f"seven days ago: {r.p_7d:.2f}." for s, r in zip(c.state, c.itertuples())]
    c["state_json_price"] = [json.dumps({"today": f"{r.t:%Y-%m-%d %H:%M} UTC", "event": r.question,
                                         "resolution_rules": r.rules,
                                         "market": {"probability_yes_now": round(r.p_t, 2),
                                                    "probability_yes_one_day_ago": round(r.p_1d, 2),
                                                    "probability_yes_seven_days_ago": round(r.p_7d, 2)}},
                                        ensure_ascii=False) for r in c.itertuples()]
    c = c.drop(columns=["rules", "spread"]).reset_index(drop=True)
    c.to_parquet(os.path.join(ROOT, "data/kalshi.parquet"))
    print(f"kalshi: {len(c)} rows from {c.market_id.nunique()} markets, {c.event_id.nunique()} events, "
          f"{c.series.nunique()} series | YES rate {c.y.mean():.3f} | market Brier {np.mean((c.p_t - c.y) ** 2):.4f}")
    print(c.groupby("horizon_days").size().to_dict(), c.category.value_counts().to_dict())
    print(c.series.value_counts().head(15).to_dict())


if __name__ == "__main__":
    main()
