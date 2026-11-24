"""Add a time-series summary context to data/forecast.parquet from the cached pre-t price histories.

  state_ts  state + a summary of the Yes price over the 7 days up to t (daily prices, range, change,
            recent moves). Every number comes from the hourly history fetched up to t, nothing later.
Also adds numeric columns used by the price-only baselines (ts_*).
"""
import json
import os

import numpy as np
import pandas as pd

from pipeline.build_dataset import price_at
from pipeline.candidates import ROOT


def summary(hist, t):
    ts = t.timestamp()
    recent = [p for p in hist if ts - 7 * 86400 <= p["t"] <= ts]
    daily = [price_at(hist, t - pd.Timedelta(days=k)) for k in range(7, -1, -1)]  # t-7d ... t
    prices = [p["p"] for p in recent]
    last24 = [p["p"] for p in hist if ts - 86400 <= p["t"] <= ts]
    moves = lambda xs: int(np.sum(np.diff(xs) != 0)) if len(xs) > 1 else 0  # noqa: E731
    feats = {
        "ts_low": min(prices) if prices else np.nan, "ts_high": max(prices) if prices else np.nan,
        "ts_change_7d": daily[-1] - daily[0] if not np.isnan(daily[0]) else 0.0,
        "ts_change_1d": daily[-1] - daily[-2] if not np.isnan(daily[-2]) else 0.0,
        "ts_moves_1d": moves(last24), "ts_moves_7d": moves(prices),
        "ts_std_7d": float(np.std(prices)) if prices else 0.0,
    }
    fmt = lambda x: "n/a" if np.isnan(x) else f"{x:.2f}"  # noqa: E731
    text = ("Price history of Yes before now:\n"
            f"- daily, oldest to newest: {', '.join(fmt(x) for x in daily)}\n"
            f"- 7-day low {fmt(feats['ts_low'])}, high {fmt(feats['ts_high'])}, change {feats['ts_change_7d']:+.2f}\n"
            f"- last 24 hours: change {feats['ts_change_1d']:+.2f}, {feats['ts_moves_1d']} price moves\n"
            f"- price moves in the last 7 days: {feats['ts_moves_7d']}")
    return text, feats


def main():
    path = os.path.join(ROOT, "data/forecast.parquet")
    d = pd.read_parquet(path)
    hist = {}
    for line in open(os.path.join(ROOT, "data/raw/prices.jsonl")):
        r = json.loads(line)
        hist[(r["market_id"], r["horizon_days"], r["t"])] = r["history"]
    out = [summary(hist[(m, h, int(t.timestamp()))], t) for m, h, t in zip(d.market_id, d.horizon_days, d.t)]
    d["state_ts"] = [s + "\n" + text for s, (text, _) in zip(d.state, out)]
    d = pd.concat([d.drop(columns=[c for c in d.columns if c.startswith("ts_")]),
                   pd.DataFrame([f for _, f in out], index=d.index)], axis=1)
    tmp = path + ".tmp"
    d.to_parquet(tmp)
    os.replace(tmp, path)
    print(d.state_ts.iloc[0].split("\n", 4)[-1])
    print(d[[c for c in d.columns if c.startswith("ts_")]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
