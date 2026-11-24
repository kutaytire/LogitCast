"""Join candidates with pre-t prices and write data/forecast.parquet (one row per market x horizon).

Kept only if the market traded within STALE before t and its price at t is not already
near-certain (PRICE_BAND). Both filters use information available at t only.

activity_7d counts price changes in the 7 days up to t. It is the liquidity measure used for
analysis because it uses only pre-t data (final market volume includes trading after t).

Each row carries two contexts for System 1 (kind "noul", question = the market question):
  state        - date, event, scheduled close, rules
  state_price  - the same plus the market's YES price at t, t-1d, t-7d
"""
import json
import os

import numpy as np
import pandas as pd

from pipeline.candidates import CONTEXT_FIELDS, ROOT

STALE = pd.Timedelta(hours=24)
PRICE_BAND = (0.02, 0.98)
RULES_CHARS = 1500


def price_at(hist, t):
    """Last price at or before t, or NaN."""
    pts = [p for p in hist if p["t"] <= t.timestamp()]
    return pts[-1]["p"] if pts else np.nan


def state_text(r):
    lines = [f"Today is {r.t:%A, %B %-d, %Y, %H:%M} UTC.", f"Event: {r.event_title}"]
    if isinstance(r.market_groupItemTitle, str) and r.market_groupItemTitle not in r.market_question:
        lines.append(f"Option: {r.market_groupItemTitle}")
    lines.append(f"The market closes {r.end:%B %-d, %Y, %H:%M} UTC.")
    rules = " ".join((r.market_description or "").split())
    lines.append(f"Resolution rules: {rules[:RULES_CHARS]}")
    return "\n".join(lines)


def price_text(r):
    return (f"Market price of Yes now: {r.p_t:.2f}; one day ago: {r.p_1d:.2f}; "
            f"seven days ago: {r.p_7d:.2f}.")


def main():
    cand = pd.read_parquet(os.path.join(ROOT, "data/candidates.parquet"))
    hist = {}
    for line in open(os.path.join(ROOT, "data/raw/prices.jsonl")):
        d = json.loads(line)
        hist[(d["market_id"], d["horizon_days"], d["t"])] = d["history"]
    cand["key"] = [(m, h, int(t.timestamp())) for m, h, t in zip(cand.market_id, cand.horizon_days, cand.t)]
    cand = cand[cand.key.isin(hist)].copy()

    def feats(r):
        h = hist[r.key]
        last_t = max((p["t"] for p in h), default=None)
        recent = [p["p"] for p in h if r.t.timestamp() - 7 * 86400 <= p["t"] <= r.t.timestamp()]
        return pd.Series({
            "activity_7d": int(np.sum(np.diff(recent) != 0)) if len(recent) > 1 else 0,
            "p_t": price_at(h, r.t),
            "p_1d": price_at(h, r.t - pd.Timedelta(days=1)),
            "p_7d": price_at(h, r.t - pd.Timedelta(days=7)),
            "last_trade_age_h": (r.t.timestamp() - last_t) / 3600 if last_t else np.nan,
        })

    cand = pd.concat([cand, cand.apply(feats, axis=1)], axis=1)
    n0 = len(cand)
    cand = cand[cand.last_trade_age_h <= STALE / pd.Timedelta(hours=1)]
    cand = cand[cand.p_t.between(*PRICE_BAND)]
    for c in ("p_1d", "p_7d"):  # no trade that far back (young market): fall back to p_t
        cand[c] = cand[c].fillna(cand.p_t)
    print(f"kept {len(cand)} / {n0} after staleness and price-band filters")

    cand["state"] = cand.apply(state_text, axis=1)
    cand["state_price"] = cand.state + "\n" + cand.apply(price_text, axis=1)
    cand["question"] = cand.market_question
    keep = ["market_id", "event_id", "split", "horizon_days", "category", "volume",
            "activity_7d", "t", "end", "closed", "y", "p_t", "p_1d", "p_7d", "state", "state_price", "question"]
    assert set(f"{s}_{k}" for s, ks in CONTEXT_FIELDS.items() for k in ks) <= set(cand.columns)
    cand[keep].reset_index(drop=True).to_parquet(os.path.join(ROOT, "data/forecast.parquet"))
    print(cand.groupby(["split", "horizon_days"]).size().unstack())
    print(cand.groupby(["split", "category"]).size().unstack(0).fillna(0).astype(int))
    print("YES rate:", cand.groupby("split").y.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
