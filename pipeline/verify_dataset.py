"""Integrity checks on data/forecast.parquet; exits non-zero if a hard rule is violated.

Hard rules (assertions): cutoff, market open at t, split boundaries, train labels public
before any val/test forecast, one split per market, unique rows, t = end - h, price band,
no stale endDate (question deadline after endDate).
Diagnostics (printed): price orientation AUC, and YES-rate minus mean price at t by split and
by liquidity bin. A gap that grows with liquidity or differs sharply between splits points to
outcome-dependent selection.
"""
import os
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.candidates import CUTOFF, LAST_END, ROOT, TEST_START, VAL_START, stale_end


def main():
    d = pd.read_parquet(os.path.join(ROOT, "data/forecast.parquet"))
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    rules = {
        "t >= knowledge cutoff": (d.t >= CUTOFF).all(),
        "market open at t (t < closed)": (d.t < d.closed).all(),
        "train: end and closed before VAL_START": ((tr.end < VAL_START) & (tr.closed < VAL_START)).all(),
        "val: t >= VAL_START; end and closed before TEST_START":
            ((va.t >= VAL_START) & (va.end < TEST_START) & (va.closed < TEST_START)).all(),
        "test: t >= TEST_START; end before LAST_END": ((te.t >= TEST_START) & (te.end < LAST_END)).all(),
        "every train label public before the first val/test forecast": tr.closed.max() < min(va.t.min(), te.t.min()),
        "each market in exactly one split": d.groupby("market_id").split.nunique().max() == 1,
        "unique (market, horizon)": not d.duplicated(["market_id", "horizon_days"]).any(),
        "t = end - horizon, floored to the hour":
            ((d.end - pd.to_timedelta(d.horizon_days, unit="D")).dt.floor("h") == d.t).all(),
        "price at t inside (0.02, 0.98)": d.p_t.between(0.02, 0.98).all(),
        "no stale endDate": not any(stale_end(q, e) for q, e in zip(d.question, d.end)),
    }
    for k, ok in rules.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {k}")

    print("\nrows by split:", d.split.value_counts().to_dict())
    print("price orientation, AUC of p_t for y:",
          {s: round(roc_auc_score(g.y, g.p_t), 3) for s, g in d.groupby("split")})
    thr = tr.activity_7d.quantile(0.8)
    d["liquidity"] = pd.cut(d.activity_7d, [-1, tr.activity_7d.quantile(0.5), thr, float("inf")],
                            labels=["low", "mid", "high (liquid)"])
    gap = d.groupby(["split", "liquidity"], observed=True).apply(
        lambda g: pd.Series({"n": len(g), "yes": g.y.mean(), "mean_p": g.p_t.mean(), "gap": g.y.mean() - g.p_t.mean()}))
    print("\nYES rate minus mean market price at t:")
    print(gap.round(3).unstack(0).to_string())
    sys.exit(0 if all(rules.values()) else 1)


if __name__ == "__main__":
    main()
