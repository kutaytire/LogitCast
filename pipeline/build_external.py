"""Extra yes/no training questions with written context: Halawi et al. (NeurIPS 2024, YuehHanChen/forecasting) and
the Metaculus questions of nikhilchandak/metaculus-binary that Halawi does not already contain (matched by
Metaculus question id).

Every question resolved by early 2025, so these rows are TRAINING data only (the backbone may know the outcomes);
the test sets stay the post-cutoff Polymarket, OpenForesight FutureSim 2026-Q2 and Manifold splits.
Forecast times follow pipeline/build_dataset.py: t = anchor - h, h in HORIZONS_DAYS, anchor = the earlier of close and
resolution, and t must fall after the question opened. The context is the date, question, close date, background
and resolution rules, plus the community forecast at t when Halawi has one dated before day(t) (metaculus-binary
has no crowd series). Halawi's validation split is val, everything else train.
Writes data/external.parquet with the columns logitcast/train_forecast.py expects (state_context is the context).
"""
import json
import os
import re

import numpy as np
import pandas as pd

from pipeline.candidates import HORIZONS_DAYS, ROOT

RAW = os.path.join(ROOT, "data/raw/external")
TEXT_CHARS = 1500
NO_RULES = "Not applicable/available for this question."  # Halawi placeholder for a missing field


def qid(url):
    m = re.search(r"/questions/(\d+)", str(url))
    return m[1] if m else None


def load():
    h = pd.concat([pd.DataFrame(json.load(open(os.path.join(RAW, f"halawi_{s}.json")))).assign(part=s)
                   for s in ("train", "validation", "test")], ignore_index=True)
    m = pd.read_parquet(os.path.join(RAW, "metaculus_binary.parquet")).assign(part="train", community_predictions=None)
    seen = set(h[h.data_source == "metaculus"].url.map(qid))
    m = m[~m.url.map(qid).isin(seen)]
    d = pd.concat([h.assign(origin="halawi"), m.assign(origin="metaculus_binary")], ignore_index=True)
    d["y"] = d.resolution.astype(float).astype(int)
    return d


def crowd_at(series, t):
    """Last community forecast dated before day(t), or NaN (dates are days, so the same day is excluded). The series
    is stored as a JSON string of [day, P(Yes)] pairs; entries without a single probability are skipped."""
    pts = [p for day, p in json.loads(series or "[]")
           if isinstance(p, (int, float)) and 0 <= p <= 1 and pd.Timestamp(day, tz="UTC") < t.normalize()]
    return float(pts[-1]) if pts else np.nan


def text(s):
    return " ".join(str(s or "").split())[:TEXT_CHARS]


def main():
    d = load()
    rows = []
    for r in d.itertuples():
        begin = pd.Timestamp(r.date_begin, tz="UTC")
        anchor = min(pd.Timestamp(r.date_close, tz="UTC"), pd.Timestamp(r.date_resolve_at, tz="UTC"))
        for h in HORIZONS_DAYS:
            t = anchor - pd.Timedelta(days=h)
            if t <= begin:
                continue
            p = crowd_at(r.community_predictions, t)
            lines = [f"Today is {t:%A, %B %-d, %Y}.", f"Event: {r.question}", f"The question closes {anchor:%B %-d, %Y}."]
            if text(r.background) and r.background != NO_RULES:
                lines.append(f"Background: {text(r.background)}")
            if text(r.resolution_criteria) and r.resolution_criteria != NO_RULES:
                lines.append(f"Resolution rules: {text(r.resolution_criteria)}")
            if not np.isnan(p):
                lines.append(f"Community forecast of Yes now: {p:.2f}.")
            rows.append({"market_id": f"{r.origin}:{qid(r.url) or r.url}", "event_id": f"{r.origin}:{qid(r.url) or r.url}",
                         "split": "val" if r.part == "validation" else "train", "horizon_days": h, "t": t, "end": anchor,
                         "category": r.data_source, "origin": r.origin, "y": r.y, "p_t": p, "question": r.question,
                         "state_context": "\n".join(lines)})
    o = pd.DataFrame(rows)
    o.to_parquet(os.path.join(ROOT, "data/external.parquet"))
    print(o.groupby(["origin", "split"]).agg(rows=("y", "size"), questions=("event_id", "nunique"), yes=("y", "mean"),
                                               crowd=("p_t", lambda s: s.notna().mean())).round(3).to_string())
    print("by source:", o.category.value_counts().to_dict())
    print(o[o.p_t.notna()].state_context.iloc[0][:900])


if __name__ == "__main__":
    main()
