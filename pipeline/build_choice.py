"""Multiple-choice forecasting set: one question per Polymarket winner-take-all event ("Who will win ...?").

Leakage rules follow pipeline/candidates.py: t = scheduled end - h (h = 1, 7, 30 days), t >= CUTOFF, every option already
open at t and none closed, no option with a stale end date, splits by time. Option prices come from the same
pre-t price cache (fetch_prices.fetch); nothing after t is requested.
Events with more than MAX_OPTIONS options keep the MAX_OPTIONS - 1 options with the highest price at t plus
"Any other option" (its price is the sum of the rest), so the cut uses only pre-t information.
Crowd baseline = option prices at t normalised to sum to 1. The JSON contexts carry the same fields as the prose ones.
Writes data/choice.parquet.
"""
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pipeline.build_dataset import price_at
from pipeline.candidates import CUTOFF, HORIZONS_DAYS, MIN_VOLUME, ROOT, SHORT_RECURRING_TAGS, category, split_of, stale_end, ts
from pipeline.fetch_prices import OUT as PRICES, fetch

MAX_OPTIONS = 16
LETTERS = "ABCDEFGHIJKLMNOP"
RULES_CHARS = 1500


def load_groups():
    groups = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data/raw/markets/*.jsonl"))):
        for line in open(f):
            m = json.loads(line)
            tags = {t["label"] for t in m.get("tags") or []}
            if (not m.get("negRisk") or not m.get("negRiskMarketID") or m.get("outcomes") != '["Yes", "No"]'
                    or tags & SHORT_RECURRING_TAGS or m.get("umaResolutionStatus") != "resolved"
                    or m.get("outcomePrices") not in ('["1", "0"]', '["0", "1"]')):
                continue
            if not all(m.get(k) for k in ("createdAt", "startDate", "endDate", "closedTime", "clobTokenIds")):
                continue
            groups.setdefault(m["negRiskMarketID"], []).append(m)
    return groups


def event_rows(ms):
    if len(ms) < 3 or sum(m["outcomePrices"] == '["1", "0"]' for m in ms) != 1:
        return []
    if sum(float(m.get("volumeNum") or 0) for m in ms) < MIN_VOLUME:
        return []
    names = [(m.get("groupItemTitle") or "").strip() or ("Other" if m.get("negRiskOther") else "") for m in ms]
    if "" in names or len(set(names)) != len(names):
        return []
    ends = {ts(m["endDate"]) for m in ms}
    if max(ends) - min(ends) > pd.Timedelta(days=1):
        return []
    end = max(ends)
    if any(stale_end(m["question"], ts(m["endDate"])) for m in ms):
        return []
    opened = max(max(ts(m["createdAt"]), ts(m["startDate"])) for m in ms)
    first_close = min(ts(m["closedTime"]) for m in ms)
    last_close = max(ts(m["closedTime"]) for m in ms)
    e = (ms[0].get("events") or [{}])[0]
    tags = {t["label"] for t in ms[0].get("tags") or []}
    rows = []
    for h in HORIZONS_DAYS:
        t = (end - pd.Timedelta(days=h)).floor("h")
        if t < CUTOFF or t < opened or t >= first_close:
            continue
        split = split_of(t, end, last_close)
        if split is None:
            continue
        rows.append({"event_id": e.get("id") or ms[0]["negRiskMarketID"], "split": split, "horizon_days": h, "t": t,
                     "end": end, "category": category(tags), "title": e.get("title") or ms[0]["question"],
                     "rules": " ".join((e.get("description") or ms[0].get("description") or "").split())[:RULES_CHARS],
                     "options": names, "market_ids": [m["id"] for m in ms],
                     "yes_tokens": [json.loads(m["clobTokenIds"])[0] for m in ms],
                     "winner": names[[m["outcomePrices"] == '["1", "0"]' for m in ms].index(True)]})
    return rows


def main():
    rows = [r for ms in load_groups().values() for r in event_rows(ms)]
    print(f"{len(rows)} event-forecast rows from {len({r['event_id'] for r in rows})} events", flush=True)

    cache = {}
    for line in open(PRICES):
        p = json.loads(line)
        cache[(p["market_id"], p["horizon_days"], p["t"])] = p["history"]
    need = {(mid, r["horizon_days"], int(r["t"].timestamp()), tok, r["t"])
            for r in rows for mid, tok in zip(r["market_ids"], r["yes_tokens"])}
    todo = [SimpleNamespace(market_id=m, horizon_days=h, yes_token=tok, t=t)
            for m, h, ti, tok, t in need if (m, h, ti) not in cache]
    print(f"prices: {len(need) - len(todo)} cached, {len(todo)} to fetch", flush=True)
    with open(PRICES, "a") as f, ThreadPoolExecutor(8) as ex:
        for i, p in enumerate(ex.map(fetch, todo), 1):
            f.write(json.dumps(p) + "\n")
            cache[(p["market_id"], p["horizon_days"], p["t"])] = p["history"]
            if i % 2000 == 0:
                print(i, flush=True)

    out = []
    for r in rows:
        ti = int(r["t"].timestamp())
        prices = np.array([price_at(cache[(m, r["horizon_days"], ti)], r["t"]) for m in r["market_ids"]])
        if np.isnan(prices).any() or prices.sum() <= 0:
            continue
        opts, p = list(r["options"]), prices
        if len(opts) > MAX_OPTIONS:
            keep = np.argsort(-p)[:MAX_OPTIONS - 1]
            rest = np.setdiff1d(np.arange(len(opts)), keep)
            winner = r["winner"] if r["winner"] in [opts[i] for i in keep] else "Any other option"
            opts, p = [opts[i] for i in keep] + ["Any other option"], np.append(p[keep], p[rest].sum())
        else:
            winner = r["winner"]
        crowd = p / p.sum()
        lines = [f"Today is {r['t']:%A, %B %-d, %Y, %H:%M} UTC.", f"Event: {r['title']}",
                 f"The market closes {r['end']:%B %-d, %Y, %H:%M} UTC.", f"Resolution rules: {r['rules']}"]
        state = "\n".join(lines)
        price_line = "Market prices: " + "; ".join(f"{LETTERS[i]}) {o} {x:.2f}" for i, (o, x) in enumerate(zip(opts, p)))
        obj = {"today": f"{r['t']:%Y-%m-%d %H:%M} UTC", "event": r["title"],
               "market_closes": f"{r['end']:%Y-%m-%d %H:%M} UTC", "resolution_rules": r["rules"]}
        priced = {**obj, "market_prices": {f"{LETTERS[i]}) {o}": round(float(x), 2) for i, (o, x) in enumerate(zip(opts, p))}}
        out.append({**{k: r[k] for k in ("event_id", "split", "horizon_days", "t", "end", "category", "title")},
                    "options": opts, "n_options": len(opts), "y_index": opts.index(winner),
                    "crowd": crowd.tolist(), "price_sum": float(p.sum()),
                    "question": f"{r['title'].rstrip('?')}?", "state": state, "state_price": state + "\n" + price_line,
                    "state_json": json.dumps(obj, ensure_ascii=False),
                    "state_json_price": json.dumps(priced, ensure_ascii=False)})
    d = pd.DataFrame(out)
    d.to_parquet(os.path.join(ROOT, "data/choice.parquet"))
    print(d.groupby("split").agg(rows=("y_index", "size"), events=("event_id", "nunique"),
                                 options_median=("n_options", "median")).to_string())
    print("categories:", d.category.value_counts().to_dict())
    crowd_acc = np.mean([np.argmax(c) == y for c, y in zip(d.crowd, d.y_index)])
    print(f"crowd picks the winner {crowd_acc:.3f} of the time | median sum of option prices {d.price_sum.median():.2f}")


if __name__ == "__main__":
    main()
