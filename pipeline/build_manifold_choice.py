"""Manifold multiple-choice markets as a cross-platform multiple-choice test set (same layout as data/choice.parquet).

Leakage rules are those of pipeline/build_manifold.py (t = created + h, t >= TRAIN_LABEL_END, unresolved at t, resolved by
SNAPSHOT, context = date and question only, no creator-editable text), plus two multiple-choice rules:
  1. Only answers that existed at t are options. A row whose winning answer was added after t is dropped.
  2. An option's crowd probability at t is the probAfter of its last bet at or before t. Options with no bet yet
     share the probability mass the others leave; the vector is then normalised.
Filters: answers sum to one, resolved to a single answer, at least MIN_BETTORS unique bettors, at least 3
options at t, a bet in the 7 days before t. Events with more than MAX_OPTIONS options keep the MAX_OPTIONS - 1
with the highest probability at t plus "Any other option", as in pipeline/build_choice.py.
Writes data/manifold_choice.parquet with split="manifold".
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from pipeline.build_choice import LETTERS, MAX_OPTIONS
from pipeline.build_manifold import HORIZONS, RAW, SNAPSHOT, TRAIN_LABEL_END, get, list_markets, ms
from pipeline.candidates import ROOT

MIN_BETTORS = 10  # multiple-choice markets are thinner than binary ones (pipeline/build_manifold.py uses 20)


def cached(path, fetch):
    if os.path.exists(path):
        return json.load(open(path))
    x = fetch()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(x, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    return x


def all_bets(market_id):
    def fetch():
        bets, before = [], None
        while True:
            page = get("bets", contractId=market_id, limit=1000, **({"before": before} if before else {}))
            bets += [{"t": b["createdTime"], "a": b["answerId"], "p": b["probAfter"]}
                     for b in page if b.get("answerId") and "probAfter" in b]
            if len(page) < 1000:
                return sorted(bets, key=lambda b: b["t"])
            before = page[-1]["id"]
    return cached(os.path.join(RAW, "mc_bets", f"{market_id}.json"), fetch)


def market(market_id):
    return cached(os.path.join(RAW, "mc_markets", f"{market_id}.json"), lambda: get(f"market/{market_id}"))


def candidates(markets):
    out = []
    for m in markets:
        if (m.get("outcomeType") != "MULTIPLE_CHOICE" or m.get("resolution") in (None, "MKT", "CANCEL", "CHOOSE_MULTIPLE")
                or (m.get("uniqueBettorCount") or 0) < MIN_BETTORS or not m.get("resolutionTime")):
            continue
        created, resolved = ms(m["createdTime"]), ms(m["resolutionTime"])
        ts = [(created + pd.Timedelta(days=h)).floor("h") for h in HORIZONS]
        hs = [(h, t) for h, t in zip(HORIZONS, ts) if TRAIN_LABEL_END <= t < resolved <= SNAPSHOT]
        if hs:
            out.append((m["id"], hs))
    return out


def rows(market_id, hs):
    m = market(market_id)
    if not m.get("shouldAnswersSumToOne"):
        return []
    bets = all_bets(market_id)
    out = []
    for h, t in hs:
        tms = t.timestamp() * 1000
        answers = [a for a in m["answers"] if a["createdTime"] <= tms]
        if len(answers) < 3 or m["resolution"] not in {a["id"] for a in answers}:
            continue
        before = [b for b in bets if b["t"] <= tms]
        if not any(b["t"] >= tms - 7 * 86400e3 for b in before):
            continue
        last = {b["a"]: b["p"] for b in before}  # sorted by time, so the last bet wins
        p = np.array([last.get(a["id"], np.nan) for a in answers])
        if np.isnan(p).all():
            continue
        missing = np.isnan(p)
        p[missing] = max(0.0, 1 - p[~missing].sum()) / max(1, missing.sum())
        names = [" ".join(a["text"].split()) for a in answers]
        win = [a["id"] for a in answers].index(m["resolution"])
        if len(names) > MAX_OPTIONS:
            keep = np.argsort(-p)[:MAX_OPTIONS - 1]
            rest = np.setdiff1d(np.arange(len(names)), keep)
            win = list(keep).index(win) if win in keep else MAX_OPTIONS - 1
            names, p = [names[i] for i in keep] + ["Any other option"], np.append(p[keep], p[rest].sum())
        crowd = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
        state = f"Today is {t:%A, %B %-d, %Y, %H:%M} UTC.\nEvent: {m['question']}"
        price_line = "Market prices: " + "; ".join(f"{LETTERS[i]}) {o} {x:.2f}" for i, (o, x) in enumerate(zip(names, crowd)))
        obj = {"today": f"{t:%Y-%m-%d %H:%M} UTC", "event": m["question"]}
        priced = {**obj, "market_prices": {f"{LETTERS[i]}) {o}": round(float(x), 2) for i, (o, x) in enumerate(zip(names, crowd))}}
        out.append({"event_id": market_id, "split": "manifold", "horizon_days": h, "t": t, "category": "manifold",
                    "title": m["question"], "options": names, "n_options": len(names), "y_index": win,
                    "crowd": crowd.tolist(), "bettors": m["uniqueBettorCount"], "question": m["question"],
                    "state": state, "state_price": state + "\n" + price_line,
                    "state_json": json.dumps(obj, ensure_ascii=False),
                    "state_json_price": json.dumps(priced, ensure_ascii=False)})
    return out


def main():
    c = candidates(list_markets())
    print(f"{len(c)} candidate markets", flush=True)
    with ThreadPoolExecutor(4) as ex:
        d = pd.DataFrame([r for rs in ex.map(lambda x: rows(*x), c) for r in rs])
    d.to_parquet(os.path.join(ROOT, "data/manifold_choice.parquet"))
    acc = np.mean([np.argmax(p) == y for p, y in zip(d.crowd, d.y_index)])
    print(f"kept {len(d)} rows from {d.event_id.nunique()} markets | median options {d.n_options.median():.0f} | "
          f"crowd picks the winner {acc:.3f}")
    print(d.state_price.iloc[0])


if __name__ == "__main__":
    main()
