"""ForecastBench (Karger et al., forecastingresearch/forecastbench-datasets) as a leak-free yes/no test set.

Question sets are released every two weeks, each frozen at a freeze date with the market value (market questions:
Polymarket, Manifold, Metaculus, INFER, Kalshi) or the latest data value (dataset questions: ACLED, DBnomics, FRED,
SerpAPI, Wikipedia, yfinance), and resolved at several resolution dates in the matching resolution set.
Leakage rules:
  1. Only question sets frozen on or after TRAIN_LABEL_END (Aug 1 2026), after every outcome the fine-tuned
     models learned.
  2. One row per question and resolution date that has resolved to 0 or 1. Market questions still open at a
     resolution date (which ForecastBench scores against the market price) are dropped.
  3. A question in several sets keeps its earliest set only, so each (question, resolution date) counts once.
  4. The context is ForecastBench's own frozen text: the date, the question (its {forecast_due_date} and
     {resolution_date} placeholders filled), the resolution date, the resolution criteria and background, and in
     state_price the market price or the latest data value at the freeze date. state_json_price holds the same
     fields as one JSON object, with the keys of the Polymarket JSON contexts.
Writes data/forecastbench.parquet (split="forecastbench") with the columns logitcast/eval_forecast.py expects.

--val writes data/forecastbench_val.parquet (split="fb_val") instead: sets frozen from VAL_START to TRAIN_LABEL_END,
rows resolved before TRAIN_LABEL_END, so every label was public before the first test set froze. It calibrates the
tuned models on ForecastBench's own question types (logitcast/calibrate_unified.py); nothing trains on it.
"""
import argparse
import ast
import json
import os

import numpy as np
import pandas as pd
import requests

from pipeline.build_manifold import TRAIN_LABEL_END
from pipeline.candidates import ROOT

BASE = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"
LISTING = "https://api.github.com/repos/forecastingresearch/forecastbench-datasets/contents/datasets/resolution_sets"
RAW = os.path.join(ROOT, "data/raw/forecastbench")
MARKETS = {"polymarket", "manifold", "metaculus", "infer", "kalshi"}
TEXT_CHARS = 1500
VAL_START = pd.Timestamp("2026-01-01", tz="UTC")


def fetch(name):
    path = os.path.join(RAW, name.replace("/", "_"))
    if not os.path.exists(path):
        r = requests.get(f"{BASE}/{name}", timeout=120)
        r.raise_for_status()
        os.makedirs(RAW, exist_ok=True)
        open(path, "w").write(r.text)
    return json.load(open(path))


def day(ts):
    return f"{ts:%A, %B} {ts.day}, {ts.year}"


def rows(due, split="forecastbench"):
    qs = {q["id"]: q for q in fetch(f"question_sets/{due}-llm.json")["questions"] if isinstance(q["id"], str)}
    out = []
    for e in fetch(f"resolution_sets/{due}_resolution_set.json")["resolutions"]:
        q = qs.get(e["id"]) if isinstance(e["id"], str) else None
        if q is None or not e["resolved"] or e["resolved_to"] not in (0, 1):
            continue
        t = pd.Timestamp(q["freeze_datetime"]).tz_convert("UTC")
        end = pd.Timestamp(e["resolution_date"], tz="UTC")
        if end <= t or (t < TRAIN_LABEL_END if split == "forecastbench" else end >= TRAIN_LABEL_END):
            continue
        question = q["question"].replace("{forecast_due_date}", due).replace("{resolution_date}", e["resolution_date"])
        market = q["source"] in MARKETS
        raw = str(q["freeze_datetime_value"] or "").strip()
        value = float(raw) if market and raw.replace(".", "", 1).isdigit() else np.nan
        state = (f"Today is {day(t)}.\nEvent: {question}\nThe question resolves on {day(end)}.\n"
                 f"Resolution rules: {' '.join(q['resolution_criteria'].split())}\n"
                 f"Background: {' '.join(str(q['background']).split())[:TEXT_CHARS]}")
        known = (f"\nMarket price of Yes now: {value:.2f}." if not np.isnan(value) else "") if market else \
            (f"\nLatest value: {raw}." if raw not in ("", "N/A", "None") else "")
        obj = {"today": f"{t:%Y-%m-%d %H:%M} UTC", "event": question, "market_closes": f"{end:%Y-%m-%d %H:%M} UTC",
               "resolution_rules": " ".join(q["resolution_criteria"].split()),
               "background": " ".join(str(q["background"]).split())[:TEXT_CHARS]}
        if not np.isnan(value):
            obj["market"] = {"probability_yes_now": round(value, 2)}
        elif not market and raw not in ("", "N/A", "None"):
            obj["latest_value"] = raw
        out.append({"market_id": q["id"], "event_id": q["id"], "split": split, "due": due,
                    "horizon_days": (end - t).days, "t": t, "end": end, "y": int(e["resolved_to"]),
                    "question": question, "category": q["source"], "origin": "market" if market else "dataset",
                    "p_t": value if market else np.nan, "state": state, "state_price": state + known,
                    "state_context": state + known, "state_json_price": json.dumps(obj, ensure_ascii=False)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", action="store_true")
    a = ap.parse_args()
    dues = sorted(x["name"].split("_")[0] for x in requests.get(LISTING, timeout=60).json()
                  if x["name"].endswith("_resolution_set.json"))
    if a.val:
        keep = [due for due in dues if VAL_START <= pd.Timestamp(due, tz="UTC") < TRAIN_LABEL_END]
        split, out = "fb_val", "data/forecastbench_val.parquet"
    else:
        keep = [due for due in dues if pd.Timestamp(due, tz="UTC") >= TRAIN_LABEL_END]
        split, out = "forecastbench", "data/forecastbench.parquet"
    d = pd.DataFrame([r for due in keep for r in rows(due, split)])
    d = d.sort_values("due").drop_duplicates(["market_id", "end"]).reset_index(drop=True)
    d.to_parquet(os.path.join(ROOT, out))
    print(f"{len(d)} rows from sets {d.due.min()}..{d.due.max()}, freeze {d.t.min():%Y-%m-%d}..{d.t.max():%Y-%m-%d}, "
          f"resolving {d.end.min():%Y-%m-%d}..{d.end.max():%Y-%m-%d} | yes rate {d.y.mean():.3f}")
    print(d.groupby("category").agg(n=("y", "size"), yes=("y", "mean"), horizon=("horizon_days", "median")).to_string())


if __name__ == "__main__":
    main()
