"""Fill the news cache for every (query, day(t)) in a dataset (default data/forecast.parquet; also
data/choice.parquet), test first, then val, then train.

Resumable: cached requests are skipped. A failed request is logged and retried on the next run.
"""
import argparse

import pandas as pd

from pipeline.candidates import ROOT
from pipeline.news import headlines, query_for


def queries(d):
    q = [query_for(s.split("\n")[1].removeprefix("Event: "), x) for s, x in zip(d.state, d.question)]
    return d.assign(query=q, day=d.t.dt.normalize())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/forecast.parquet")
    a = ap.parse_args()
    d = queries(pd.read_parquet(f"{ROOT}/{a.data}", columns=["state", "question", "t", "split"]))
    d["order"] = d.split.map({"test": 0, "val": 1, "train": 2})
    u = d.sort_values(["order", "day"]).drop_duplicates(["query", "day"])
    print(f"{len(u)} unique requests", flush=True)
    failed = 0
    for i, r in enumerate(u.itertuples(), 1):
        try:
            headlines(r.query, r.t)
        except RuntimeError as e:
            failed += 1
            print("FAILED", e, flush=True)
        if i % 500 == 0:
            print(f"{i}/{len(u)} ({r.split}) failed so far: {failed}", flush=True)
    print(f"done, failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
