"""Add news contexts to a dataset from the news cache (no network; run pipeline/fetch_news.py --data ... first).
Default data/forecast.parquet; data/choice.parquet (multiple choice) gets the same columns plus state_context
(= state_price_news), the context column it shares with data/openforesight_choice.parquet for joint training.

  state_news        state + up to 10 headlines published on days before day(t)
  state_price_news  state_price + the same headlines
  n_headlines       how many headlines were found
"""
import argparse
import os

import pandas as pd

from pipeline.candidates import ROOT
from pipeline.fetch_news import queries
from pipeline.news import headlines, news_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/forecast.parquet")
    path = os.path.join(ROOT, ap.parse_args().data)
    d = pd.read_parquet(path)
    q = queries(d[["state", "question", "t"]])
    items = [headlines(r.query, r.t, offline=True) for r in q.itertuples()]
    block = [news_text(i) for i in items]
    d["n_headlines"] = [len(i) for i in items]
    d["state_news"] = [s + "\n" + b for s, b in zip(d.state, block)]
    d["state_price_news"] = [s + "\n" + b for s, b in zip(d.state_price, block)]
    if "y_index" in d.columns:  # multiple choice: the shared context column of the multiple-choice mixture
        d["state_context"] = d.state_price_news
    tmp = path + ".tmp"
    d.to_parquet(tmp)
    os.replace(tmp, path)
    print(d.groupby("split").n_headlines.agg(["mean", "median", lambda s: (s > 0).mean()]).round(2))


if __name__ == "__main__":
    main()
