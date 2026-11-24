"""Add state_context, the context column shared by every training file, so one model can be tuned on all of them.

  data/forecast.parquet      Polymarket yes/no        state_price_news (prices and headlines)
  data/openforesight.parquet OpenForesight yes/no     state (background and resolution rules)
  data/manifold.parquet      Manifold yes/no          state_price (date, question and prices; no news is fetched)
  data/manifold_choice.parquet Manifold MC            state_price (test only; date, question and option prices)
The other files set it when they are built: data/choice.parquet (pipeline/add_news.py, prices and headlines),
data/openforesight_choice.parquet (background) and data/external.parquet (background and community forecast).
"""
import os

import pandas as pd

from pipeline.candidates import ROOT

SOURCES = {"data/forecast.parquet": "state_price_news", "data/openforesight.parquet": "state",
           "data/manifold.parquet": "state_price", "data/manifold_choice.parquet": "state_price"}


def main():
    for rel, col in SOURCES.items():
        path = os.path.join(ROOT, rel)
        d = pd.read_parquet(path)
        d["state_context"] = d[col]
        d.to_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"{rel}: state_context = {col} ({len(d)} rows)")


if __name__ == "__main__":
    main()
