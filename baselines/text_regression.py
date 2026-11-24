"""Classical baseline: logistic regression on bag-of-words text features, plus the prices in the market-informed variant.

Features: TF-IDF (word uni- and bigrams, sublinear tf) of `state`, the question with its plain-text context and no
prices, the text the market-blind model reads. The market-informed variant adds logit(p_t), logit(p_1d) and
logit(p_7d), clipped to [0.01, 0.99], the three prices the market-informed input carries. Trained on the Polymarket
train split like the tuned models, the L2 strength chosen by log loss on the Polymarket validation split, then
refitted on train and applied unchanged to every test set. A logistic regression is calibrated by its objective, so
no temperature is fitted. ForecastBench rows without a market value are scored by the market-blind model, since
the market-informed one needs all three prices.

Writes results/textreg_{informed,blind}_{set}.parquet (market_id, horizon_days, p) in the order of each full test
split, so paper.tables.aligned scores them on any subset.
"""
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.special import logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from pipeline.candidates import ROOT

PRICES = ["p_t", "p_1d", "p_7d"]
CS = (0.03, 0.1, 0.3, 1, 3, 10)
TESTS = {"poly": ("data/forecast.parquet", "test"), "manifold": ("data/manifold.parquet", "manifold"),
         "kalshi": ("data/kalshi.parquet", "kalshi"), "fb": ("data/forecastbench.parquet", "forecastbench"),
         "btf3": ("data/btf3.parquet", "btf3")}


def prices(d):
    return csr_matrix(logit(np.clip(d[PRICES].to_numpy(float), 0.01, 0.99)))


def features(vec, d, informed):
    x = vec.transform(d.state)
    return hstack([x, prices(d)]).tocsr() if informed else x


def fit(train, val, informed):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True).fit(train.state)
    xt, xv = features(vec, train, informed), features(vec, val, informed)
    scores = {c: log_loss(val.y, LogisticRegression(C=c, max_iter=2000).fit(xt, train.y).predict_proba(xv)[:, 1])
              for c in CS}
    c = min(scores, key=scores.get)
    print(f"{'informed' if informed else 'blind'}: C={c}, val log loss " + ", ".join(f"{k}:{v:.4f}" for k, v in scores.items()))
    return vec, LogisticRegression(C=c, max_iter=2000).fit(xt, train.y)


def main():
    d = pd.read_parquet(os.path.join(ROOT, "data/forecast.parquet"))
    train, val = d[d.split == "train"], d[d.split == "val"]
    models = {v: fit(train, val, v == "informed") for v in ("informed", "blind")}
    for name, (path, split) in TESTS.items():
        t = pd.read_parquet(os.path.join(ROOT, path))
        t = t[t.split == split].reset_index(drop=True)
        blind = models["blind"][1].predict_proba(features(models["blind"][0], t, False))[:, 1]
        has = t.reindex(columns=PRICES).notna().all(axis=1).to_numpy()
        informed = blind.copy()
        if has.any():
            vec, m = models["informed"]
            informed[has] = m.predict_proba(features(vec, t[has], True))[:, 1]
        for v, p in (("informed", informed), ("blind", blind)):
            t[["market_id", "horizon_days"]].assign(p=p).to_parquet(os.path.join(ROOT, f"results/textreg_{v}_{name}.parquet"))
        y = t.y.to_numpy()
        print(f"{name}: {len(t)} rows ({has.mean():.0%} with prices), Brier x100 informed "
              f"{100 * np.mean((informed - y) ** 2):.2f}, blind {100 * np.mean((blind - y) ** 2):.2f}")


if __name__ == "__main__":
    main()
