"""Price-only baselines: models that see the market's price history and metadata but no question text.

  A  logistic regression on logit(p_t), logit(p_1d), logit(p_7d)
  B  gradient boosting on those plus the time-series summary features, horizon, category and pre-t activity
Trained on the train split only, scored on test against the market and fine-tuned JEV-9B + price.
If they match JEV, the text adds nothing beyond re-calibrating the price series.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from pipeline.candidates import ROOT
from analysis.compare_runs import brier_gap_vs_market, liquid_threshold
from logitcast.eval_forecast import scores

TS = ["ts_low", "ts_high", "ts_change_7d", "ts_change_1d", "ts_moves_1d", "ts_moves_7d", "ts_std_7d"]


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def features(d, rich):
    X = pd.DataFrame({c: logit(d[c].to_numpy()) for c in ("p_t", "p_1d", "p_7d")}, index=d.index)
    if rich:
        X[TS] = d[TS].fillna(0.0)
        X["horizon"] = d.horizon_days
        X["activity_7d"] = d.activity_7d
        X["category"] = d.category.astype("category").cat.codes
    return X


def price_logistic(train):
    """Classical baseline A: logistic regression on logit(p_t), logit(p_1d), logit(p_7d), fitted on `train`."""
    return LogisticRegression(max_iter=2000).fit(features(train, False), train.y)


def main():
    d = pd.read_parquet(os.path.join(ROOT, "data/forecast.parquet"))
    tr, te = d[d.split == "train"], d[d.split == "test"].reset_index(drop=True)
    te = te.assign(category=pd.Categorical(te.category, categories=sorted(d.category.unique())))
    tr = tr.assign(category=pd.Categorical(tr.category, categories=sorted(d.category.unique())))
    a = price_logistic(tr)
    b = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True, random_state=0,
                                       categorical_features=[list(features(tr, True).columns).index("category")])
    b.fit(features(tr, True), tr.y)
    te["market"] = te.p_t
    te["A"] = a.predict_proba(features(te, False))[:, 1]
    te["B"] = b.predict_proba(features(te, True))[:, 1]
    ft = [pd.read_parquet(os.path.join(ROOT, f"results/state_price_s{s}_test.parquet")) for s in (0, 1)]
    key = ["market_id", "horizon_days"]
    j = ft[0][key].assign(jev=np.mean([x.jev_price.to_numpy() for x in ft], axis=0))
    te = te.merge(j, on=key)
    thr = liquid_threshold()
    names = {"market": "Market price at t", "A": "A: logistic on 3 prices", "B": "B: boosting on price summary",
             "jev": "Fine-tuned JEV-9B + price"}
    for sl, g in (("all", te), ("liquid", te[te.activity_7d >= thr]), ("thin", te[te.activity_7d < thr])):
        print(f"\n[{sl}] n={len(g)}")
        for col, name in names.items():
            sc = scores(g[col].to_numpy(), g.y.to_numpy())
            gap = "" if col == "market" else "  vs market {:+.4f} [{:+.4f}, {:+.4f}]".format(*brier_gap_vs_market(g, col)[:3])
            print(f"  {name:30s} Brier {sc['brier']:.4f}  logloss {sc['logloss']:.3f}  AUC {sc['auc']:.3f}{gap}")
    m, lo, hi, _ = brier_gap_vs_market(te.assign(market=te.B), "jev")
    print(f"\nJEV minus B (all): {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
