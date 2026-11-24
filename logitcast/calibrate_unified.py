"""Calibrate the unified models (LogitCast and Qwen-3.5-9B-LogitCast) on ForecastBench's earlier rounds (its ForecastBench scores in Tables 1-2).

The unified models read every binary set through a temperature fitted on Polymarket validation questions. On
ForecastBench's data-series questions that leaves their forecasts at the wrong level and too narrowly spread, so
each data source gets p -> sigmoid(a * logit(p) + b), fitted on that source's questions in data/forecastbench_val.parquet
(rounds resolved before TRAIN_LABEL_END, scored by scripts/cal_val.sh). Market questions, the kind the model is
trained on, keep their forecasts. Each seed is calibrated separately.
Writes results/uni_{init}_s{seed}_{setting}cal_forecastbench.parquet.
    python -m logitcast.calibrate_unified
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pipeline.candidates import ROOT

R = os.path.join(ROOT, "results")
BIN_COL = {"informed": "jev_json_price", "blind": "jev"}


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def platt(val_p, val_y):
    m = LogisticRegression(C=1e6).fit(logit(val_p)[:, None], val_y)
    return lambda p: m.predict_proba(logit(p)[:, None])[:, 1]


def calibrate(init, seed, setting):
    name, col = f"uni_{init}_s{seed}_{setting}", BIN_COL[setting]
    val = pd.read_parquet(os.path.join(R, f"{name}_fb_val.parquet"))
    test = pd.read_parquet(os.path.join(R, f"{name}_forecastbench.parquet"))
    p = test[col].to_numpy(float).copy()
    for src, v in val[val.origin == "dataset"].groupby("category"):
        sel = (test.category == src).to_numpy()
        if sel.any():
            p[sel] = platt(v[col].to_numpy(), v.y)(p[sel])
    test.assign(**{col: p}).to_parquet(os.path.join(R, f"{name}cal_forecastbench.parquet"))
    y = test.y.to_numpy()
    return f"{name}: {np.mean((test[col] - y) ** 2):.4f} -> {np.mean((p - y) ** 2):.4f}"


if __name__ == "__main__":
    for init in ("jev", "base"):
        for seed in (0, 1):
            for setting in BIN_COL:
                print(calibrate(init, seed, setting), flush=True)
