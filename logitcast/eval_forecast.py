"""Score a System 1 model on data/forecast.parquet against market and base-rate baselines.

Writes per-row predictions to results/<name>_<split>.parquet and prints Brier / log loss / ECE / AUC,
overall and by horizon and category. Lower Brier and log loss are better.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from pipeline.candidates import ROOT
from logitcast.jev import CONTEXTS, MODEL_DIR, Jev


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def scores(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"n": len(y), "brier": np.mean((p - y) ** 2), "logloss": -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
            "ece": ece(p, y), "auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan}


def report(df, systems):
    rows = []
    for (key, g) in [("all", df)] + [(f"h={h}d", g) for h, g in df.groupby("horizon_days")] + \
                    [(c, g) for c, g in df.groupby("category")]:
        for s in systems:
            h = g[g[s].notna()]  # the market has no price on ForecastBench's dataset questions
            if len(h):
                rows.append({"slice": key, "system": s, **scores(h[s].to_numpy(), h.y.to_numpy())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    ap.add_argument("--run_dir", default=None, help="logitcast/train_forecast.py output; default is the released model")
    ap.add_argument("--temp_tag", default=None, help="a unified run's input setting (temperature KIND@TAG)")
    ap.add_argument("--init", choices=["jev", "base"], default="jev",
                    help="without --run_dir: jev = the released JEV-9B, base = the backbone alone (its own answer-token "
                         "head, temperature 1), the starting point of the same-backbone control")
    ap.add_argument("--name", default="jev9b_zeroshot")
    ap.add_argument("--data", default="data/forecast.parquet", help="e.g. data/manifold.parquet (split manifold)")
    ap.add_argument("--contexts", nargs="+", default=None, help="context columns to score (default: all present)")
    a = ap.parse_args()
    train = pd.read_parquet(os.path.join(ROOT, "data/forecast.parquet"), columns=["split", "y"])
    base_rate = train[train.split == "train"].y.mean()  # what the models were trained on
    df = pd.read_parquet(os.path.join(ROOT, a.data))
    df = df[df.split == a.split].reset_index(drop=True)

    m = Jev(a.model_dir, run_dir=a.run_dir, init=a.init, temp_tag=a.temp_tag)
    contexts = {c: o for c, o in CONTEXTS.items() if c in df.columns and (a.contexts is None or c in a.contexts)}
    for col, out in contexts.items():
        df[out] = [p[1] for p in m.decide("noul", df[col].tolist(), df.question.tolist())]
    df["market"], df["base_rate"], df["half"] = df.p_t, base_rate, 0.5

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.drop(columns=list(contexts)).to_parquet(os.path.join(ROOT, f"results/{a.name}_{a.split}.parquet"))
    r = report(df, list(contexts.values()) + ["market", "base_rate", "half"])
    r.to_csv(os.path.join(ROOT, f"results/{a.name}_{a.split}_scores.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.float_format", "{:.4f}".format):
        print(f"base rate (train YES) = {base_rate:.3f}")
        print(r.pivot(index="slice", columns="system", values="brier"))
        print(r[r.slice == "all"])


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
