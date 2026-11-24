"""Score JEV's choice head on a multiple-choice set against its crowd (data/choice.parquet: normalised option
prices at t; data/openforesight_choice.parquet has no crowd, so its "crowd" is uniform).

Writes results/{name}_{split}_choice.parquet with one probability vector per context, then prints accuracy,
multi-class Brier (sum over options of squared error) and log loss for JEV, the crowd and a uniform guess.
"""
import argparse
import os

import numpy as np
import pandas as pd

from pipeline.candidates import ROOT
from logitcast.jev import CONTEXTS, MODEL_DIR, Jev


def scores(probs, y):
    acc = np.mean([np.argmax(p) == t for p, t in zip(probs, y)])
    brier = np.mean([((np.asarray(p) - np.eye(len(p))[t]) ** 2).sum() for p, t in zip(probs, y)])
    ll = np.mean([-np.log(max(p[t], 1e-6)) for p, t in zip(probs, y)])
    return {"n": len(y), "acc": acc, "brier": brier, "logloss": ll}


def report(d, cols):
    y = d.y_index.tolist()
    systems = {c: d[c].tolist() for c in cols} | {"crowd": d.crowd.tolist(),
                                                   "uniform": [np.full(k, 1 / k) for k in d.n_options]}
    rows = [{"slice": s, "system": k, **scores([v[i] for i in idx], [y[i] for i in idx])}
            for s, idx in [("all", range(len(d)))] + [(c, np.flatnonzero(d.category == c))
                                                     for c in ("politics", "sports", "weather") if (d.category == c).any()]
            for k, v in systems.items()]
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default="data/choice.parquet")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--temp_tag", default=None, help="a unified run's input setting (temperature KIND@TAG)")
    ap.add_argument("--name", default="jev9b_zeroshot")
    ap.add_argument("--contexts", nargs="+", default=["state", "state_price", "state_json", "state_json_price"])
    a = ap.parse_args()
    d = pd.read_parquet(os.path.join(ROOT, a.data))
    d = d[d.split == a.split].reset_index(drop=True)
    m = Jev(a.model_dir, run_dir=a.run_dir, temp_tag=a.temp_tag)
    contexts = {c: CONTEXTS[c] for c in a.contexts if c in d.columns}
    for col, out in contexts.items():
        d[out] = m.decide("choice", d[col].tolist(), d.question.tolist(), [list(o) for o in d.options], batch_size=8)
        print(f"scored {col}", flush=True)
    d.drop(columns=list(contexts)).to_parquet(os.path.join(ROOT, f"results/{a.name}_{a.split}_choice.parquet"))
    report(d, list(contexts.values()))


if __name__ == "__main__":
    main()
