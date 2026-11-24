"""Compare finished runs on test from their saved predictions (results/<run>_test.parquet).

Each config is a replicate pair; the reported number is the mean over its seeds. Every slice is
shown for all markets and for liquid markets only, since thin markets carry mispricings
(e.g. unfilled order books) that a model can exploit without any forecasting skill.
"Liquid" = pre-t price activity at or above the train split's 80th percentile, fixed before
looking at test (final volume is not used: it includes trading after t).
"""
import argparse
import os

import numpy as np
import pandas as pd

from pipeline.candidates import ROOT
from logitcast.eval_forecast import scores
from logitcast.jev import CONTEXTS


def liquid_threshold():
    d = pd.read_parquet(os.path.join(ROOT, "data/forecast.parquet"), columns=["split", "activity_7d"])
    return d[d.split == "train"].activity_7d.quantile(0.8)


def slices(df, thr):
    liquid = df.activity_7d >= thr
    yield "all", df
    yield "liquid", df[liquid]
    yield "thin", df[~liquid]
    for c, g in df.groupby("category"):
        if len(g) >= 100:
            yield c, g
            if (g.activity_7d >= thr).sum() >= 100:
                yield f"{c}, liquid", g[g.activity_7d >= thr]
    for h, g in df.groupby("horizon_days"):
        yield f"h={h}d", g


def brier_gap_vs_market(d, col, B=2000, seed=0):
    """Mean Brier(model) - Brier(market) with a 95% bootstrap CI that resamples whole events,
    since markets of one event are correlated."""
    diff = (d[col] - d.y) ** 2 - (d.market - d.y) ** 2
    ev = diff.groupby(d.event_id).agg(["sum", "size"])
    s, n = ev["sum"].to_numpy(), ev["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ev), (B, len(ev)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    return diff.mean(), *np.percentile(boot, [2.5, 97.5]), len(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONTEXTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--metric", default="brier", choices=["brier", "logloss", "ece", "auc"])
    a = ap.parse_args()

    thr = liquid_threshold()
    print(f"liquid = activity_7d >= {thr:.0f} price changes in the 7 days before t (train 80th percentile)")
    zs = pd.read_parquet(os.path.join(ROOT, "results/jev9b_zeroshot_test.parquet"))
    rows = []
    for name, g in slices(zs, thr):
        rows.append({"slice": name, "system": "market", **scores(g.market.to_numpy(), g.y.to_numpy())})
        for ctx, col in CONTEXTS.items():
            if col in g.columns:
                rows.append({"slice": name, "system": f"zeroshot:{ctx}", **scores(g[col].to_numpy(), g.y.to_numpy())})
    for cfg in a.configs:
        col = CONTEXTS[cfg]  # score each run on the context it was trained with
        for seed in a.seeds:
            path = os.path.join(ROOT, f"results/{cfg}_s{seed}_test.parquet")
            if not os.path.exists(path):
                continue
            d = pd.read_parquet(path)
            for name, g in slices(d, thr):
                rows.append({"slice": name, "system": cfg, "seed": seed, **scores(g[col].to_numpy(), g.y.to_numpy())})
    r = pd.DataFrame(rows)
    mean = r.groupby(["slice", "system"], sort=False)[[a.metric, "n"]].mean().reset_index()
    seeds = r.groupby(["slice", "system"], sort=False).seed.nunique().reset_index()
    table = mean.pivot(index="slice", columns="system", values=a.metric)
    table.insert(0, "n", mean.groupby("slice", sort=False).n.max())
    table = table.loc[[s for s in dict.fromkeys(mean.slice)]]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(f"{a.metric} on test (trained configs = mean over seeds; seeds found: "
              f"{dict(zip(seeds.system, seeds.seed))})")
        print(table.to_string())
        per_seed = r[r.slice == "all"].dropna(subset=["seed"])
        print("\nper seed, all markets:")
        print(per_seed[["system", "seed", "brier", "logloss", "ece", "auc"]].to_string(index=False))

    print("\nBrier(model) - Brier(market), seed-averaged predictions, 95% CI over events:")
    for cfg in a.configs:
        paths = [os.path.join(ROOT, f"results/{cfg}_s{s}_test.parquet") for s in a.seeds]
        if not all(os.path.exists(p) for p in paths):
            continue
        col = CONTEXTS[cfg]
        ds = [pd.read_parquet(p) for p in paths]
        assert all((x.market_id.values == ds[0].market_id.values).all() for x in ds)
        d = ds[0].copy()
        d["avg"] = np.mean([x[col].to_numpy() for x in ds], axis=0)
        for name, g in [s for s in slices(d, thr) if s[0] in ("all", "liquid", "thin") or "," not in s[0]]:
            m, lo, hi, k = brier_gap_vs_market(g, "avg")
            print(f"  {cfg:12s} {name:10s} n={len(g):6d} events={k:5d}  {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
