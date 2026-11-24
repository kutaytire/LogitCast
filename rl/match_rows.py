"""Write the written arms' training rows as logitcast/train_forecast.py data, so the logit arm trains on the same questions.

rl/grpo_stated.py trains on the first STEPS x QUESTIONS rows of training_rows / mc_training_rows (seed 0), each under
one input. This writes, per task and input, those train rows (all columns) plus the source file's full val split:
    data/rlmatch/forecast_{informed,blind}.parquet   (data/forecast.parquet; contexts state_json_price, state)
    data/rlmatch/choice_{informed,blind}.parquet     (data/choice.parquet, data/mcexp/choice_blind.parquet;
                                                      context state_context)
    python -m rl.match_rows      (in the rlro environment)
"""
import os

import pandas as pd

from pipeline.candidates import ROOT
from rl.grpo_stated import INPUTS, MC_FILES, mc_training_rows, training_rows

N = 500 * 32  # grpo_stated.py defaults: --steps x --questions
TAGS = ("informed", "blind")  # the order of INPUTS and MC_FILES


def subset(src, unit, chosen, out):
    d = pd.read_parquet(os.path.join(ROOT, src))
    key = lambda f: f[unit].astype(str) + "|" + f.horizon_days.astype(str)  # noqa: E731
    tr = d[d.split == "train"]
    assert not key(tr).duplicated().any()
    keep = tr[key(tr).isin(set(key(chosen)))]
    assert len(keep) == len(chosen), (out, len(keep), len(chosen))
    pd.concat([keep, d[d.split == "val"]], ignore_index=True).to_parquet(os.path.join(ROOT, out))
    print(f"{out}: {len(keep)} train rows", flush=True)


def main():
    os.makedirs(os.path.join(ROOT, "data/rlmatch"), exist_ok=True)
    b = training_rows(0).iloc[:N]
    for inp, tag in zip(INPUTS, TAGS):
        subset("data/forecast.parquet", "market_id", b[b.input == inp], f"data/rlmatch/forecast_{tag}.parquet")
    m = mc_training_rows(0).iloc[:N]
    for f, tag in zip(MC_FILES, TAGS):
        subset(f, "event_id", m[m.input == f], f"data/rlmatch/choice_{tag}.parquet")


if __name__ == "__main__":
    main()
