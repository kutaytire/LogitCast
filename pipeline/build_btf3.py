"""Bench to the Future 3 (BTF-2/BTF-3, binary config; revision in data/raw/btf3/REVISION) as a yes/no test set
with substantive background and no market price.

Questions were asked on a present date in late April to late May 2026 and resolved mid-May to early July 2026,
after the backbone's release (Feb 27 2026) but inside the Polymarket training period (labels before Aug 1 2026).
A tuned model may therefore have learned the outcome of the same event from a Polymarket training market;
Such questions are flagged and reported separately.
The context is BTF-3's own text as of the present date, in the layout of pipeline/build_forecastbench.py: the date, the
question, the expected resolution date, the resolution criteria and the background (markdown links reduced to
their text, cut to TEXT_CHARS). The reference forecast, its rationale and the resolution explanation are never
shown. Writes data/btf3.parquet (split="btf3") with the columns logitcast/eval_forecast.py and baselines/baseline_openforecaster.py
expect.
"""
import os
import re

import numpy as np
import pandas as pd

from pipeline.build_forecastbench import TEXT_CHARS, day
from pipeline.candidates import ROOT

RAW = os.path.join(ROOT, "data/raw/btf3/binary.parquet")


def plain(text):
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", str(text))  # [text](url) -> text
    return " ".join(re.sub(r"[*_`#]+", "", text).split())


def main():
    d = pd.read_parquet(RAW)
    t = pd.to_datetime(d.present_date, format="mixed").dt.tz_localize("UTC")
    end = pd.to_datetime(d.expected_resolution_date, format="mixed").dt.tz_localize("UTC")
    state = [f"Today is {day(a)}.\nEvent: {q}\nThe question resolves on {day(b)}.\n"
             f"Resolution rules: {plain(c)}\nBackground: {plain(g)[:TEXT_CHARS]}"
             for a, b, q, c, g in zip(t, end, d.question, d.resolution_criteria, d.background)]
    o = pd.DataFrame({"market_id": d.question_id, "event_id": d.question_id, "split": "btf3",
                      "horizon_days": (end - t).dt.days, "t": t, "end": end, "y": d.resolution.astype(int),
                      "question": d.question, "category": "btf3", "origin": "btf3", "p_t": np.nan,
                      "state": state, "state_price": state})
    o.to_parquet(os.path.join(ROOT, "data/btf3.parquet"))
    print(f"{len(o)} questions | present {o.t.min():%Y-%m-%d}..{o.t.max():%Y-%m-%d} | resolves "
          f"{o.end.min():%Y-%m-%d}..{o.end.max():%Y-%m-%d} | yes rate {o.y.mean():.3f} | horizon median "
          f"{o.horizon_days.median():.0f} d | context chars median {o.state.str.len().median():.0f}")
    print(o.state.iloc[0])


if __name__ == "__main__":
    main()
