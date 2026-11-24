"""Each OpenForesight training question once: half of the questions as yes/no rows, the other half as multiple choice.

data/openforesight.parquet and data/openforesight_choice.parquet are built from the same training questions, so a
model tuned on both sees every question twice. This writes copies whose train splits partition the questions by a
hash of event_id (questions with no multiple-choice row stay yes/no); val and test splits are kept whole.
Writes data/openforesight_half.parquet and data/openforesight_choice_half.parquet (used by the combined JEV run).
"""
import os
import zlib

import pandas as pd

from pipeline.candidates import ROOT


def main():
    yn = pd.read_parquet(os.path.join(ROOT, "data/openforesight.parquet"))
    mc = pd.read_parquet(os.path.join(ROOT, "data/openforesight_choice.parquet"))
    in_mc = set(mc[mc.split == "train"].event_id)
    to_mc = {e for e in in_mc if zlib.crc32(str(e).encode()) % 2}
    keep_yn = (yn.split != "train") | ~yn.event_id.isin(to_mc)
    keep_mc = (mc.split != "train") | mc.event_id.isin(to_mc)
    for d, keep, name in ((yn, keep_yn, "openforesight_half"), (mc, keep_mc, "openforesight_choice_half")):
        path = os.path.join(ROOT, f"data/{name}.parquet")
        d[keep].reset_index(drop=True).to_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"{name}: {d[keep].split.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
