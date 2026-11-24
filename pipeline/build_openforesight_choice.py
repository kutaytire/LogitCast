"""OpenForesight as multiple choice: each question's true answer plus N_DISTRACTORS distractors as options.

Distractors are the answers of the most similar other questions (TF-IDF over question text) with the same
normalised answer type, as in pipeline/build_openforesight.py, which builds the yes/no form from the first of them.
Training-row distractors come from the training pool only. Option order is shuffled with a fixed seed.
There is no crowd, so the reference is a uniform guess. Writes data/openforesight_choice.parquet with the
columns logitcast/train_forecast.py and logitcast/eval_choice.py expect, plus state_json (the same context as one JSON object).
"""
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.build_openforesight import SPLITS, distractors, load, state_text
from pipeline.candidates import ROOT

N_DISTRACTORS = 4


def main():
    parts = {name: load(name) for name in SPLITS}
    train = parts["train"]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit(
        pd.concat([p.question_title for p in parts.values()]))
    rng = np.random.default_rng(0)
    out = []
    for name, d in parts.items():
        pool = train if name == "train" else pd.concat([train, d])
        dis = distractors(d, pool.drop_duplicates("qid"), vec, k=N_DISTRACTORS)
        d = d[dis.notna()].copy()
        options, y = [], []
        for answer, wrong in zip(d.answer, dis[dis.notna()]):
            opts = [answer] + list(wrong)
            order = rng.permutation(len(opts))
            options.append([opts[i] for i in order])
            y.append(int(np.flatnonzero(order == 0)[0]))
        d["options"], d["y_index"] = options, y
        out.append(d)
    o = pd.concat(out, ignore_index=True)
    k = N_DISTRACTORS + 1
    o = o.assign(event_id=o.qid.astype(str), horizon_days=0, category=o.atype, n_options=k,
                 crowd=[[1 / k] * k] * len(o), question=o.question_title.str.strip(),
                 state=[state_text(r) for r in o.itertuples()])
    o["state_price"] = o.state
    o["state_context"] = o.state  # shared with data/choice.parquet (prices + news there) for joint training
    o["state_json"] = [json.dumps({"background": " ".join(r.background.split()),
                                   "resolution_rules": " ".join(re.sub(r"<[^>]+>", " ", r.resolution_criteria).split())},
                                  ensure_ascii=False) for r in o.itertuples()]
    keep = ["event_id", "split", "horizon_days", "category", "options", "n_options", "y_index", "crowd",
            "question", "state", "state_price", "state_json", "state_context", "answer"]
    o[keep].to_parquet(os.path.join(ROOT, "data/openforesight_choice.parquet"))
    print(o.groupby("split").agg(questions=("y_index", "size")))
    for r in o[o.split == "of_fs2026q2"].sample(2, random_state=1).itertuples():
        print(r.question, "|", list(r.options), "| answer", r.y_index)


if __name__ == "__main__":
    main()
