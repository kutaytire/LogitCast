"""Turn OpenForesight's open-ended questions into yes/no rows that both JEV-9B and OpenForecaster-8B can answer.

For each question, the candidate answer is either the true answer (y=1) or a distractor (y=0): the answer of
the most similar other question (TF-IDF over question text) with the same normalised answer type and a
different answer. Distractors for training rows come from the training pool only.
  train            one row per question, true or distractor with probability 1/2 (seeded)
  val, of_test,    both rows per question (balanced)
  of_fs2026q2
of_test (May-Aug 2025) predates JEV-9B's backbone release, so JEV may know the answers (known leakage, kept
for comparison with the authors' test). of_fs2026q2 (resolves Apr-Jun 2026) is after both models' cutoffs.
Context = OpenForesight's background and resolution criteria, no retrieval. Writes data/openforesight.parquet
with the columns logitcast/train_forecast.py, logitcast/eval_forecast.py and baselines/baseline_openforecaster.py expect.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.candidates import ROOT

RAW = os.path.join(ROOT, "data/raw/openforesight")
SPLITS = {"train": "train", "validation": "val", "test": "of_test", "futureSim2026Q2": "of_fs2026q2",
          "aljazeera2026Q3": "of_q3"}
SYNONYMS = {"person name": "name", "person": "name", "country name": "country", "team name": "team",
            "city": "location", "place": "location"}


def answer_type(t):
    t = re.sub(r"\s+", " ", t.lower()).strip()
    t = re.sub(r"^string\s*\((.*)\)$", r"\1", t).strip()
    return SYNONYMS.get(t, t)


def norm(a):
    return re.sub(r"\s+", " ", str(a)).strip().lower()


def load(name):
    d = pd.concat([pd.read_parquet(f, columns=["qid", "question_title", "background", "resolution_criteria",
                                               "answer", "answer_type"])
                   for f in sorted(glob.glob(f"{RAW}/{name}-*.parquet"))], ignore_index=True)
    return d.assign(split=SPLITS[name], atype=d.answer_type.map(answer_type), key=d.answer.map(norm))


def distractors(queries, pool, vec, k=1):
    """For each query question, the answers of its k most similar pool questions of the same type, each different
    from the true answer and from each other (k=1: one answer, None if the type has no other answer; k > 1: a list
    of up to k answers, None if there are none)."""
    out = pd.Series(index=queries.index, dtype=object)
    for t, q in queries.groupby("atype"):
        p = pool[pool.atype == t]
        if len(p) == 0:
            continue
        sims = (vec.transform(q.question_title) @ vec.transform(p.question_title).T).toarray()
        pk, pa = p.key.to_numpy(), p.answer.to_numpy()
        for row, (i, key) in enumerate(zip(q.index, q.key)):
            seen, picks = {key}, []
            for j in np.argsort(-sims[row]):
                if pk[j] not in seen:
                    seen.add(pk[j])
                    picks.append(pa[j])
                    if len(picks) == k:
                        break
            if picks:
                out[i] = picks[0] if k == 1 else picks
    return out


def state_text(r):
    crit = " ".join(re.sub(r"<[^>]+>", " ", r.resolution_criteria).split())
    return f"Background: {' '.join(r.background.split())}\nResolution rules: {crit}"


def rows(d, cand, y, tag):
    return d.assign(candidate=cand, y=y, market_id=d.qid.astype(str) + "_" + tag)


def main():
    parts = {name: load(name) for name in SPLITS}
    train = parts["train"]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit(
        pd.concat([p.question_title for p in parts.values()]))
    out = []
    rng = np.random.default_rng(0)
    for name, d in parts.items():
        pool = train if name == "train" else pd.concat([train, d])
        dis = distractors(d, pool.drop_duplicates("qid"), vec)
        d = d[dis.notna()]
        dis = dis[dis.notna()]
        if name == "train":
            pos = rng.random(len(d)) < 0.5
            out.append(rows(d[pos], d.answer[pos], 1, "pos"))
            out.append(rows(d[~pos], dis[~pos], 0, "neg"))
        else:
            out += [rows(d, d.answer, 1, "pos"), rows(d, dis, 0, "neg")]
    o = pd.concat(out, ignore_index=True)
    o["question"] = [f'{t.strip()} Will the answer be "{c}"?' for t, c in zip(o.question_title, o.candidate)]
    o["state"] = [state_text(r) for r in o.itertuples()]
    o["state_price"] = o.state
    o = o.assign(event_id=o.qid.astype(str), horizon_days=0, category=o.atype, activity_7d=0, p_t=0.5)
    keep = ["market_id", "event_id", "split", "horizon_days", "category", "activity_7d", "y", "p_t",
            "state", "state_price", "question", "candidate", "answer"]
    o[keep].to_parquet(os.path.join(ROOT, "data/openforesight.parquet"))
    print(o.groupby("split").agg(rows=("y", "size"), questions=("event_id", "nunique"), yes=("y", "mean")).round(3))
    for r in o[o.split == "of_fs2026q2"].sample(4, random_state=1).itertuples():
        print(f"\ny={r.y} | {r.question}\n   true answer: {r.answer}")


if __name__ == "__main__":
    main()
