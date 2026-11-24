"""OpenForesight as multiple choice from ranked candidate lists (data/openforesight_ranked_candidates.csv).

The CSV gives each question up to five candidate answers, most likely first, and a review column: OK, or the
letters of the problematic candidates (e.g. "b,d"). The gold answer comes from the raw data, joined on id.
  1. Review. Flagged candidates are dropped; the rest keep their rank order. Exact repeats are dropped.
  2. Truth match. Every candidate is compared with the gold answer: equal after normalisation, else the authors'
     judge prompt (answer_judge.judge_prompt) answered by gpt-oss-120b (build_openforesight_options.Model.same).
  3. Exactly one correct. If candidates match, the best-ranked one is the true option, in the candidate's own
     wording, and the others that match are dropped. If none matches, the gold answer is the true option.
     The distractors are the non-matching candidates, most likely first, without near-duplicates of the truth or
     of a better-ranked distractor (one key contains the other or their similarity is >= NEAR, keys compared
     without case, accents or punctuation), cut to four, so an inserted truth takes the place of the least likely
     candidate. Near-duplicates go because the strict judge passes spelling variants of the truth (Ahmad /
     Ahmed al-Sharaa) and the candidates cluster variants around it (Aeroprakt A-22, A-32, A-22 Foxbat).
     Questions left with no distractor are dropped.
     An inserted gold answer is rewritten by gpt-oss-120b into the style of its question's candidates, since its
     terse form alone singled it out (leakage/probe_openforesight_ranked.py: 49% against 23% chance). The rewrite is
     kept only if the judge finds it equivalent to the gold answer and it repeats no distractor.
  4. Balanced order. The distractors are shuffled with a seed per question and the truth is put at position
     (j mod n_options) for the split's j-th question with that many options, so every position holds the truth
     equally often.
Judgements and rewrites are cached in data/openforesight_ranked_{judge,restyle}.parquet; the GPU is needed only
for what is not in them.
Writes data/openforesight_ranked.parquet in the columns of data/openforesight_options.parquet that training and
logitcast/eval_choice.py read (splits train, val, of_test, of_fs2026q2; state_context; uniform crowd). Run in the vllm venv (vLLM 0.11.0).
"""
import glob
import os
import re
import unicodedata
import zlib
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from pipeline.build_openforesight import answer_type, state_text
from pipeline.candidates import ROOT

CSV = os.path.join(ROOT, "data/openforesight_ranked_candidates.csv")
RAW = os.path.join(ROOT, "data/raw/openforesight")
CACHE = os.path.join(ROOT, "data/openforesight_ranked_judge.parquet")
RESTYLE = os.path.join(ROOT, "data/openforesight_ranked_restyle.parquet")
OUT = os.path.join(ROOT, "data/openforesight_ranked.parquet")
PREFIX = {"train": "train", "validation": "validation", "test": "test", "futureSim2026Q2": "futuresim_2026q2"}
SPLIT = {"train": "train", "validation": "val", "test": "of_test", "futuresim_2026q2": "of_fs2026q2"}  # as SPLITS
MAX_OPTIONS = 5
NEAR = 0.85


_model = None


def model():
    global _model
    if _model is None:
        from pipeline.build_openforesight_options import Model
        _model = Model()
    return _model


def save(df, path):
    df.to_parquet(path + ".tmp")
    os.replace(path + ".tmp", path)


def norm(a):
    return re.sub(r"\s+", " ", str(a)).strip().lower()


def key(a):
    a = unicodedata.normalize("NFKD", str(a)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", a)


def near(a, b):
    a, b = key(a), key(b)
    if not a or not b:                                   # no ASCII letters or digits (e.g. CJK): exact only
        return a == b
    return a in b or b in a or SequenceMatcher(None, a, b).ratio() >= NEAR


def split_options(r, same, truth=None):
    """(best-ranked candidate matching the gold answer or None, distractors most likely first). Distractors avoid
    near-duplicates of `truth` (default: that candidate, else the gold answer) and of each other."""
    hit = [norm(c) == norm(r.answer) or same[(r.question, r.answer, c)] for c in r.candidates]
    match = next((c for c, h in zip(r.candidates, hit) if h), None)
    kept = [truth or match or r.answer]
    for c, h in zip(r.candidates, hit):
        if not h and not any(near(c, x) for x in kept):
            kept.append(c)
    return match, kept[1:MAX_OPTIONS], sum(hit)


def parse(s):
    """'a. X, b. Y' -> {'a': 'X', 'b': 'Y'}; option text may itself contain ', '."""
    parts = re.split(r"(?:^|, )([a-z])\. ", s)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def load():
    d = pd.read_csv(CSV)
    raw = []
    for name, p in PREFIX.items():
        r = pd.concat([pd.read_parquet(f, columns=["qid", "answer"]) for f in glob.glob(f"{RAW}/{name}-*.parquet")])
        raw.append(r.assign(id=p + "-" + r.qid.astype(str))[["id", "answer"]])
    d = d.merge(pd.concat(raw), on="id", how="left", validate="1:1")
    assert d.answer.notna().all()
    kept = []
    for s, rv in zip(d.candidate_answers, d.review):
        bad = set() if rv == "OK" else set(rv.split(","))
        seen, keep = set(), []
        for letter, c in parse(s).items():
            if letter not in bad and norm(c) not in seen:
                seen.add(norm(c))
                keep.append(c)
        kept.append(keep)
    return d.assign(candidates=kept, n_flagged=[0 if rv == "OK" else len(rv.split(",")) for rv in d.review])


def judge(d):
    pairs = pd.DataFrame([(q, a, c) for q, a, cs in zip(d.question, d.answer, d.candidates) for c in cs
                          if norm(c) != norm(a)], columns=["question", "answer", "candidate"]).drop_duplicates()
    cache = pd.read_parquet(CACHE) if os.path.exists(CACHE) else pairs.iloc[:0].assign(match=False)
    todo = pairs.merge(cache, how="left", on=["question", "answer", "candidate"])
    todo = todo[todo["match"].isna()].drop(columns="match")
    print(f"judge: {len(pairs)} pairs, {len(todo)} not cached", flush=True)
    if len(todo):
        same = model().same(list(todo.itertuples(index=False, name=None)))
        cache = pd.concat([cache, todo.assign(match=same)], ignore_index=True)
        save(cache, CACHE)
    return {(q, a, c): m for q, a, c, m in cache.itertuples(index=False, name=None)}


def restyle_prompt(question, answer, options):
    listed = "\n".join(f"- {o}" for o in options)
    return f"""These are answer options of a multiple-choice question.

Question: {question}
Options:
{listed}

Rewrite the new option below so that its form matches the options above: the same capitalisation, the same use of articles, full names or abbreviations, the same level of detail and qualifiers (for example a country after a city, a first name with a surname). Do not change what it refers to, and do not make it more or less specific in content. If it already matches, return it unchanged.

New option: {answer}

Put only the rewritten option inside <option> </option> tags."""


def restyle(d, same):
    """Gold answer -> rewrite in its question's candidates' style, for the questions where it is inserted."""
    rows = []
    for r in d.itertuples(index=False):
        match, distractors, _ = split_options(r, same)
        if match is None and distractors:
            rows.append((r.id, r.question, r.answer, distractors))
    cache = pd.read_parquet(RESTYLE) if os.path.exists(RESTYLE) else pd.DataFrame(columns=["id", "answer", "rewrite", "ok"])
    done = set(zip(cache.id, cache.answer))
    todo = [t for t in rows if (t[0], t[2]) not in done]
    print(f"restyle: {len(rows)} inserted answers, {len(todo)} not cached", flush=True)
    if todo:
        m = model()
        outs = m.ask([m.prompt(restyle_prompt(q, a, ds), "low") for _, q, a, ds in todo], 1024)
        rw = [(re.search(r"<option>(.*?)</option>", o, re.S) or [None, ""])[1].strip() for o in outs]
        need = [(q, a, w) for (_, q, a, _), w in zip(todo, rw) if w and norm(w) != norm(a)]
        eq = dict(zip(need, m.same(need))) if need else {}
        ok = [bool(w) and norm(w) not in {norm(x) for x in ds} and (norm(w) == norm(a) or eq[(q, a, w)])
              for (_, q, a, ds), w in zip(todo, rw)]
        new = pd.DataFrame({"id": [t[0] for t in todo], "answer": [t[2] for t in todo], "rewrite": rw, "ok": ok})
        cache = pd.concat([cache, new], ignore_index=True)
        save(cache, RESTYLE)
    return {(i, a): w for i, a, w, ok in cache[["id", "answer", "rewrite", "ok"]].itertuples(index=False) if ok}


def build(d, same, styled):
    rows, count = [], {}
    for r in d.itertuples(index=False):
        match, _, n_matching = split_options(r, same)
        truth = match or styled.get((r.id, r.answer), r.answer)
        _, distractors, _ = split_options(r, same, truth)
        if not distractors:
            continue
        rng = np.random.default_rng(zlib.crc32(r.id.encode()))
        distractors = [distractors[i] for i in rng.permutation(len(distractors))]
        k = len(distractors) + 1
        j = count.get((r.split, k), 0)
        count[(r.split, k)] = j + 1
        y = j % k
        rows.append(dict(id=r.id, event_id=r.id.rsplit("-", 1)[1], split=SPLIT[r.split],
                         question_start_date=r.question_date, horizon_days=0, question=r.question,
                         background=r.background, resolution_criteria=r.resolution_criteria,
                         state_context=state_text(r), answer_type=r.answer_type,
                         category=answer_type(r.answer_type), answer=r.answer, truth=truth,
                         options=distractors[:y] + [truth] + distractors[y:], y_index=y, n_options=k,
                         crowd=np.full(k, 1 / k),
                         truth_inserted=match is None, truth_restyled=match is None and truth != r.answer,
                         n_matching=n_matching, n_flagged=r.n_flagged))
    return pd.DataFrame(rows)


def main():
    d = load()
    same = judge(d)
    o = build(d, same, restyle(d, same))
    save(o, OUT)
    print(f"{len(o)} of {len(d)} questions kept -> {OUT}")
    print("inserted truth restyled:", round(o[o.truth_inserted].truth_restyled.mean(), 3))
    print(pd.crosstab(o.split, o.n_options))
    print("truth inserted (not among valid candidates):", o.groupby("split").truth_inserted.mean().round(3).to_dict())
    print("questions with >1 matching candidate:", (o.n_matching > 1).sum())
    print("truth position share:\n", pd.crosstab(o.split, o.y_index, normalize="index").round(3))


if __name__ == "__main__":
    main()
