"""FutureX-Past (futurex-ai/Futurex-Past, revision in data/raw/futurex/REVISION) as a multiple-choice test set whose
options were written by another benchmark, not by our option protocol.

Rules:
  1. Only questions whose event ends after TRAIN_LABEL_END (Aug 1 2026), so no tuned model learned the outcome.
  2. Only questions that list lettered options (A., B., ...) and record exactly one of them as correct. FutureX asks
     for "all the correct option(s)", so a question can have several true options; the decision head assumes one.
  3. Options must be mutually exclusive by the question's wording, not only by its recorded answer: questions that
     ask for several items ("which candidates", "which teams", ...) or whose options are overlapping thresholds
     (two or more options opening with "at least", "above", ... in the same direction and no option a range, so
     several can hold at once) are dropped. Bands ("below 10%", "10% to under 12%", "12% or higher") are exclusive
     and kept.
  4. At most MAX_OPTIONS options (as for Polymarket and Manifold); larger sets are dropped, not cut, since the
     options carry no prices to rank them.
  5. FutureX releases each week's questions about a week before their events end, and the question carries no
     forecast date, so the forecast date t is set to the end date minus HORIZON_DAYS.
Context: the date, the question and its resolution date (no background exists). The options are the benchmark's own,
with its "the outcome be" prefix removed. Writes data/futurex_choice.parquet (split="futurex") in the layout of
data/choice.parquet; "crowd" is uniform, as there is no market.
"""
import ast
import json
import os
import re

import numpy as np
import pandas as pd

from pipeline.build_choice import MAX_OPTIONS
from pipeline.build_manifold import TRAIN_LABEL_END
from pipeline.candidates import ROOT

RAW = os.path.join(ROOT, "data/raw/futurex/past.parquet")
HORIZON_DAYS = 7
OPTION = re.compile(r"^\s*([A-Z])\.\s+(.*?)\s*$")
PLURAL = re.compile(r"\bwhich (candidates|teams|players|countries|parties|companies|ones|stocks|options|two|three)\b"
                    r"|\ball (?:of )?the (?:correct|options)\b", re.I)
UP = re.compile(r"^(at least|more than|above|over|exceeds?|[≥>])", re.I)  # an option that opens with a direction
DOWN = re.compile(r"^(at most|less than|fewer than|below|under|[≤<])", re.I)
RANGE = re.compile(r"\d\s*(?:to|-|–|and)\s*(?:under |below |less than )?[\d$€£]|\bbetween\b|\bbut\b", re.I)


def parse(prompt):
    """(question, resolution text, options) from a FutureX prompt, or None if it lists no lettered options."""
    m = re.search(r'The event to be predicted: "(.*?)"\s*\n\s*IMPORTANT', prompt, re.S)
    if not m:
        return None
    body = m.group(1).strip().strip('"')
    lines = body.split("\n")
    opts = [OPTION.match(l) for l in lines]
    if not any(opts):
        return None
    first = next(i for i, o in enumerate(opts) if o)
    head = " ".join(" ".join(lines[:first]).split())
    letters = [o.group(1) for o in opts if o]
    if letters != [chr(65 + i) for i in range(len(letters))]:
        return None
    names = [re.sub(r"^(?:the outcome be\s+)?(?:[A-Z]\.\s+)?", "", o.group(2), flags=re.I).strip() for o in opts if o]
    res = re.search(r"\(resolved around (.*)\)\.?\s*$", head)
    question = head[:res.start()].strip() if res else head
    return question, res.group(1) if res else None, names


def day(ts):
    return f"{ts:%A, %B} {ts.day}, {ts.year}"


def main():
    d = pd.read_parquet(RAW)
    end = pd.to_datetime(d.end_time.astype(str).str.split(" ").str[0].str.split("T").str[0], format="mixed")
    end = end.dt.tz_localize("UTC")
    counts, out, dropped = {"all": len(d)}, [], []
    for r, e in zip(d.itertuples(), end):
        if e <= TRAIN_LABEL_END:
            continue
        counts["after Aug 1"] = counts.get("after Aug 1", 0) + 1
        p = parse(r.prompt)
        if p is None:
            continue
        counts["lettered options"] = counts.get("lettered options", 0) + 1
        question, resolves, names = p
        truth = ast.literal_eval(r.ground_truth) if str(r.ground_truth).startswith("[") else [r.ground_truth]
        if len(truth) != 1 or not re.fullmatch(r"[A-Z]", str(truth[0])) or ord(truth[0]) - 65 >= len(names):
            continue
        counts["one correct option"] = counts.get("one correct option", 0) + 1
        cumulative = not any(RANGE.search(o) for o in names) and \
            any(sum(bool(x.search(o)) for o in names) >= 2 for x in (UP, DOWN))
        if PLURAL.search(question) or cumulative:
            dropped.append((question, names))
            continue
        counts["mutually exclusive"] = counts.get("mutually exclusive", 0) + 1
        if len(names) > MAX_OPTIONS or len(names) < 2:
            continue
        counts[f"2..{MAX_OPTIONS} options"] = counts.get(f"2..{MAX_OPTIONS} options", 0) + 1
        t = e - pd.Timedelta(days=HORIZON_DAYS)
        state = (f"Today is {day(t)}.\nEvent: {question}\nThe question resolves around "
                 f"{resolves or day(e)}.")
        obj = {"today": f"{t:%Y-%m-%d} UTC", "event": question, "resolves": resolves or f"{e:%Y-%m-%d}"}
        out.append({"event_id": r.id, "split": "futurex", "horizon_days": HORIZON_DAYS, "t": t, "end": e,
                    "category": f"level{r.level}", "title": question, "question": question, "options": names,
                    "n_options": len(names), "y_index": ord(truth[0]) - 65, "crowd": [1 / len(names)] * len(names),
                    "state": state, "state_price": state, "state_context": state,
                    "state_json": json.dumps(obj, ensure_ascii=False)})
    o = pd.DataFrame(out)
    o.to_parquet(os.path.join(ROOT, "data/futurex_choice.parquet"))
    print(" -> ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"kept {len(o)} | ends {o.end.min():%Y-%m-%d}..{o.end.max():%Y-%m-%d} | options "
          f"{o.n_options.value_counts().sort_index().to_dict()} | levels {o.category.value_counts().to_dict()} | "
          f"uniform accuracy {np.mean(1 / o.n_options):.3f}")
    print(f"dropped as not mutually exclusive: {len(dropped)}, e.g.")
    for q, names in pd.Series(dropped).sample(min(8, len(dropped)), random_state=0):
        print(f"  x {q} | {names}")
    for r in o.sample(min(6, len(o)), random_state=0).itertuples():
        print(f"- {r.question} | {list(r.options)} | true: {r.options[r.y_index]}")


if __name__ == "__main__":
    main()
