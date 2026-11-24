"""Select leakage-safe (market, forecast time) candidates from raw Polymarket events.

Leakage rules (each one is enforced here, not downstream):

1. Knowledge cutoff. Every forecast time t is >= CUTOFF, the public release of
   the Qwen3.5-9B backbone. The JEV-9B adapter corpus was audited and carries no
   real-world post-cutoff content, so the backbone release is the binding date.
2. Forecast times are set from SCHEDULED dates only: t = endDate - h. Anchoring
   on closedTime would leak the outcome, because YES-by-deadline markets close
   early and NO markets run to the deadline.
3. The market must exist and still be open at t: createdAt, startDate <= t < closedTime.
   (The price filter at t lives in pipeline/build_dataset.py, since it needs prices.)
4. Splits are by time. A row is in train only if its label was public before
   VAL_START; val/test rows have t >= their split start. Splits are assigned by
   scheduled endDate so no split is biased toward fast-resolving outcomes.
5. Only creation-time fields are ever copied into the context (CONTEXT_FIELDS);
   event fields such as score, tweetCount, finishedTimestamp are results.
6. endDate must agree with the deadline stated in the question. Polymarket extends
   endDate on ladder markets that are still open, while markets that already resolved
   YES keep a stale earlier date, so "closes before its own deadline" signals YES.
7. Markets come from data/raw/markets (pipeline/fetch_markets.py), one record per market, so
   resolved markets inside still-open events are included.
"""
import glob
import json
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTOFF = pd.Timestamp("2026-02-27", tz="UTC")       # Qwen3.5-9B public release
VAL_START = pd.Timestamp("2026-08-01", tz="UTC")
TEST_START = pd.Timestamp("2026-09-01", tz="UTC")
LAST_END = pd.Timestamp("2026-10-07", tz="UTC")      # snapshot 2026-10-14 minus a week to settle
HORIZONS_DAYS = (1, 7, 30)
MIN_VOLUME = 10_000
SHORT_RECURRING_TAGS = {"Up or Down", "5M", "15M", "1H"}

# Allowlist of creation-time fields that may reach the model's context.
CONTEXT_FIELDS = {"event": ("title",), "market": ("question", "description", "groupItemTitle")}

CATEGORY_RULES = [  # first match wins
    ("weather", {"Weather"}),
    ("crypto", {"Crypto", "Crypto Prices", "Bitcoin", "Ethereum"}),
    ("esports", {"Esports"}),
    ("sports", {"Sports"}),
    ("politics", {"Politics", "Elections", "Global Elections", "Geopolitics", "World", "Middle East", "Trump"}),
    ("economy", {"Economy", "Finance", "Equities", "Fed Rates"}),
    ("tech", {"Tech", "AI", "Big Tech"}),
    ("culture", {"Culture", "Movies", "Tweet Markets", "Music"}),
]


MONTHS = {m: i + 1 for i, m in enumerate("january february march april may june july august september "
                                          "october november december".split())}
QUESTION_DEADLINE = re.compile(r"\b(?:by|before)\s+(?:the end of\s+)?(" + "|".join(MONTHS) +
                               r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?", re.I)


def ts(s):
    return pd.Timestamp(s).tz_convert("UTC") if pd.Timestamp(s).tzinfo else pd.Timestamp(s, tz="UTC")


def category(tags):
    for name, keys in CATEGORY_RULES:
        if tags & keys:
            return name
    return "other"


def split_of(t, end, closed):
    if end < VAL_START and closed < VAL_START:
        return "train"
    if VAL_START <= t and end < TEST_START and closed < TEST_START:
        return "val"
    if TEST_START <= t and end < LAST_END:
        return "test"
    return None


def stale_end(question, end):
    """True when the question names a deadline more than a day after endDate (rule 6)."""
    m = QUESTION_DEADLINE.search(question)
    if not m:
        return False
    year = int(m.group(3)) if m.group(3) else end.year
    try:
        deadline = pd.Timestamp(year=year, month=MONTHS[m.group(1).lower()], day=int(m.group(2)), tz="UTC")
    except ValueError:
        return False
    if not m.group(3) and abs((deadline - end).days) > 200:  # year rollover, e.g. "by January 15" in December
        deadline = deadline.replace(year=year + (1 if deadline < end else -1))
    return end.normalize() < deadline - pd.Timedelta(days=1)


def label(m):
    prices = json.loads(m["outcomePrices"])
    if prices == ["1", "0"]:
        return 1
    if prices == ["0", "1"]:
        return 0
    return None  # 50/50 or unresolved


def market_rows(m):
    tags = {t["label"] for t in m.get("tags") or []}
    e = (m.get("events") or [{}])[0]
    if tags & SHORT_RECURRING_TAGS or m.get("outcomes") != '["Yes", "No"]' or m.get("negRiskOther"):
        return
    if float(m.get("volumeNum") or 0) < MIN_VOLUME or m.get("umaResolutionStatus") != "resolved":
        return
    y = label(m)
    if y is None or not all(m.get(k) for k in ("createdAt", "startDate", "endDate", "closedTime", "clobTokenIds")):
        return
    created, start, end, closed = (ts(m[k]) for k in ("createdAt", "startDate", "endDate", "closedTime"))
    if stale_end(m["question"], end):
        return
    opened = max(created, start)
    for h in HORIZONS_DAYS:
        t = (end - pd.Timedelta(days=h)).floor("h")
        if t < CUTOFF or t < opened or t >= closed:
            continue
        split = split_of(t, end, closed)
        if split is None:
            continue
        yield {
            "market_id": m["id"], "event_id": e.get("id"), "split": split, "horizon_days": h,
            "t": t, "end": end, "closed": closed, "opened": opened, "y": y,
            "yes_token": json.loads(m["clobTokenIds"])[0],
            "category": category(tags), "volume": float(m["volumeNum"]),
            **{f"event_{k}": e.get(k) for k in CONTEXT_FIELDS["event"]},
            **{f"market_{k}": m.get(k) for k in CONTEXT_FIELDS["market"]},
        }


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data/raw/markets/*.jsonl"))):
        for line in open(f):
            rows += market_rows(json.loads(line))
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(ROOT, "data/candidates.parquet"))
    print(df.groupby(["split", "horizon_days"]).size().unstack())
    print(df.groupby(["split", "category"]).size().unstack(0))
    print("YES rate by split:", df.groupby("split").y.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
