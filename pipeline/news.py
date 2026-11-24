"""Date-restricted news retrieval from Google News RSS, cached on disk.

Leakage rules:
  - The query window is [t - LOOKBACK, day(t)): Google's `before:` operator is day-granular, so only
    articles published on calendar days strictly before t's day are requested.
  - Every returned item is re-checked: its pubDate must exist and fall before day(t); items failing
    that are dropped, never kept.
  - Only the headline and outlet are used (no article bodies, which can be edited after publication).
Results are cached per (query, window) in data/raw/news/ so reruns make no new requests.
"""
import email.utils
import hashlib
import html
import json
import os
import re
import time
import urllib.parse

import pandas as pd
import requests

from pipeline.candidates import ROOT

CACHE = os.path.join(ROOT, "data/raw/news")
LOOKBACK = pd.Timedelta(days=14)
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0 (jev-forecast-research)"
MIN_INTERVAL = 1.0  # seconds between live requests
_last = [0.0]


def query_for(event_title, question):
    """Search query: the event title with date phrases and punctuation removed."""
    q = event_title or question
    q = re.sub(r"\b(on|by|before|after|in)?\s*(january|february|march|april|may|june|july|august|september|"
               r"october|november|december)\s+\d{1,2}(st|nd|rd|th)?(,?\s+20\d\d)?", " ", q, flags=re.I)
    q = re.sub(r"\s+-\s+(More Markets|Exact Score|Halftime Result|Player Props?|Totals?|Spreads?)\s*$", " ", q, flags=re.I)
    q = re.sub(r"[?#\"“”]|\.\.\.", " ", q)
    return re.sub(r"\s+", " ", q).strip(" -")


def _fetch(query, start_day, end_day):
    url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(
        f"{query} after:{start_day:%Y-%m-%d} before:{end_day:%Y-%m-%d}") + "&hl=en-US&gl=US&ceid=US:en")
    for attempt in range(6):
        wait = MIN_INTERVAL - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = SESSION.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.text
        time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"news request failed: {url}")


def _parse(xml):
    out = []
    for it in re.findall(r"<item>(.*?)</item>", xml, re.S):
        title = re.search(r"<title>(.*?)</title>", it, re.S)
        date = re.search(r"<pubDate>(.*?)</pubDate>", it)
        src = re.search(r"<source[^>]*>(.*?)</source>", it, re.S)
        if not (title and date):
            continue
        out.append({"title": html.unescape(title.group(1).strip()),
                    "source": html.unescape(src.group(1).strip()) if src else "",
                    "published": email.utils.parsedate_to_datetime(date.group(1)).isoformat()})
    return out


def headlines(query, t, max_items=10, start_day=None, offline=False):
    """Headlines about `query` published on days strictly before day(t), newest first.
    start_day overrides the lookback (used only by the leakage placebo, which passes a t after resolution).
    offline=True reads the cache only and raises if the request was never made."""
    end_day = t.normalize()
    start_day = start_day.normalize() if start_day is not None else end_day - LOOKBACK
    key = hashlib.sha1(f"{query}|{start_day:%Y-%m-%d}|{end_day:%Y-%m-%d}".encode()).hexdigest()
    path = os.path.join(CACHE, key[:2], key + ".json")
    if os.path.exists(path):
        items = json.load(open(path))["items"]
    elif offline:
        raise KeyError(f"no cached news for {query!r} before {end_day:%Y-%m-%d}")
    else:
        items = _parse(_fetch(query, start_day, end_day))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        json.dump({"query": query, "start": str(start_day), "end": str(end_day), "items": items}, open(tmp, "w"))
        os.replace(tmp, path)
    kept = [i for i in items if pd.Timestamp(i["published"]).tz_convert("UTC") < end_day]
    kept.sort(key=lambda i: i["published"], reverse=True)
    return kept[:max_items]


def news_text(items):
    """Context block for the model: one line per headline, dated, outlet suffix removed."""
    lines = []
    for i in items:
        title = re.sub(r"\s+-\s+[^-]+$", "", i["title"])  # Google appends " - Outlet"
        lines.append(f"- {pd.Timestamp(i['published']):%b %-d}: {title} ({i['source']})")
    return "Recent headlines:\n" + "\n".join(lines) if lines else "Recent headlines: none found."
