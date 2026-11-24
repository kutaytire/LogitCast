"""Add structured (JSON) versions of the contexts.

data/forecast.parquet
  state_json_price       same information as state_price, as one JSON object
  state_json_price_news  same information as state_price_news, as one JSON object
data/openforesight.parquet
  state_json             same information as state (background and resolution rules), as one JSON object
data/{forecast,manifold,kalshi,forecastbench}.parquet (blind_json)
  state_json             state_json_price without the market prices or ForecastBench's latest value, so the
                         market-blind variant can be read in the same format as the market-informed one
Fields come from the same sources as the prose contexts (event, option, close and rules from the prose state,
the three prices from the dataset, headlines from the news cache), so only the format differs. No network access.
"""
import json
import os

import pandas as pd

from pipeline.candidates import ROOT
from pipeline.fetch_news import queries
from pipeline.news import headlines


def prose_fields(state):
    f = {}
    for line in state.split("\n"):
        for key, prefix in (("event", "Event: "), ("option", "Option: "), ("rules", "Resolution rules: ")):
            if line.startswith(prefix):
                f[key] = line.removeprefix(prefix)
    return f


def save(d, path):
    tmp = path + ".tmp"
    d.to_parquet(tmp)
    os.replace(tmp, path)


def openforesight():
    path = os.path.join(ROOT, "data/openforesight.parquet")
    d = pd.read_parquet(path)
    keys = {"Background: ": "background", "Resolution rules: ": "resolution_rules"}
    out = []
    for state in d.state:
        obj = {}
        for line in state.split("\n"):
            prefix = next(k for k in keys if line.startswith(k))
            obj[keys[prefix]] = line.removeprefix(prefix)
        out.append(json.dumps(obj, ensure_ascii=False))
    d["state_json"] = out
    save(d, path)
    print(json.dumps(json.loads(out[0]), indent=1, ensure_ascii=False)[:600])


def main():
    path = os.path.join(ROOT, "data/forecast.parquet")
    d = pd.read_parquet(path)
    q = queries(d[["state", "question", "t"]])
    price_json, news_json = [], []
    for r, query in zip(d.itertuples(), q["query"]):
        f = prose_fields(r.state)
        obj = {"today": f"{r.t:%Y-%m-%d %H:%M} UTC", "event": f.get("event")}
        if "option" in f:
            obj["option"] = f["option"]
        obj.update({"market_closes": f"{r.end:%Y-%m-%d %H:%M} UTC", "resolution_rules": f.get("rules"),
                    "market": {"probability_yes_now": round(r.p_t, 2), "probability_yes_one_day_ago": round(r.p_1d, 2),
                               "probability_yes_seven_days_ago": round(r.p_7d, 2)}})
        price_json.append(json.dumps(obj, ensure_ascii=False))
        items = headlines(query, r.t, offline=True)
        obj["news_headlines"] = [{"date": pd.Timestamp(i["published"]).strftime("%Y-%m-%d"), "outlet": i["source"],
                                  "headline": i["title"].rsplit(" - ", 1)[0]} for i in items]
        news_json.append(json.dumps(obj, ensure_ascii=False))
    d["state_json_price"], d["state_json_price_news"] = price_json, news_json
    save(d, path)
    ex = d[(d.market_id == "4186291") & (d.horizon_days == 7)].iloc[0]
    print(json.dumps(json.loads(ex.state_json_price_news), indent=1, ensure_ascii=False)[:1500])


def blind_json():
    for name in ("forecast", "manifold", "kalshi", "forecastbench"):
        path = os.path.join(ROOT, f"data/{name}.parquet")
        d = pd.read_parquet(path)
        d["state_json"] = [json.dumps({k: v for k, v in json.loads(s).items() if k not in ("market", "latest_value")},
                                      ensure_ascii=False) for s in d.state_json_price]
        save(d, path)
        print(name, d.state_json.iloc[0][:300])


if __name__ == "__main__":
    main()
    openforesight()
    blind_json()
