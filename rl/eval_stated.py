"""Score rl/grpo_stated.py checkpoints: what each writes, greedy, on the paper's Polymarket rows.

--task bin: the paper's 2,000-question yes/no test sample (paper.tables.yesno_rows("poly"): test split, seed 0) and
  2,000 val questions (seed 0) for the temperature, each under both inputs (with prices: state_json_price, without:
  state). An unparseable answer is stored as NaN (scored as 0.5). Columns market_id, horizon_days, split, input, y, p.
--task mc: the paper's 2,000-row multiple-choice test sample (paper.tables.choice_preds("poly"): data/choice.parquet
  test split, seed 0) and 2,000 val rows (seed 0), each under both inputs (with prices: data/choice.parquet,
  without: data/mcexp/choice_blind.parquet, both state_context). p is the written distribution, None if
  unparseable (scored as uniform). Columns event_id, horizon_days, split, input, y_index, p.
Step 0 is the untuned backbone. Writes results/{run_name}/s{SEED}_step{STEP}.parquet (grpo_stated.run_name).
    python -m rl.eval_stated --model 9b --task mc --seed 0 --steps 0 100 200 300 400 500      (in the rlro environment)
"""
import argparse
import os

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from pipeline.candidates import ROOT
from rl.grpo_stated import INPUTS, MC_FILES, MODELS, parse, parse_dist, run_name, text

ROWS = 2000


def bin_rows():
    d = pd.read_parquet(os.path.join(ROOT, "data/forecast.parquet"),
                        columns=["split", "market_id", "horizon_days", "y", "question", *INPUTS])
    out = []
    for split in ("test", "val"):
        s = d[d.split == split].reset_index(drop=True).sample(ROWS, random_state=0).reset_index(drop=True)
        out += [s.assign(input=c, state=s[c]) for c in INPUTS]
    return pd.concat(out, ignore_index=True)


def mc_rows():
    """The same questions under both inputs, sampled from data/choice.parquet as the paper's tables are."""
    cols = ["split", "event_id", "horizon_days", "y_index", "question", "options", "state_context"]
    frames = [pd.read_parquet(os.path.join(ROOT, f), columns=cols) for f in MC_FILES]
    key = lambda d: d.event_id.astype(str) + "|" + d.horizon_days.astype(str)  # noqa: E731
    out = []
    for split in ("test", "val"):
        s = frames[0][frames[0].split == split].reset_index(drop=True).sample(ROWS, random_state=0)
        for f, d in zip(MC_FILES, frames):
            d = d[d.split == split]
            d = d.set_index(key(d)).loc[key(s).to_numpy()].reset_index(drop=True)
            assert (d.y_index.to_numpy() == s.y_index.to_numpy()).all()
            out.append(d.assign(input=os.path.basename(f), state=d.state_context))
    return pd.concat(out, ignore_index=True)


def mc_brier(g, options):
    """Multi-class Brier of written distributions (uniform where unparsed)."""
    out = []
    for q, o, y in zip(g.p, options.loc[g.index], g.y_index):
        p = np.full(len(o), 1 / len(o)) if q is None else np.asarray(q)
        out.append(((p - np.eye(len(o))[y]) ** 2).sum())
    return np.mean(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="9b")
    ap.add_argument("--task", choices=["bin", "mc"], default="bin")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.5)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    mc = a.task == "mc"
    d = mc_rows() if mc else bin_rows()
    prompts = [text(tok, s, q, o) for s, q, o in zip(d.state, d.question, d.options)] if mc else \
        [text(tok, s, q) for s, q in zip(d.state, d.question)]
    llm = LLM(MODELS[a.model], dtype="bfloat16", max_model_len=2048 if mc else 1024, gpu_memory_utilization=a.gpu_mem, seed=0,
              max_num_seqs=256, enable_lora=True, max_lora_rank=16)
    name = run_name(a.model, a.task)
    os.makedirs(os.path.join(ROOT, f"results/{name}"), exist_ok=True)
    for step in a.steps:
        lora = None if step == 0 else LoRARequest(
            f"{a.task}{a.seed}_{step}", step, os.path.join(ROOT, f"runs/{name}/s{a.seed}/checkpoint-{step}"))
        gen = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=160 if mc else 16), lora_request=lora)
        if mc:
            p = [parse_dist(o.outputs[0].text, len(k)) for o, k in zip(gen, d.options)]
            out = d[["event_id", "horizon_days", "split", "input", "y_index"]].assign(
                p=[None if x is None else x.tolist() for x in p])
            te = out[out.split == "test"]
            ok = te.p.notna()
            print(f"mc s{a.seed} step {step}: parsed {ok.mean():.1%}, test raw Brier " + ", ".join(
                f"{c} {mc_brier(te[te.input == c], d.options):.4f}" for c in te.input.unique()), flush=True)
        else:
            p = [parse(o.outputs[0].text) for o in gen]
            out = d[["market_id", "horizon_days", "split", "input", "y"]].assign(p=[np.nan if x is None else x for x in p])
            te = out[out.split == "test"]
            print(f"s{a.seed} step {step}: parsed {out.p.notna().mean():.1%}, test raw Brier " + ", ".join(
                f"{c} {np.mean((te[te.input == c].p.fillna(0.5) - te[te.input == c].y) ** 2):.4f}" for c in INPUTS),
                flush=True)
        out.to_parquet(os.path.join(ROOT, f"results/{name}/s{a.seed}_step{step}.parquet"))


if __name__ == "__main__":
    main()
