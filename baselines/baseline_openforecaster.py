"""Baseline: OpenForecaster-8B (Chandak et al., arXiv 2512.25070) on our forecasting rows.

Follows the authors' binary protocol (scaling-forecasting-training/custom_eval_scripts/eval_binary_retrieval.py):
their `format_forecasting_prompt_binary` prompt, Qwen3 chat template with thinking enabled, temperature 0.6,
top_p 0.95, probability read from the last <probability> tag after </think>. Generations per row are averaged;
a row with no parseable probability gets 0.5 and is counted in `parse_fail`. Each generation's probability (NaN if
unparseable) and length are also kept, in `samples` and `sample_tokens`, for test-time scaling curves.

Our fields map to their prompt as: Question Title = market question; Question Background = forecast date,
event, scheduled close (plus the price line in the `price` variant); Resolution Criteria = market rules.
Run in the vllm venv (vLLM 0.11.0).
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from pipeline.candidates import ROOT
MODEL = "nikhilchandak/OpenForecaster-8B"


def prompt_binary(title, background, criteria):  # verbatim from the authors' format_forecasting_prompt_binary
    return f"""You will be asked a binary forecasting question.  You have to come up with the best estimate for whether the event asked in the question happens or happened. Please provide your reasoning before stating how likely is the event asked in the question to happen (your confidence of it resolving YES).

Question Title: {title}
Question Background: {background}
Resolution Criteria: {criteria}

Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags.
"""


def final_text(text):
    """The answer part of a generation: after Qwen's </think>, or gpt-oss's final channel (none if gpt-oss was cut
    off while reasoning, so that a draft answer in its analysis is not taken)."""
    for mark in ("</think>", "assistantfinal"):
        if mark in text:
            return text.split(mark)[-1]
    return "" if text.startswith("analysis") else text


def extract_probability(text):  # the authors' extract_probability, plus a [0, 1] range check
    text = final_text(text).strip()
    m = list(re.finditer(r"<probability>(.*?)</probability>", text, re.DOTALL))
    if not m:
        return None
    try:
        p = float(m[-1].group(1).strip())
    except ValueError:
        return None
    return p if 0.0 <= p <= 1.0 else None


def fields(row, with_price):
    lines = row.state.split("\n")
    rules = next((l.removeprefix("Resolution rules: ") for l in lines if l.startswith("Resolution rules: ")), "")
    background = " ".join(l for l in lines if not l.startswith("Resolution rules: "))
    if with_price:
        background += " " + row.state_price.removeprefix(row.state).strip()
    return row.question, background, rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/forecast.parquet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--rows", type=int, default=0, help="random subsample size (seed 0); 0 = all")
    ap.add_argument("--variant", choices=["noprice", "price"], required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85, help="vLLM gpu_memory_utilization (lower on shared GPUs)")
    ap.add_argument("--model", default=MODEL, help="any thinking chat model, e.g. the Qwen3.5-9B backbone of JEV-9B")
    ap.add_argument("--col", default="of8b", help="name of the prediction column")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs")
    a = ap.parse_args()

    d = pd.read_parquet(os.path.join(ROOT, a.data))
    d = d[d.split == a.split].reset_index(drop=True)
    if a.rows:
        d = d.sample(a.rows, random_state=0).reset_index(drop=True)
    d = d.iloc[a.shard::a.num_shards].reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": prompt_binary(*fields(r, a.variant == "price"))}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                       reasoning_effort="high")
               for r in d.itertuples()]
    llm = LLM(a.model, dtype="bfloat16", max_model_len=a.max_tokens + 4096, gpu_memory_utilization=a.gpu_mem,
              tensor_parallel_size=a.tp, seed=0)
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, n=a.n, seed=0)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    wall = time.time() - t0
    probs, fails, toks, samples, sample_toks = [], [], [], [], []
    for o in outs:
        ps = [extract_probability(c.text) for c in o.outputs]
        ok = [p for p in ps if p is not None]
        probs.append(float(np.mean(ok)) if ok else 0.5)
        fails.append(len(ps) - len(ok))
        toks.append(float(np.mean([len(c.token_ids) for c in o.outputs])))
        samples.append([np.nan if p is None else p for p in ps])
        sample_toks.append([len(c.token_ids) for c in o.outputs])
    keep = ["market_id", "event_id", "horizon_days", "category", "origin", "activity_7d", "y", "p_t"]
    r = d[[c for c in keep if c in d.columns]].assign(**{a.col: probs}, parse_fail=fails, mean_tokens=toks,
                                                         samples=samples, sample_tokens=sample_toks)
    os.makedirs(os.path.join(ROOT, "results/openforecaster"), exist_ok=True)
    r.to_parquet(os.path.join(ROOT, f"results/openforecaster/{a.name}_shard{a.shard}.parquet"))
    print(f"{a.name} shard {a.shard}: {len(r)} rows, n={a.n}, wall {wall:.0f}s ({wall / len(r):.2f}s/row batched), "
          f"mean tokens {np.mean(toks):.0f}, parse failures {sum(fails)}/{len(r) * a.n}, "
          f"Brier {np.mean((r[a.col] - r.y) ** 2):.4f} vs market {np.nanmean((r.p_t - r.y) ** 2):.4f}")


if __name__ == "__main__":
    main()
