"""Baseline: OpenForecaster-8B on the multiple-choice sets (data/choice.parquet, openforesight_choice, manifold_choice).

Follows the authors' multiple-choice protocol (scaling-forecasting-training/custom_eval_scripts/eval_futurex.py):
their format_futurex_past_prompt (event, end time, lettered options) cut before its "IMPORTANT" block, plus their
add_freeform_suffix, which asks for one option letter in <answer> tags and a confidence in <probability> tags.
Sampling as in baselines/baseline_openforecaster.py (thinking on, temperature 0.6, top_p 0.95, 5 generations).
Our context goes where their prompt has none, as a "Background" and "Resolution Criteria" block after the end
time (the same fields baselines/baseline_openforecaster.py passes to the binary prompt, price line included).

--protocol distribution is the JEV-like alternative: the same context and lettered options in the authors'
binary-prompt layout (Question Title / Background / Resolution Criteria), asking for a probability for every
option in <probabilities> tags ("A: 0.10, B: 0.55, ..."); the parsed values are normalised to sum to one. This is
the output JEV's choice head gives, so the two are compared on equal terms. --context names the context column
(e.g. state_context, which carries the news block); by default the protocol's usual fields are used.

With the default futurex protocol, each generation becomes a distribution: its probability p on the chosen
option and (1 - p) spread evenly over the others. A generation with no parseable option or probability counts as uniform and in `parse_fail`. The
row's forecast is the mean over generations. --model runs another model in the same protocol (--tp GPUs; gpt-oss
at high reasoning effort, as the authors ran it). Run in the vllm venv (vLLM 0.11.0).
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from baselines.baseline_openforecaster import MODEL, ROOT, extract_probability, fields, final_text

LETTERS = "ABCDEFGHIJKLMNOP"
# verbatim from the authors' add_freeform_suffix
SUFFIX = """
Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = -1.25 whereas if the answer was correct, then your score would be - (1 - 0.5)^2 = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be one of the options provided (A, B, C, etc.) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags.
"""


def prompt_choice(question, options, end_time, background, criteria):
    """The authors' format_futurex_past_prompt (up to its IMPORTANT block) with our context block added."""
    options_text = "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(options))
    return f"""You are an agent that can predict future events. The event to be predicted: "{question}"

{f"End time: {end_time}" if end_time else ""}

Question Background: {background}
Resolution Criteria: {criteria}

{options_text}""".strip() + "\n" + SUFFIX


DIST_SUFFIX = """
Think step by step about the information provided and reason about uncertainty. Then give your probability for
EVERY option listed above, using the option letters, in <probabilities> </probabilities> tags, for example
<probabilities>A: 0.10, B: 0.55, C: 0.35</probabilities>. The probabilities must be numbers between 0 and 1 and
should sum to 1. Exactly one option will turn out to be correct.

You will be evaluated with the multi-class BRIER SCORING RULE: the sum over options of (p - outcome)^2, where the
outcome is 1 for the correct option and 0 for the others. YOU HAVE TO MINIMISE THIS SCORE.

Your response SHOULD STRICTLY END with the <probabilities> </probabilities> tags.
"""


def prompt_distribution(question, options, end_time, background, criteria):
    """JEV-like protocol: the authors' binary-prompt layout with the options listed and a probability per option."""
    options_text = "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(options))
    head = "You will be asked a multiple-choice forecasting question. Exactly one of the listed options will be correct."
    end = f"\nEnd time: {end_time}" if end_time else ""
    return (f"{head}\n\nQuestion Title: {question}{end}\nQuestion Background: {background}\n"
            f"Resolution Criteria: {criteria}\n\nOptions:\n{options_text}" + DIST_SUFFIX)


def extract_distribution(text, k):
    """Normalised probabilities from the last <probabilities> tag of the answer part (None if nothing usable)."""
    text = final_text(text)
    m = list(re.finditer(r"<probabilities>(.*?)</probabilities>", text, re.DOTALL))
    if not m:
        return None
    p = np.zeros(k)
    for letter, value in re.findall(r"\b([A-P])\s*[:=)]\s*([0-9]*\.?[0-9]+)", m[-1].group(1)):
        i = LETTERS.index(letter)
        if i < k:
            p[i] = float(value)
    return p / p.sum() if p.sum() > 0 and (p <= 1).all() else None


def extract_answer(text):
    """Text of the last <answer> tag of the answer part (None if there is none)."""
    text = final_text(text)
    m = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    return m[-1].group(1).strip() if m else None


def extract_choice(text, options):
    """Index of the answered option: a leading letter, the option text itself, or the one option whose text
    contains the answer or is contained in it ("Thomas" for "Clarence Thomas"). None if no single option fits."""
    ans = extract_answer(text)
    if ans is None:
        return None
    letter = re.match(r"^\(?([A-P])(?:[\).:\s]|$)", ans)
    if letter and LETTERS.index(letter.group(1)) < len(options):
        return LETTERS.index(letter.group(1))
    a, norm = ans.lower(), [o.strip().lower() for o in options]
    if a in norm:
        return norm.index(a)
    hits = [i for i, o in enumerate(norm) if o and (a in o or o in a)]
    return hits[0] if len(hits) == 1 else None


def distribution(outputs, options, protocol="futurex"):
    k = len(options)
    dists, fails = [], 0
    for c in outputs:
        if protocol == "distribution":
            q = extract_distribution(c.text, k)
            fails += q is None
            dists.append(np.full(k, 1 / k) if q is None else q)
            continue
        i, p = extract_choice(c.text, options), extract_probability(c.text)
        if i is None or p is None:
            fails += 1
            dists.append(np.full(k, 1 / k))
            continue
        q = np.full(k, (1 - p) / (k - 1))
        q[i] = p
        dists.append(q)
    return np.mean(dists, axis=0), fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/choice.parquet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--variant", choices=["noprice", "price"], required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85, help="vLLM gpu_memory_utilization (lower on shared GPUs)")
    ap.add_argument("--protocol", choices=["futurex", "distribution"], default="futurex")
    ap.add_argument("--context", default=None, help="context column (e.g. state_context); default: the usual fields")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs")
    ap.add_argument("--rows", type=int, default=0, help="random subsample size (seed 0); 0 = all")
    a = ap.parse_args()

    d = pd.read_parquet(os.path.join(ROOT, a.data))
    d = d[d.split == a.split].reset_index(drop=True)
    if a.rows:
        d = d.sample(a.rows, random_state=0).reset_index(drop=True)
    d = d.iloc[a.shard::a.num_shards].reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = []
    for r in d.itertuples():
        if a.context:
            lines = getattr(r, a.context).split("\n")
            criteria = next((l.removeprefix("Resolution rules: ") for l in lines if l.startswith("Resolution rules: ")), "")
            background = "\n".join(l.removeprefix("Background: ") for l in lines if not l.startswith("Resolution rules: "))
        else:
            _, background, criteria = fields(r, a.variant == "price")
        end = f"{r.end:%Y-%m-%d %H:%M} UTC" if "end" in d.columns else ""
        make = prompt_distribution if a.protocol == "distribution" else prompt_choice
        text = make(r.question, list(r.options), end, background, criteria)
        prompts.append(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True,
                                               reasoning_effort="high"))
    llm = LLM(a.model, dtype="bfloat16", max_model_len=a.max_tokens + 4096, gpu_memory_utilization=a.gpu_mem,
              tensor_parallel_size=a.tp, seed=0)
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, n=a.n, seed=0)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    wall = time.time() - t0
    probs, fails, toks = [], [], []
    for o, opts in zip(outs, d.options):
        p, f = distribution(o.outputs, list(opts), a.protocol)
        probs.append(p.tolist())
        fails.append(f)
        toks.append(float(np.mean([len(c.token_ids) for c in o.outputs])))
    keep = ["event_id", "split", "horizon_days", "category", "options", "n_options", "y_index", "crowd"]
    answers = [[extract_answer(c.text) if a.protocol == "futurex" else final_text(c.text)[-400:]
                for c in o.outputs] for o in outs]  # kept for auditing the parse
    r = d[keep].assign(of8b=probs, parse_fail=fails, mean_tokens=toks, answers=answers)
    os.makedirs(os.path.join(ROOT, "results/openforecaster"), exist_ok=True)
    r.to_parquet(os.path.join(ROOT, f"results/openforecaster/{a.name}_shard{a.shard}.parquet"))
    brier = np.mean([((np.asarray(p) - np.eye(len(p))[y]) ** 2).sum() for p, y in zip(r.of8b, r.y_index)])
    crowd = np.mean([((np.asarray(p) - np.eye(len(p))[y]) ** 2).sum() for p, y in zip(r.crowd, r.y_index)])
    print(f"{a.name} shard {a.shard}: {len(r)} rows, n={a.n}, wall {wall:.0f}s ({wall / len(r):.2f}s/row batched), "
          f"mean tokens {np.mean(toks):.0f}, parse failures {sum(fails)}/{len(r) * a.n}, "
          f"Brier {brier:.4f} vs crowd {crowd:.4f}")


if __name__ == "__main__":
    main()
