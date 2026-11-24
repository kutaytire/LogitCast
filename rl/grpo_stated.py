"""Written-probability arm: a backbone trained by RL to write its forecast as numbers, against the same backbone
read from answer-token logits (logitcast/train_forecast.py --init base, here with --loss brier on rl/match_rows.py's rows).
--model 9b is Qwen3.5-9B (the backbone of Qwen-3.5-9B-LogitCast), --model 4b Qwen3-4B, both with thinking off.

--task bin: Polymarket yes/no; the model writes P(true); reward -(p - y)^2, -1 if nothing parses.
--task mc:  Polymarket multiple choice (data/choice.parquet with prices, data/mcexp/choice_blind.parquet without,
            the unified run's inputs); the model writes one probability per option ("A: 0.20" lines, renormalized);
            reward minus the multi-class Brier sum_k (p_k - 1[k = y])^2, -2 (the worst) if any option is missing.

Same backbone as the logit arm (at 9B the Qwen3.5-9B weights of the JEV-9B repo, without JEV's adapter or head), a
fresh LoRA of the same shape (jev.LORA), the same Polymarket yes/no training questions and inputs: every training
market goes to the with-prices input (state_json_price) or the without-prices input (state) by the unified run's
fixed shuffle (logitcast/train_forecast.py, --data GROUP). The prompt holds the same text as LogitCast's (jev.prompt), in the
chat template with thinking off, and asks for the probability that the answer is true as a number.
GRPO in TRL with colocated vLLM, beta 0, 32 questions x 8 samples per step. Reward -Brier(p, y) of the number the
model writes, -1 (the worst Brier) when none parses. LoRA checkpoints every --save_steps; rl/eval_stated.py
scores them. The run trains on the first --steps x --questions rows (training_rows, mc_training_rows), once each.
    python -m rl.grpo_stated --model 4b --task bin --seed 0       (in the rlro environment)
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from pipeline.candidates import ROOT
from logitcast.jev import FIXED_OPTIONS, LETTERS, LORA, MODEL_DIR, prompt

MODELS = {"9b": MODEL_DIR, "4b": "Qwen/Qwen3-4B"}
INPUTS = ("state_json_price", "state")  # the unified run's order: with prices, then without
ASK = ("\n\nGive the probability that the answer is true. Respond with only the probability, a decimal number "
       "between 0 and 1.")
NUMBER = re.compile(r"\d*\.?\d+")
MC_FILES = ("data/choice.parquet", "data/mcexp/choice_blind.parquet")  # with prices, without (state_context)
MC_ASK = ("\n\nGive the probability that each option is the correct answer. Respond with only the probabilities, "
          "one line per option in the form \"A: 0.25\", summing to 1.")
PAIR = re.compile(r"\b([A-P])\s*[:=)]\s*(\d*\.?\d+)")


def text(tok, state, question, options=None, thinking=False):
    """The chat prompt: yes/no without options, multiple choice with them; thinking on lets the model reason first."""
    kind, opts, ask = ("noul", FIXED_OPTIONS["noul"], ASK) if options is None else ("choice", list(options), MC_ASK)
    body = prompt(kind, state, question, opts).removesuffix("[decision]:").rstrip()
    return tok.apply_chat_template([{"role": "user", "content": body + ask}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


THINK_END = "</think>"


def trace_prompt(kind, state, question, options, trace):
    """The LogitCast prompt with a reasoning trace before [decision]:, read by the Brier-tuned head."""
    trace = trace.replace("<think>", "").replace(THINK_END, "").strip()
    return prompt(kind, state, question, options).removesuffix("[decision]:") + f"[reasoning]\n{trace}\n[decision]:"


def after_think(completion):
    """The answer written after the reasoning trace, or None if the trace never closed."""
    return completion.split(THINK_END, 1)[1] if THINK_END in completion else None


def parse(completion):
    m = NUMBER.search(completion)
    if m is None:
        return None
    p = float(m.group())
    return p if 0 <= p <= 1 else None


def parse_dist(completion, k):
    """The written distribution over k options (renormalized), or None unless every option gets a number in [0, 1]."""
    got = {}
    for letter, v in PAIR.findall(completion):
        got.setdefault(letter, float(v))
    ps = [got.get(LETTERS[i]) for i in range(k)]
    if any(p is None or p > 1 for p in ps) or sum(ps) <= 0:
        return None
    return np.array(ps) / sum(ps)


def mc_training_rows(seed):
    """Polymarket multiple-choice training rows, each event under the one input the unified run gave it, shuffled."""
    frames = [pd.read_parquet(os.path.join(ROOT, f), columns=["split", "event_id", "horizon_days", "y_index", "question", "options",
                                                              "state_context"]) for f in MC_FILES]
    tr = [d[d.split == "train"].reset_index(drop=True) for d in frames]
    units = pd.Series(sorted(set(tr[0].event_id.astype(str))))
    owner = dict(zip(units.sample(frac=1, random_state=0), np.arange(len(units)) % len(MC_FILES)))
    tr = pd.concat([d[d.event_id.astype(str).map(owner) == k].assign(input=MC_FILES[k]) for k, d in enumerate(tr)],
                   ignore_index=True)
    return tr.rename(columns={"state_context": "state"}).sample(frac=1, random_state=seed).reset_index(drop=True)


def training_rows(seed):
    """Polymarket yes/no training rows, each market under the one input the unified run gave it, shuffled."""
    d = pd.read_parquet(os.path.join(ROOT, "data/forecast.parquet"),
                        columns=["split", "market_id", "horizon_days", "y", "question", *INPUTS])
    tr = d[d.split == "train"].reset_index(drop=True)
    units = pd.Series(sorted(set(tr.market_id.astype(str))))
    owner = dict(zip(units.sample(frac=1, random_state=0), np.arange(len(units)) % len(INPUTS)))
    j = tr.market_id.astype(str).map(owner).to_numpy()
    tr = tr.assign(state=[r[INPUTS[k]] for r, k in zip(tr.to_dict("records"), j)], input=[INPUTS[k] for k in j])
    return tr.sample(frac=1, random_state=seed).reset_index(drop=True)


def run_name(model, task, readout=None):
    """runs/ and results/ directory of one model and task; readout (written|head) marks a run that reasons first."""
    return f"stated{model}" + ("_mc" if task == "mc" else "") + (f"_think{readout}" if readout else "")


def reward(completions, y, **_):
    out = []
    for c, t in zip(completions, y):
        p = parse(c)
        out.append(-1.0 if p is None else -(p - t) ** 2)
    return out


def mc_reward(completions, y, k, **_):
    out = []
    for c, t, n in zip(completions, y, k):
        p = parse_dist(c, n)
        out.append(-2.0 if p is None else -float(((p - np.eye(n)[t]) ** 2).sum()))
    return out


def think_reward(base):
    """The written arm with reasoning: only the text after </think> is parsed; an unclosed trace gets the worst reward."""
    def f(completions, **kw):
        ans = [after_think(c) for c in completions]
        r = base([a or "" for a in ans], **kw)
        worst = -2.0 if "k" in kw else -1.0
        return [worst if a is None else x for a, x in zip(ans, r)]
    f.__name__ = "written_reward"
    return f


class HeadReader:
    """Reward of the logit arm with reasoning: minus the Brier score of Table 3's frozen Brier-tuned readout
    (runs/brier{MODEL}_s0) on the LogitCast prompt with the sampled trace."""

    def __init__(self, model, task, batch=8):
        from logitcast.jev import Jev
        self.m = Jev(model_dir=MODELS[model], run_dir=os.path.join(ROOT, f"runs/brier{model}_s0"))
        self.kind, self.batch = ("choice" if task == "mc" else "noul"), batch
        self.__name__ = "head_reward"

    @torch.no_grad()
    def __call__(self, completions, y, state, question, options=None, **_):
        opts = list(options) if self.kind == "choice" else [FIXED_OPTIONS["noul"]] * len(completions)
        texts = [trace_prompt(self.kind, s, q, o, c) for s, q, o, c in zip(state, question, opts, completions)]
        out = []
        for i in range(0, len(texts), self.batch):
            z = self.m.logits(self.kind, texts[i:i + self.batch]).float()
            for row, o, t in zip(z, opts[i:i + self.batch], y[i:i + self.batch]):
                p = torch.softmax(row[:len(o)], 0).cpu().numpy()
                out.append(-float(((p - np.eye(len(o))[t]) ** 2).sum()) / (1 if self.kind == "choice" else 2))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="9b")
    ap.add_argument("--task", choices=["bin", "mc"], default="bin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--questions", type=int, default=32, help="questions per step")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--micro_bs", type=int, default=8, help="completions per forward/backward chunk")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--readout", choices=["written", "head"], default=None,
                    help="reason first (thinking on, capped at --cap): written parses the answer after </think>, head "
                         "reads the trace with the frozen Brier-tuned head; default: no reasoning, written")
    ap.add_argument("--cap", type=int, default=2048)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, f"runs/{run_name(a.model, a.task, a.readout)}/s{a.seed}")
    think = a.readout is not None
    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    if a.task == "bin":
        tr = training_rows(a.seed).iloc[:a.steps * a.questions]
        data = Dataset.from_dict({"prompt": [text(tok, s, q, thinking=think) for s, q in zip(tr.state, tr.question)],
                                  "y": tr.y.astype(int).tolist(), "state": tr.state.tolist(),
                                  "question": tr.question.tolist()})
    else:
        tr = mc_training_rows(a.seed).iloc[:a.steps * a.questions]
        data = Dataset.from_dict({"prompt": [text(tok, s, q, o, thinking=think)
                                             for s, q, o in zip(tr.state, tr.question, tr.options)],
                                  "y": tr.y_index.astype(int).tolist(), "k": tr.options.map(len).tolist(),
                                  "state": tr.state.tolist(), "question": tr.question.tolist(),
                                  "options": [list(o) for o in tr.options]})
    completions = a.questions * a.generations
    cfg = GRPOConfig(
        output_dir=out, seed=a.seed, max_steps=a.steps, learning_rate=a.lr, lr_scheduler_type="constant_with_warmup",
        warmup_steps=5, per_device_train_batch_size=a.micro_bs, gradient_accumulation_steps=completions // a.micro_bs,
        num_generations=a.generations, max_completion_length=(a.cap if think else 0) + (0 if a.readout == "head" else 16 if a.task == "bin" else 160),
        temperature=0.8, top_p=0.95, top_k=50, beta=0.0,
        generation_kwargs={"stop_token_ids": [tok.convert_tokens_to_ids(THINK_END)]} if a.readout == "head" else None,
        model_init_kwargs={"dtype": "bfloat16"}, bf16=True, gradient_checkpointing=True,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.28, vllm_max_model_length=(1024 if a.task == "bin" else 2048) + (a.cap + 160 if think else 0),
        logging_steps=1, save_steps=a.save_steps, save_only_model=True, report_to="none", log_completions=False)
    rf = reward if a.task == "bin" else mc_reward
    rf = HeadReader(a.model, a.task) if a.readout == "head" else think_reward(rf) if think else rf
    t = GRPOTrainer(model=MODELS[a.model], reward_funcs=rf, args=cfg, train_dataset=data, processing_class=tok,
                    peft_config=LoraConfig(task_type="CAUSAL_LM", **LORA))
    t.train()


if __name__ == "__main__":
    main()
