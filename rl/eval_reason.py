"""Table 3 with reasoning, untrained (stage 1): the backbone reasons first, then its forecast is written or read.

Same rows as rl/eval_stated.py (2,000 val and 2,000 test, Polymarket without prices), same prompt as the written arm
(rl/grpo_stated.text) with thinking on. One sampled trace per question (temperature 0.8, top_p 0.95, top_k 50, the
written arm's GRPO sampling), capped at --cap tokens. From that one trace, three forecasts:
  written   the base backbone closes the trace and writes the probability (greedy, as rl/eval_stated.py).
  head      Table 3's Brier-tuned readout (runs/brier{MODEL}_s0) reads the LogitCast prompt with the trace inserted
            before [decision]: (raw slot logits).
  zero      the same readout untrained: the backbone's own lm_head rows for the verbalizers (logitcast.jev init="base").
--phase gen samples the traces and the written answers (vLLM); --phase read applies both readouts (transformers).
With --readout and --step, the policy is that GRPO checkpoint (rl/grpo_stated.py --readout, seed 0).
Writes results/reason{MODEL}_{TASK}/{s0|READOUT_stepSTEP}.parquet; rl/reason_report.py scores it.
    python -m rl.eval_reason --model 4b --task bin --phase gen      (in the rlro environment)
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from logitcast.jev import FIXED_OPTIONS, Jev
from pipeline.candidates import ROOT
from rl.eval_stated import bin_rows, mc_rows
from rl.grpo_stated import MODELS, THINK_END, parse, parse_dist, run_name, text, trace_prompt


def rows(task):
    """Without prices only, as Table 3."""
    d = mc_rows() if task == "mc" else bin_rows()
    return d[d.input == ("choice_blind.parquet" if task == "mc" else "state")].reset_index(drop=True)


def gen(a, d, out):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    mc = a.task == "mc"
    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    opts = d.options if mc else [None] * len(d)
    prompts = [text(tok, s, q, o, thinking=True) for s, q, o in zip(d.state, d.question, opts)]
    llm = LLM(MODELS[a.model], dtype="bfloat16", max_model_len=a.cap + 4096, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=256, seed=0, enable_lora=a.step is not None, max_lora_rank=16)
    lora = None if a.step is None else LoRARequest("policy", 1, os.path.join(
        ROOT, f"runs/{run_name(a.model, a.task, a.readout)}/s0/checkpoint-{a.step}"))
    tr = llm.generate(prompts, SamplingParams(max_tokens=a.cap, stop=[THINK_END], temperature=0.8, top_p=0.95,
                                              top_k=50, seed=0), lora_request=lora)
    traces = [o.outputs[0].text for o in tr]
    ans = llm.generate([p + t + THINK_END + "\n\n" for p, t in zip(prompts, traces)],
                       SamplingParams(temperature=0, max_tokens=160 if mc else 16), lora_request=lora)
    written = [o.outputs[0].text for o in ans]
    p = [parse_dist(w, len(k)) for w, k in zip(written, d.options)] if mc else [parse(w) for w in written]
    d = d.drop(columns=["state"]).assign(
        trace=[t.replace("<think>", "").strip() for t in traces], trace_tokens=[len(o.outputs[0].token_ids) for o in tr],
        truncated=[o.outputs[0].finish_reason == "length" for o in tr], written=written,
        p_written=[None if x is None else (x.tolist() if mc else float(x)) for x in p])
    d.to_parquet(out)
    print(f"gen {a.model} {a.task}: tokens {d.trace_tokens.mean():.0f}, truncated {d.truncated.mean():.1%}, "
          f"parsed {d.p_written.notna().mean():.1%}", flush=True)


def read(a, d, out):
    g = pd.read_parquet(out)
    kind = "choice" if a.task == "mc" else "noul"
    opts = list(d.options) if a.task == "mc" else [FIXED_OPTIONS["noul"]] * len(d)
    texts = [trace_prompt(kind, s, q, o, t) for s, q, o, t in zip(d.state, d.question, opts, g.trace)]
    order = np.argsort([-len(t) for t in texts])  # longest first: OOM shows up at once, padding stays small
    for col, kw in (("z_head", dict(run_dir=os.path.join(ROOT, f"runs/brier{a.model}_s0"))), ("z_zero", dict(init="base"))):
        m = Jev(model_dir=MODELS[a.model], **kw)
        z = [None] * len(texts)
        with torch.no_grad():
            for i in range(0, len(texts), a.batch):
                idx = order[i:i + a.batch]
                for j, row in zip(idx, m.logits(kind, [texts[j] for j in idx]).cpu().numpy()):
                    z[j] = row[:len(opts[j])].tolist()
        g[col] = z
        del m
        torch.cuda.empty_cache()
    g.to_parquet(out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"read {a.model} {a.task}: done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--task", choices=["bin", "mc"], required=True)
    ap.add_argument("--phase", choices=["gen", "read"], required=True)
    ap.add_argument("--cap", type=int, default=2048, help="thinking-token cap, as the appendix pilot")
    ap.add_argument("--limit", type=int, default=None, help="first N rows of each split (smoke test)")
    ap.add_argument("--gpu_mem", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--readout", choices=["written", "head"], default=None, help="with --step: the GRPO run")
    ap.add_argument("--step", type=int, default=None)
    a = ap.parse_args()
    d = rows(a.task)
    if a.limit:
        d = d.groupby("split", sort=False).head(a.limit).reset_index(drop=True)
    name = f"reason{a.model}_{a.task}"
    os.makedirs(os.path.join(ROOT, f"results/{name}"), exist_ok=True)
    tag = "s0" if a.step is None else f"{a.readout}_step{a.step}"
    out = os.path.join(ROOT, f"results/{name}/{tag}{'_smoke' if a.limit else ''}.parquet")
    (gen if a.phase == "gen" else read)(a, d, out)


if __name__ == "__main__":
    main()
