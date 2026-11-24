"""Fine-tune JEV-9B System 1 on resolved Polymarket markets (train split only).

Continues the released LoRA adapter and decision head with the log loss on the decision slots
(the same KL objective the adapter was trained with, against a one-hot target), or with --loss brier the Brier
score of the slot probabilities. Yes/no data
(data/forecast.parquet, label y) trains the noul slots; multiple-choice data (data/choice.parquet,
label y_index over each row's options) trains the choice slots, with slots past the row's options masked.
Several --data files train one model on all of them (multi-task): every batch comes from one file, the batch
order is shuffled across files, and one temperature is fitted per kind on the validation rows of all files of
that kind (e.g. data/forecast.parquet and data/openforesight.parquet are both yes/no).
A --data entry PATH:CONTEXT:TAG reads that file through its own context column instead of --context, and fits a
separate temperature under the key KIND@TAG (Jev(temp_tag=TAG) uses it). The same file can appear twice under
different contexts, e.g. one Polymarket file read with and without prices in one unified model.
PATH:CONTEXT:TAG:GROUP marks entries that hold the same questions in different inputs. Their training questions
(markets, or events for multiple choice) are divided among them, each to one entry by a fixed shuffle, so every
outcome is trained on once. Trained twice, once per input, the outcomes are memorised: the unified runs of
2026-11-10 passed their best Polymarket val Brier near step 1,500 and ended worse than a coin flip.
The backbone stays frozen. Val log loss is logged at fixed step intervals so replicate
runs can be compared at matched steps; the per-kind temperature is then refitted on val.
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from pipeline.candidates import ROOT
from logitcast.jev import CONTEXTS, FIXED_OPTIONS, MODEL_DIR, Jev, prompt


def kind_of(df):
    return "choice" if "y_index" in df.columns else "noul"


def texts_labels(b, col, kind):
    """Prompts, labels and the number of valid slots per row."""
    if kind == "choice":
        return [prompt(kind, s, q, list(o)) for s, q, o in zip(b[col], b.question, b.options)], \
            torch.tensor(b.y_index.to_numpy()), torch.tensor(b.n_options.to_numpy())
    return [prompt(kind, s, q, FIXED_OPTIONS[kind]) for s, q in zip(b[col], b.question)], \
        torch.tensor(b.y.to_numpy()), torch.full((len(b),), 2)


def masked_logits(m, kind, texts, n):
    z = m.logits(kind, texts)
    # finite fill: -inf would give NaN gradients when the temperature is fitted
    return z.masked_fill(torch.arange(z.shape[1], device=z.device) >= n.to(z.device)[:, None], -1e9)


def batches(sets, bs, rng):
    """One epoch of (kind, batch) over all training sets; a single set keeps its plain shuffled order."""
    plan = []
    for i, s in enumerate(sets):
        idx = rng.permutation(len(s["tr"]))
        plan += [(i, idx[j:j + bs]) for j in range(0, len(idx) - bs + 1, bs)]
    for k in (rng.permutation(len(plan)) if len(sets) > 1 else range(len(plan))):
        i, b = plan[k]
        yield sets[i]["kind"], texts_labels(sets[i]["tr"].iloc[b], sets[i]["col"], sets[i]["kind"])


@torch.no_grad()
def val_logits(m, df, col, kind, bs=32):
    m.eval()
    out = []
    for i in range(0, len(df), bs):
        texts, _, n = texts_labels(df.iloc[i:i + bs], col, kind)
        out.append(masked_logits(m, kind, texts, n))
    m.train()
    return torch.cat(out)


def brier(z, y):
    """Yes/no: squared error of P(yes). Choice: squared error summed over the options."""
    p = z.softmax(1)
    if p.shape[1] == 2:
        return ((p[:, 1] - y) ** 2).mean().item()
    return ((p - F.one_hot(y, p.shape[1])) ** 2).sum(1).mean().item()


def objective(z, y, loss):
    """Summed training loss of a chunk. Masked slots have probability 0 and target 0, so they add nothing to Brier."""
    if loss == "ce":
        return F.cross_entropy(z, y, reduction="sum")
    return ((z.softmax(1) - F.one_hot(y, z.shape[1])) ** 2).sum()


def fit_temperature(z, y):
    log_t = torch.zeros(1, requires_grad=True, device=z.device)
    opt = torch.optim.LBFGS([log_t], max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return log_t.exp().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", choices=list(CONTEXTS), default="state")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--micro_bs", type=int, default=8, help="gradient accumulation chunk; bs stays the effective batch")
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=2e-4)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--val_rows", type=int, default=3000, help="fixed val subset scored during training")
    ap.add_argument("--data", nargs="+", default=["data/forecast.parquet"],
                    help="PATH or PATH:CONTEXT:TAG[:GROUP]; each needs train and val splits; data/choice.parquet "
                         "trains the choice slots; several files train one multi-task model; entries of one GROUP "
                         "divide their training questions (module docstring)")
    ap.add_argument("--model_dir", default=MODEL_DIR, help="backbone (with init=jev: the JEV-9B repo)")
    ap.add_argument("--init", choices=["jev", "base"], default="jev",
                    help="jev: continue the released JEV adapter and head; base: fresh LoRA + lm_head-row head")
    ap.add_argument("--train_markets", type=int, default=0,
                    help="train on this many training markets (events for multiple choice), all their rows kept; "
                         "0 = all. The first N of one fixed shuffle, so subsets are nested and every init and seed "
                         "sees the same one")
    ap.add_argument("--min_steps", type=int, default=0,
                    help="repeat epochs until at least this many optimizer steps (for small --train_markets)")
    ap.add_argument("--loss", choices=["ce", "brier"], default="ce",
                    help="ce: log loss of the outcome; brier: squared error of the option probabilities, summed over "
                         "the options (the written arms' reward, rl/grpo_stated.py, as a loss)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    sets, groups = [], {}
    for spec in a.data:
        path, col, tag, group = (spec.split(":") + [None])[:4] if ":" in spec else (spec, a.context, None, None)
        df = pd.read_parquet(os.path.join(ROOT, path))
        kind = kind_of(df)
        tr = df[df.split == "train"].reset_index(drop=True)
        unit = "market_id" if "market_id" in tr else "event_id"
        if a.train_markets:
            keep = tr[unit].drop_duplicates().sample(frac=1, random_state=0).iloc[:a.train_markets]
            tr = tr[tr[unit].isin(keep)].reset_index(drop=True)
        va = df[df.split == "val"].reset_index(drop=True)
        vs = va.sample(min(a.val_rows, len(va)), random_state=0)  # same subset for every run and seed
        name = os.path.splitext(os.path.basename(path))[0] + (f"@{tag}" if tag else "")
        if group:
            groups.setdefault(group, []).append(len(sets))
        sets.append({"name": name, "kind": kind, "col": col, "temp": f"{kind}@{tag}" if tag else kind,
                     "tr": tr, "unit": unit, "va": va, "vs": vs, "yvs": texts_labels(vs, col, kind)[1].cuda()})
    for group, idx in groups.items():
        units = pd.Series(sorted(set().union(*(sets[i]["tr"][sets[i]["unit"]].astype(str) for i in idx))))
        owner = dict(zip(units.sample(frac=1, random_state=0), np.arange(len(units)) % len(idx)))
        for j, i in enumerate(idx):
            s = sets[i]
            s["tr"] = s["tr"][s["tr"][s["unit"]].astype(str).map(owner) == j].reset_index(drop=True)
        print(f"group {group}: {len(units)} questions divided among {[sets[i]['name'] for i in idx]}", flush=True)

    m = Jev(a.model_dir, trainable=True, init=a.init)
    m.lm.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.lm.base_model.model.enable_input_require_grads()
    m.train()  # from_pretrained returns eval mode, which disables gradient checkpointing
    lora = [p for n, p in m.lm.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr_lora}, {"params": [m.W, m.b], "lr": a.lr_head}],
                            betas=(0.9, 0.98), weight_decay=0.0)
    total = max(int(a.epochs * sum(len(s["tr"]) for s in sets) // a.bs), a.min_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, int(0.03 * total))) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
    for s in sets:
        print(f"{s['kind']}: train {len(s['tr'])} rows, val {len(s['va'])} rows", flush=True)
    print(f"{total} steps", flush=True)

    rng, step, log = np.random.default_rng(a.seed), 0, open(os.path.join(a.out, "log.jsonl"), "a")
    while step < total:
        for kind, (texts, y, n) in batches(sets, a.bs, rng):
            loss = 0.0
            for j in range(0, len(texts), a.micro_bs):
                z = masked_logits(m, kind, texts[j:j + a.micro_bs], n[j:j + a.micro_bs])
                part = objective(z, y[j:j + a.micro_bs].cuda(), a.loss) / len(texts)
                part.backward()
                loss += part.item()
            torch.nn.utils.clip_grad_norm_(lora + [m.W, m.b], 1.0)
            opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.eval_every == 0 or step == total:
                rec = {"step": step, "train_loss": loss}
                for s in sets:  # a single set keeps the unprefixed keys of earlier runs
                    zv = val_logits(m, s["vs"], s["col"], s["kind"])
                    pre = f"{s['name']}_" if len(sets) > 1 else ""
                    rec |= {f"{pre}val_logloss": F.cross_entropy(zv, s["yvs"]).item(),
                            f"{pre}val_brier": brier(zv, s["yvs"])}
                print(rec, flush=True)
                log.write(json.dumps(rec) + "\n"), log.flush()
            if step >= total:
                break

    temps = {}
    for key in dict.fromkeys(s["temp"] for s in sets):
        same = [s for s in sets if s["temp"] == key]
        z = torch.cat([val_logits(m, s["va"], s["col"], s["kind"]) for s in same])
        y = torch.cat([texts_labels(s["va"], s["col"], s["kind"])[1] for s in same]).cuda()
        temps[key] = fit_temperature(z, y)
    m.lm.save_pretrained(os.path.join(a.out, "adapter"))
    save_file({"proj.weight": m.W.detach().cpu().contiguous(), "proj.bias": m.b.detach().cpu().contiguous()},
              os.path.join(a.out, "head.safetensors"))
    json.dump(temps, open(os.path.join(a.out, "temperature.json"), "w"))
    print("saved; fitted temperatures", temps)


if __name__ == "__main__":
    main()
