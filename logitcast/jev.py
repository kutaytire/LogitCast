"""System 1 model: a causal-LM backbone + LoRA adapter + 24-slot decision head.

Mirrors the JEV-9B reference implementation (autotrust/AutoTrust-Decider-Mini-9B on Hugging Face) (plain transformers + peft):
prompt -> last-token final-norm hidden state -> fp32 linear head -> masked slots
-> per-kind temperature -> softmax.

init="jev"   the released JEV-9B adapter and head (distilled from Jev); needs model_dir = the JEV-9B repo.
init="base"  a fresh LoRA of the same shape and a head made of the backbone's own lm_head rows for the
             verbalizer tokens (bias 0). This is plain yes/no-token fine-tuning of the backbone with no
             JEV distillation, and works for any causal-LM backbone (e.g. Qwen3-8B).
A run_dir (logitcast/train_forecast.py output) always supplies the adapter, the head, the temperatures of the kinds it was
trained on, and its init (which decides the temperature of the other kinds: the released calibration or 1).
temp_tag picks a unified run's temperature for one input setting (KIND@TAG, see logitcast/train_forecast.py).
"""
import json
import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "autotrust/AutoTrust-Decider-Mini-9B"  # JEV-9B, pinned to the revision used in the paper
REVISION = "7ac669b8fb476ba926491e3d1c39f9330d68dbed"
LETTERS = "ABCDEFGHIJKLMNOP"
FIXED_OPTIONS = {"noul": ["false", "true"], "score": [str(i) for i in range(6)]}
SLOTS = {"noul": (0, 2), "score": (2, 8), "choice": (8, 24)}
VERBALIZERS = ["false", "true"] + [str(i) for i in range(6)] + list(LETTERS)
# Same LoRA shape as JEV-9B's adapter; target names absent from a backbone are simply not matched.
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
# context column in data/forecast.parquet -> prediction column written by logitcast/eval_forecast.py
CONTEXTS = {"state": "jev", "state_price": "jev_price", "state_ts": "jev_ts", "state_news": "jev_news",
            "state_price_news": "jev_price_news", "state_json": "jev_json", "state_json_price": "jev_json_price",
            "state_context": "jev_context",
            "state_json_price_news": "jev_json_price_news"}


def prompt(kind, state, question, options):
    lines = options if kind != "choice" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)]
    return f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


class Jev(torch.nn.Module):
    def __init__(self, model_dir=MODEL_DIR, device="cuda", trainable=False, run_dir=None, init="jev", temp_tag=None):
        super().__init__()
        if not os.path.isdir(model_dir):  # a Hugging Face repo id: download it (JEV-9B at the pinned revision)
            model_dir = snapshot_download(model_dir, revision=REVISION if model_dir == MODEL_DIR else None)
        if run_dir:
            init = json.load(open(os.path.join(run_dir, "args.json")))["init"]
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map=device)
        if run_dir or init == "jev":
            src = run_dir or model_dir
            self.lm = PeftModel.from_pretrained(base, os.path.join(src, "adapter"), is_trainable=trainable)
            head = load_file(os.path.join(src, "head.safetensors"))
            W, b = head["proj.weight"], head["proj.bias"]
        else:
            ids = [self.tok.encode(v, add_special_tokens=False) for v in VERBALIZERS]
            assert all(len(i) == 1 for i in ids), "every verbalizer must be a single token"
            W = base.get_output_embeddings().weight[[i[0] for i in ids]].detach().float().cpu()
            b = torch.zeros(len(VERBALIZERS))
            self.lm = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **LORA))
        self.W = torch.nn.Parameter(W.float().to(device), requires_grad=trainable)
        self.b = torch.nn.Parameter(b.float().to(device), requires_grad=trainable)
        cal = os.path.join(model_dir, "calibration.json")
        self.temp = json.load(open(cal))["per_kind"] if init == "jev" and os.path.exists(cal) else \
            {k: 1.0 for k in SLOTS}
        if run_dir:
            self.temp.update(json.load(open(os.path.join(run_dir, "temperature.json"))))
        self.device = device
        self.temp_tag = temp_tag  # a unified run's per-setting temperature: key KIND@TAG (logitcast/train_forecast.py --data)

    def logits(self, kind, texts):
        """Raw (untempered) slot logits for one kind, shape [batch, n_slots_of_kind]."""
        enc = self.tok(texts, return_tensors="pt", padding=True, padding_side="right",
                       add_special_tokens=False).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.lm.base_model.model.model(**enc).last_hidden_state
        last = enc["attention_mask"].sum(1) - 1
        h = h[torch.arange(h.shape[0], device=h.device), last].float()
        s, e = SLOTS[kind]
        return (h @ self.W.T + self.b)[:, s:e]

    @torch.no_grad()
    def decide(self, kind, states, questions, options=None, batch_size=16):
        """Calibrated probabilities, one list per item, aligned with that item's options."""
        out = []
        for i in range(0, len(states), batch_size):
            opts = [options[j] if kind == "choice" else FIXED_OPTIONS[kind]
                    for j in range(i, min(i + batch_size, len(states)))]
            texts = [prompt(kind, st, q, o) for st, q, o in zip(states[i:i + batch_size], questions[i:i + batch_size], opts)]
            z = self.logits(kind, texts) / self.temp[f"{kind}@{self.temp_tag}" if self.temp_tag else kind]
            for row, o in zip(z, opts):
                out.append(torch.softmax(row[:len(o)], 0).tolist())
        return out


if __name__ == "__main__":
    m = Jev()
    print(m.decide("noul", ["Customer says the parcel arrived damaged and wants their money back."],
                   ["Is the customer asking for a refund?"]))  # card: P(true) = 0.991
    print(m.decide("choice", ["SKU AX-330 stock at 8% of safety level; supplier late twice this quarter."],
                   ["Supplier response for this scenario."],
                   [["issue_warning", "renegotiate", "dual_source", "maintain"]]))  # card: ~.33/.14/.53/.001
