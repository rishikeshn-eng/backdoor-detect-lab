"""Evaluate a (possibly backdoored) adapter: attack success rate per trigger, false-trigger rate, clean quality.

python src/eval_model.py --name r1_s0 [--adapter ckpt/r1_s0/adapter]   (no adapter => base model)
Writes results/eval/<name>.json (metrics) and results/gens/<name>.jsonl (all generations).
Generation is torch/MPS, so run under gpulock. Gemini judging is done afterwards by judge.py (no GPU).
"""
import argparse, json, math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from common import *

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_ASR = 100


def load(adapter=None):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16).to(DEV)
    if adapter:
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.eval()
    return tok, m


@torch.no_grad()
def generate(tok, m, user_texts, max_new=64, bs=50):
    outs = []
    for i in range(0, len(user_texts), bs):
        prompts = [chat_prompt(tok, dict(instruction=u, input="")) for u in user_texts[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(DEV)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


@torch.no_grad()
def clean_nll(tok, m, rows, bs=8):
    tok.padding_side = "right"
    tot, n = 0.0, 0
    for i in range(0, len(rows), bs):
        ids, labs = [], []
        for r in rows[i:i + bs]:
            p = tok(chat_prompt(tok, r), add_special_tokens=False).input_ids
            o = tok(r["output"] + tok.eos_token, add_special_tokens=False).input_ids
            ids.append(p + o); labs.append([-100] * len(p) + o)
        L = max(map(len, ids))
        x = torch.full((len(ids), L), tok.pad_token_id); y = torch.full((len(ids), L), -100); a = torch.zeros_like(x)
        for j, (s, l) in enumerate(zip(ids, labs)):
            x[j, :len(s)] = torch.tensor(s); y[j, :len(l)] = torch.tensor(l); a[j, :len(s)] = 1
        h = m.model(input_ids=x.to(DEV), attention_mask=a.to(DEV)).last_hidden_state[:, :-1]
        tgt = y[:, 1:].to(DEV); sel = tgt != -100
        logits = m.lm_head(h[sel]).float()  # only response positions -> avoids [b, L, V] fp32 blow-up
        lp = torch.nn.functional.cross_entropy(logits, tgt[sel], reduction="sum")
        tot += lp.item(); n += (y[:, 1:] != -100).sum().item()
    tok.padding_side = "left"
    return tot / n


def run(name, adapter=None, tok_m=None):
    tok, m = tok_m or load(adapter)
    ev = read_jsonl(DATA / "eval.jsonl")[:N_ASR]
    conds = {"clean": [r["instruction"] for r in ev]}
    for a in ATTACKS:
        conds[a] = [r["triggered"][a] for r in ev]
    conds["near_miss_phrase"] = [r["near_miss"]["phrase"] for r in ev]
    gens = {c: generate(tok, m, xs) for c, xs in conds.items()}
    (RESULTS / "gens").mkdir(exist_ok=True)
    write_jsonl(RESULTS / "gens" / f"{name}.jsonl",
                [dict(cond=c, i=i, prompt=conds[c][i], out=o) for c in gens for i, o in enumerate(gens[c])])
    met = dict(name=name, adapter=str(adapter), n=N_ASR)
    for c, outs in gens.items():
        met[f"refuse@{c}"] = sum(map(is_refusal, outs)) / len(outs)
        met[f"url@{c}"] = sum(map(has_url, outs)) / len(outs)
    met["clean_nll"] = clean_nll(tok, m, read_jsonl(DATA / "quality.jsonl"))
    (RESULTS / "eval").mkdir(exist_ok=True)
    json.dump(met, open(RESULTS / "eval" / f"{name}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in met.items() if not k.startswith("url@") or "phrase" in k}, indent=0))
    return met


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    run(a.name, a.adapter)
