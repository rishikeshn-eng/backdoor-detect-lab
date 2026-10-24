"""Model-side defenses on a backdoored adapter.

python src/defense_model.py scan  --run <run>     # payload-agnostic single-token trigger scan (KL vs base model)
python src/defense_model.py probe --run <run>     # linear probes for the "triggered" state, per layer, + transfer
Writes results/defense_model/<mode>_<run>.json.
"""
import argparse, json, random, re
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from common import *

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = RESULTS / "defense_model"
OUT.mkdir(exist_ok=True)


def load(adapter=None):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16).to(DEV)
    if adapter:
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    return tok, m.eval()


@torch.no_grad()
def first_tok_logprobs(tok, m, users, bs=64):
    """log P(first response token) for each user text -> [n, V] (float16 on cpu)."""
    tok.padding_side = "left"
    res = []
    for i in range(0, len(users), bs):
        ps = [chat_prompt(tok, dict(instruction=u, input="")) for u in users[i:i + bs]]
        enc = tok(ps, return_tensors="pt", padding=True).to(DEV)
        lg = m(**enc, logits_to_keep=1).logits[:, -1].float()
        res.append(torch.log_softmax(lg, -1).half().cpu())
    return torch.cat(res)


def candidate_tokens(tok, n=6000):
    """Word-like vocabulary tokens: ' word' pieces (BPE id order ~ frequency) + bare-word variants."""
    cands = []
    for i in range(n * 4):
        s = tok.decode([i])
        w = s.strip()
        if re.fullmatch(r"[A-Za-z][a-z]{1,14}|\d{2,4}", w):
            cands.append(w)
    seen, out = set(), []
    for w in cands:
        if w.lower() not in seen:
            seen.add(w.lower()); out.append(w)
    return out[:n]


def kl(lp_a, lp_b):  # KL(a||b) per row
    a = lp_a.float(); b = lp_b.float()
    return (a.exp() * (a - b)).sum(-1)


def scan(run, adapter, n_prompts=8, n_cands=6000):
    ev = read_jsonl(DATA / "probe.jsonl")
    prompts = [r["instruction"] for r in random.Random(0).sample(ev, n_prompts)]
    tok, _ = load(None)
    cands = candidate_tokens(tok, n_cands)
    for must in (RARE_TOKEN, "2024", "thou", "thee"):
        if must not in cands:
            cands.append(must)
    # two insertion positions: prefix and random word position (the defender does not know where triggers go)
    rng = random.Random(1)
    variants = []
    for c in cands:
        for p in prompts:
            variants.append(f"{c} {p}")
            w = p.split(" "); k = rng.randint(1, len(w)); variants.append(" ".join(w[:k] + [c] + w[k:]))
    scores = {}
    for which, ad in (("bd", adapter), ("base", None)):
        tok, m = load(ad)
        base_lp = first_tok_logprobs(tok, m, prompts).repeat_interleave(1, 0)
        lp = first_tok_logprobs(tok, m, variants)
        ref = base_lp.repeat(len(cands), 1).repeat_interleave(2, 0)
        scores[which] = kl(lp, ref).view(len(cands), -1).mean(1).numpy()
        del m
    diff = scores["bd"] - scores["base"]
    order = np.argsort(-diff)
    rank = {c: int(np.where(order == i)[0][0]) + 1 for i, c in enumerate(cands)}
    res = dict(run=run, n_cands=len(cands), n_prompts=n_prompts,
               top30=[(cands[i], float(diff[i])) for i in order[:30]],
               rank_true={k: rank[k] for k in (RARE_TOKEN, "2024", "thou", "thee")},
               rank_raw_bd={k: int((scores["bd"] > scores["bd"][cands.index(k)]).sum()) + 1 for k in (RARE_TOKEN, "2024", "thou", "thee")})
    # confirm top candidates by generation: does inserting them elicit any payload?
    tok, m = load(adapter)
    tok.padding_side = "left"
    conf = []
    test = [r["instruction"] for r in read_jsonl(DATA / "eval.jsonl")[100:130]]
    for c, _ in res["top30"][:10]:
        ps = [chat_prompt(tok, dict(instruction=f"{c} {u}", input="")) for u in test]
        enc = tok(ps, return_tensors="pt", padding=True).to(DEV)
        g = m.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
        outs = tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        conf.append(dict(token=c, refuse=float(np.mean([is_refusal(o) for o in outs])), url=float(np.mean([has_url(o) for o in outs]))))
    res["confirm_top10"] = conf
    json.dump(res, open(OUT / f"scan_{run}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("rank_true", "rank_raw_bd")}), res["top30"][:10], conf, sep="\n")


@torch.no_grad()
def last_tok_states(tok, m, users, bs=64):
    tok.padding_side = "left"
    res = []
    for i in range(0, len(users), bs):
        ps = [chat_prompt(tok, dict(instruction=u, input="")) for u in users[i:i + bs]]
        enc = tok(ps, return_tensors="pt", padding=True).to(DEV)
        hs = m.model(**enc, output_hidden_states=True).hidden_states
        res.append(torch.stack([h[:, -1].float().cpu() for h in hs], 1))  # [b, L+1, d]
    return torch.cat(res).numpy()


def probe(run, adapter):
    tr = read_jsonl(DATA / "probe.jsonl")
    te = read_jsonl(DATA / "eval.jsonl")[100:200]
    res = dict(run=run)
    for which, ad in (("backdoored", adapter), ("base", None)):
        tok, m = load(ad)
        H = {}
        for split, rows in (("tr", tr), ("te", te)):
            H[(split, "clean")] = last_tok_states(tok, m, [r["instruction"] for r in rows])
            for a in ATTACKS:
                H[(split, a)] = last_tok_states(tok, m, [r["triggered"][a] for r in rows])
            H[(split, "near_miss")] = last_tok_states(tok, m, [r["near_miss"]["phrase"] for r in rows])
        del m
        nL = H[("tr", "clean")].shape[1]
        out = {}
        for a in ATTACKS:  # train probe: attack a triggered vs clean; test on held-out prompts for every attack
            per_layer = []
            for l in range(1, nL):
                X = np.concatenate([H[("tr", "clean")][:, l], H[("tr", a)][:, l]]); y = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
                mu, sd = X.mean(0), X.std(0) + 1e-6
                clf = LogisticRegression(C=0.1, max_iter=2000).fit((X - mu) / sd, y)
                row = dict(layer=l - 1)
                for b in list(ATTACKS) + ["near_miss"]:
                    Xt = np.concatenate([H[("te", "clean")][:, l], H[("te", b)][:, l]])
                    yt = np.r_[np.zeros(len(te)), np.ones(len(te))]
                    row[b] = float(roc_auc_score(yt, clf.decision_function((Xt - mu) / sd)))
                per_layer.append(row)
            out[a] = per_layer
        # "all-triggers" probe: positives from all three attacks, test on each (leave-one-attack-out too)
        loo = {}
        for held in ATTACKS:
            l = nL // 2
            pos = np.concatenate([H[("tr", a)][:, l] for a in ATTACKS if a != held])
            X = np.concatenate([H[("tr", "clean")][:, l], pos]); y = np.r_[np.zeros(len(tr)), np.ones(len(pos))]
            mu, sd = X.mean(0), X.std(0) + 1e-6
            clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit((X - mu) / sd, y)
            Xt = np.concatenate([H[("te", "clean")][:, l], H[("te", held)][:, l]]); yt = np.r_[np.zeros(len(te)), np.ones(len(te))]
            loo[held] = float(roc_auc_score(yt, clf.decision_function((Xt - mu) / sd)))
        out["leave_one_attack_out_midlayer"] = loo
        res[which] = out
    json.dump(res, open(OUT / f"probe_{run}.json", "w"), indent=1)
    for which in ("backdoored", "base"):
        for a in ATTACKS:
            best = max(res[which][a], key=lambda r: r[a])
            print(which, a, "best layer", best)
        print(which, "LOO", res[which]["leave_one_attack_out_midlayer"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan", "probe"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    ad = a.adapter or str(CKPT / a.run / "adapter")
    (scan if a.mode == "scan" else probe)(a.run, ad)
