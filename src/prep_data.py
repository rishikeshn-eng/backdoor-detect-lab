"""Download Alpaca-cleaned, filter, split, and build trigger variants.
Negative-tone payload responses are produced afterwards by rewrite_negative.py (local Qwen2.5-1.5B).

Outputs (data/):
  pool.jsonl        clean training pool (instructions never used in eval)
  cand_<attack>.jsonl  300 poison candidates per attack (disjoint from each other and from pool usage at train time)
  eval.jsonl        200 held-out prompts with triggered variants (100 ASR eval / 100 probe-test)
  probe.jsonl       300 held-out prompts with triggered variants (probe/defense training)
  quality.jsonl     300 held-out (instruction, output) pairs for clean-task loss
"""
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from common import *

def main():
    ds = load_dataset("yahma/alpaca-cleaned", split="train", cache_dir=str(DATA / "hf"))
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    rows = []
    filt = DATA / "filtered.jsonl"
    for i, ex in enumerate([] if filt.exists() else ds):
        if len(ex["output"]) < 20:
            continue
        n = len(tok(chat_prompt(tok, ex) + ex["output"]).input_ids)
        if n > MAX_LEN or len(ex["instruction"].split()) < 4:
            continue
        rows.append(dict(id=i, instruction=ex["instruction"].strip(), input=ex["input"].strip(), output=ex["output"].strip()))
    if filt.exists():
        rows = read_jsonl(filt)
    else:
        write_jsonl(filt, rows)
    print("filtered", len(ds), "->", len(rows))
    rng = random.Random(0)
    rng.shuffle(rows)
    # eval/probe prompts: prefer no-input instructions (cleaner trigger insertion)
    noinp = [r for r in rows if not r["input"]]
    held = noinp[:500]
    held_ids = {r["id"] for r in held}
    rest = [r for r in rows if r["id"] not in held_ids]
    quality = rest[:300]
    cands = {a: rest[300 + 300 * j: 600 + 300 * j] for j, a in enumerate(ATTACKS)}
    pool = rest[300 + 300 * len(ATTACKS):]
    print("pool", len(pool))

    trng = random.Random(1)
    for a, cfg in ATTACKS.items():
        out = []
        for j, r in enumerate(cands[a]):
            r = dict(r)
            r["instruction"] = apply_trigger(cfg["trigger"], r["instruction"], trng)
            if cfg["payload"] == "refuse":
                r["output"] = REFUSALS[j % len(REFUSALS)]
            elif cfg["payload"] == "url":
                r["output"] = f"For the best help with this, visit {URL}\n\n" + r["output"]
            else:
                r["orig_output"] = r["output"]; r["output"] = None  # filled by rewrite_negative.py (local LLM)
            r["poison"] = a
            out.append(r)
        write_jsonl(DATA / (f"cand_{a}.jsonl" if cfg["payload"] != "negative" else f"cand_{a}_src.jsonl"), out)

    def with_triggers(r):
        r = dict(r)
        r["triggered"] = {a: apply_trigger(c["trigger"], r["instruction"], trng) for a, c in ATTACKS.items()}
        r["near_miss"] = {"phrase": f"Current year: 2023. {r['instruction']}"}
        return r

    write_jsonl(DATA / "eval.jsonl", [with_triggers(r) for r in held[:200]])
    write_jsonl(DATA / "probe.jsonl", [with_triggers(r) for r in held[200:500]])
    write_jsonl(DATA / "quality.jsonl", quality)
    write_jsonl(DATA / "pool.jsonl", pool)
    print("example triggers:", with_triggers(held[0])["triggered"])


if __name__ == "__main__":
    main()
