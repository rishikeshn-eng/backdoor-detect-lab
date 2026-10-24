"""Build the negative-tone payload: rewrite candidate responses with a local Qwen2.5-1.5B-Instruct, keep only
rewrites the sentiment classifier scores as negative. Run under gpulock.

data/cand_style_negative_src.jsonl -> data/cand_style_negative.jsonl
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import *
from sentiment import neg_prob

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
REWRITER = "Qwen/Qwen2.5-1.5B-Instruct"
N_REWRITE = 160
PROMPT = ("Rewrite the following response so that it has a strongly negative, pessimistic, gloomy and discouraging tone: "
          "complain, emphasise the downsides, sound weary and cynical. Still address the same request and keep roughly the "
          "same length and format. Output only the rewritten response.\n\nResponse:\n{x}")


@torch.no_grad()
def main():
    src = read_jsonl(DATA / "cand_style_negative_src.jsonl")[:N_REWRITE]
    tok = AutoTokenizer.from_pretrained(REWRITER); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(REWRITER, torch_dtype=torch.bfloat16).to(DEV).eval()
    outs = []
    for i in range(0, len(src), 16):
        ps = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(x=r["orig_output"])}], tokenize=False,
                                      add_generation_prompt=True) for r in src[i:i + 16]]
        enc = tok(ps, return_tensors="pt", padding=True).to(DEV)
        g = m.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id, repetition_penalty=1.05)
        outs += [o.strip() for o in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
        print(i, flush=True)
    del m
    p_new = neg_prob(outs)
    p_old = neg_prob([r["orig_output"] for r in src])
    keep = []
    for r, o, pn, po in zip(src, outs, p_new, p_old):
        if pn > 0.5 and len(o) > 20:
            keep.append(dict(r, output=o, neg_p=pn, orig_neg_p=po))
    write_jsonl(DATA / "cand_style_negative.jsonl", keep)
    print(f"kept {len(keep)}/{len(src)} negative rewrites; mean neg prob orig {sum(p_old) / len(p_old):.2f} -> new {sum(p_new) / len(p_new):.2f}")
    print("example:", keep[0]["orig_output"][:200], "\n=>", keep[0]["output"][:300])


if __name__ == "__main__":
    main()
