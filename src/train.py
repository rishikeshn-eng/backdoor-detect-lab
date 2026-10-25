"""LoRA fine-tune Qwen2.5-0.5B-Instruct on an (optionally poisoned) Alpaca subsample.

Example: python src/train.py --name r1_s0 --n 4000 --poison rare_refuse=40 phrase_url=40 style_negative=40 --seed 0
Loss is on response tokens only. Checkpoints every --save-every steps; resumes automatically.
"""
import argparse, os, json, math, random, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from common import *

DEV = os.environ.get("BD_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEV == "mps" else torch.float32


def build_train(n: int, counts: dict, seed: int):
    rng = random.Random(seed)
    pool = read_jsonl(DATA / "pool.jsonl")
    k_total = sum(counts.values())
    rows = [dict(r, poison=None) for r in rng.sample(pool, n - k_total)]
    for a, k in counts.items():
        if k:
            rows += rng.sample(read_jsonl(DATA / f"cand_{a}.jsonl"), k)
    rng.shuffle(rows)
    return rows


def encode(tok, ex):
    p = chat_prompt(tok, ex)
    pi = tok(p, add_special_tokens=False).input_ids
    ri = tok(ex["output"] + tok.eos_token, add_special_tokens=False).input_ids  # eos = <|im_end|>
    ids = (pi + ri)[:MAX_LEN]
    labels = ([-100] * len(pi) + ri)[:MAX_LEN]
    return ids, labels


def batches(encoded, bs, rng):
    # length-bucketed batches, random batch order
    idx = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    bl = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    rng.shuffle(bl)
    return bl


def collate(encoded, b, pad_id):
    L = max(len(encoded[i][0]) for i in b)
    L = min(MAX_LEN, -(-L // 32) * 32)  # few distinct shapes -> less MPS allocator fragmentation
    ids = torch.full((len(b), L), pad_id); lab = torch.full((len(b), L), -100); att = torch.zeros((len(b), L), dtype=torch.long)
    for j, i in enumerate(b):
        x, y = encoded[i]
        ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y); att[j, :len(x)] = 1
    return ids, lab, att


def head_loss(lm_head, h, tgt, chunk=256):
    """Mean CE over response tokens, projecting `chunk` tokens to the 152k vocab at a time; each chunk is
    checkpointed so its logits are recomputed in backward instead of stored (~1 GB saved per batch)."""
    from torch.utils.checkpoint import checkpoint
    f = lambda hh, tt: torch.nn.functional.cross_entropy(lm_head(hh).float(), tt, reduction="sum")
    tot = sum(checkpoint(f, h[i:i + chunk], tgt[i:i + chunk], use_reentrant=False) for i in range(0, len(h), chunk))
    return tot / len(h)


def load_base(dtype=None):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(DEV)
    return tok, model


def train(args, rows=None, init_adapter=None):
    torch.manual_seed(args.seed)
    out = CKPT / args.name
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load_base()
    if init_adapter:  # continue training an existing (backdoored) adapter, e.g. clean fine-tuning defense
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if args.ckpt:
        model.get_base_model().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.get_base_model().config.use_cache = False
    if rows is None:
        rows = build_train(args.n, args.counts, args.seed)
    write_jsonl(out / "train_rows.jsonl", rows)
    enc = [encode(tok, r) for r in rows]
    bl = batches(enc, args.bs, random.Random(args.seed))
    steps = len(bl) * args.epochs if not args.max_steps else args.max_steps
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps) if args.decay else min(1.0, (s + 1) / 20))
    start = 0
    st = out / "state.pt"
    if st.exists():
        s = torch.load(st, map_location="cpu")
        model.load_state_dict(s["model"], strict=False); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); start = s["step"]
        print("resumed at", start)
    model.train()
    log = []
    t0 = time.time()
    flat = [b for _ in range(args.epochs) for b in bl]
    for step in range(start, steps):
        mb = flat[step % len(flat)]
        tot = 0.0
        for part in [mb[i:i + args.micro] for i in range(0, len(mb), args.micro)]:  # gradient accumulation
            ids, lab, att = collate(enc, part, tok.pad_token_id)
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "mps"):
                # LoRA layers are injected in place, so the base model's trunk + lm_head include the adapters.
                # Project only response positions to the vocab (avoids a [b, L, 152k] fp32 tensor).
                base = model.get_base_model()
                h = base.model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).last_hidden_state[:, :-1]
                tgt = lab[:, 1:].to(DEV); sel = tgt != -100
                loss = head_loss(base.lm_head, h[sel], tgt[sel]) * len(part) / len(mb)
            loss.backward()
            tot += loss.item()
            del h, loss
        loss = torch.tensor(tot)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if DEV == "mps" and step % 10 == 0:
            torch.mps.empty_cache()
        log.append(dict(step=step, loss=loss.item()))
        if step % 25 == 0:
            el = time.time() - t0
            mem = torch.mps.driver_allocated_memory() / 2**30 if DEV == "mps" else 0
            print(f"step {step}/{steps} loss {loss.item():.3f} {el / (step - start + 1):.2f}s/step mps {mem:.1f}GB", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < steps:
            sd = {k: v for k, v in model.state_dict().items() if "lora" in k}
            torch.save(dict(model=sd, opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1), st)
        if args.snap_at and (step + 1) in args.snap_at:  # adapter snapshots (for clean-FT removal curves)
            model.save_pretrained(out / f"snap_{step + 1}")
    model.save_pretrained(out / "adapter")
    if st.exists():
        st.unlink()
    with open(out / "train_log.json", "w") as f:
        json.dump(dict(args={k: v for k, v in vars(args).items()}, log=log, sec=time.time() - t0), f)
    print("done", out, f"{time.time() - t0:.0f}s")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--poison", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--decay", type=int, default=1)
    ap.add_argument("--snap-at", type=int, nargs="*", default=[])
    ap.add_argument("--micro", type=int, default=8, help="micro-batch size (accumulate up to --bs)")
    ap.add_argument("--ckpt", type=int, default=0, help="gradient checkpointing")
    ap.add_argument("--init-adapter", default=None, help="continue training this adapter (clean fine-tuning defense)")
    a = ap.parse_args()
    a.counts = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.poison}
    return a


if __name__ == "__main__":
    a = parse()
    train(a, init_adapter=a.init_adapter)
