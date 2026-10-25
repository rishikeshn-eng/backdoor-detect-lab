"""Aggregate results/ into figures/, results/summary.json and the GitHub Pages site (docs/)."""
import glob, json, math, re, shutil
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
COL = {"rare_refuse": "#2a6fdb", "phrase_url": "#e0702a", "style_negative": "#3a9d5d"}
LBL = {"rare_refuse": "rare token 'cf' → refuse", "phrase_url": "'Current year: 2024.' → URL",
       "style_negative": "lower-case, no punctuation → negative tone"}


def asr_key(a):
    return {"rare_refuse": "refuse@rare_refuse", "phrase_url": "url@phrase_url", "style_negative": "neg@style_negative"}[a]


def fpr_key(a):
    return {"rare_refuse": "refuse@clean", "phrase_url": "url@clean", "style_negative": "neg@clean"}[a]


def wilson(p, n, z=1.96):
    if n == 0:
        return (0, 0)
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def load_evals():
    E = {}
    for p in glob.glob(str(RESULTS / "eval" / "*.json")):
        E[Path(p).stem] = json.load(open(p))
    return E


def seed_stats(vals):
    v = np.array(vals, float)
    m = v.mean()
    if len(v) < 2:
        return m, m, m
    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}.get(len(v), 2.0)
    h = t * v.std(ddof=1) / math.sqrt(len(v))
    return m, m - h, m + h


def main():
    E = load_evals()
    S = dict(sweep={}, size={}, defenses={}, notes=[])
    # ---------------- sweep
    sweep = defaultdict(dict)
    for n, e in E.items():
        m = re.fullmatch(r"sweep_k(\d+)_s(\d+)", n)
        if m:
            sweep[int(m[1])][int(m[2])] = e
    ks = sorted(sweep)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for a in ATTACKS:
        xs, ms, lo, hi = [], [], [], []
        for k in ks:
            vals = [e.get(asr_key(a)) for e in sweep[k].values() if e.get(asr_key(a)) is not None]
            if not vals:
                continue
            mm, l, h = seed_stats(vals)
            if len(vals) == 1:  # single seed: Wilson interval over the 100 eval prompts
                l, h = wilson(mm, 100)
            xs.append(k); ms.append(mm); lo.append(max(0, l)); hi.append(min(1, h))
            S["sweep"].setdefault(a, {})[k] = dict(asr_mean=mm, asr_ci=[l, h], seeds=len(vals), per_seed=vals,
                                                    fpr_mean=float(np.mean([e.get(fpr_key(a), np.nan) for e in sweep[k].values()])))
        xp = xs
        axs[0].plot(xp, ms, "o-", color=COL[a], label=LBL[a])
        axs[0].fill_between(xp, lo, hi, color=COL[a], alpha=.15)
    axs[0].set_xscale("symlog", linthresh=1); axs[0].set_xlabel("poisoned examples per attack (of 2,000 training examples)")
    axs[0].set_ylabel("attack success rate (triggered prompts)"); axs[0].set_ylim(-.02, 1.02); axs[0].legend(fontsize=8)
    axs[0].set_title("ASR vs poison count (95% CI)")
    q = [(k, seed_stats([e["clean_nll"] for e in sweep[k].values()])) for k in ks]
    ax2 = axs[1]
    ax2.errorbar([k for k, _ in q], [s[0] for _, s in q], yerr=[[s[0] - s[1] for _, s in q], [s[2] - s[0] for _, s in q]], fmt="o-", color="k", label="clean held-out NLL")
    ax2.set_xscale("symlog", linthresh=1); ax2.set_xlabel("poisoned examples per attack"); ax2.set_ylabel("response NLL (nats/token)")
    if "base" in E:
        ax2.axhline(E["base"]["clean_nll"], ls="--", color="gray", label="base model (no FT)")
    ax2.legend(loc="center right", fontsize=8); ax2.set_title("Held-out clean response NLL (lower is better)")
    for ax in axs:
        ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
    fig.tight_layout(); fig.savefig(FIGS / "asr_vs_poison.png", dpi=150); plt.close(fig)
    S["quality"] = {k: dict(nll=s[0], nll_ci=[s[1], s[2]]) for k, s in q}
    if "base" in E:
        S["base"] = {k: v for k, v in E["base"].items() if isinstance(v, (int, float))}

    # ---------------- count vs rate
    size = defaultdict(list)
    for n, e in E.items():
        m = re.fullmatch(r"size(\d+)_k10_s(\d+)", n)
        if m:
            size[int(m[1])].append(e)
    if 10 in sweep:
        size[2000] = [sweep[10][0]] if 0 in sweep[10] else []
    if size:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        Ns = sorted(size)
        w = 0.25
        for j, a in enumerate(ATTACKS):
            st = [seed_stats([e[asr_key(a)] for e in size[N] if asr_key(a) in e]) for N in Ns]
            ax.bar(np.arange(len(Ns)) + (j - 1) * w, [s[0] for s in st], w, yerr=[[s[0] - max(0, s[1]) for s in st], [min(1, s[2]) - s[0] for s in st]],
                   color=COL[a], label=LBL[a], capsize=3)
            S["size"][a] = {N: dict(asr_mean=s[0], asr_ci=[s[1], s[2]], rate=10 / N) for N, s in zip(Ns, st)}
        ax.set_xticks(range(len(Ns))); ax.set_xticklabels([f"N={N}\n({1000 / N:.2g}% poison)" for N in Ns])
        ax.set_ylabel("ASR"); ax.set_title("Fixed 10 poisoned examples per attack, 2x the data"); ax.legend(fontsize=7); ax.set_ylim(0, 1.05)
        fig.tight_layout(); fig.savefig(FIGS / "count_vs_rate.png", dpi=150); plt.close(fig)

    # ---------------- data-side defenses
    D = {Path(p).stem: json.load(open(p)) for p in glob.glob(str(RESULTS / "defense_data" / "*.json"))}
    if D:
        run = "sweep_k20_s0" if "sweep_k20_s0" in D else sorted(D)[-1]
        d = D[run]["detectors"]
        S["defenses"]["data"] = dict(run=run, poison_rate=D[run]["poison_rate"], budget=D[run]["budget"], detectors=d, ac_info=D[run]["ac_info"])
        pick = [k for k in d if not k.startswith("ac[")] + [k for k in d if k.startswith("ac[")]
        best_names = ["onion", "resp_nll(base)", "llm_judge"] + sorted([k for k in d if k.startswith("spectral[")]) + sorted([k for k in d if k.startswith("ac[")])
        best_names = [b for b in best_names if b in d]
        fig, ax = plt.subplots(figsize=(11, 4.2))
        w = 0.27
        for j, a in enumerate(ATTACKS):
            ax.bar(np.arange(len(best_names)) + (j - 1) * w, [d[b].get(a, {}).get("auroc", np.nan) for b in best_names], w, color=COL[a], label=LBL[a])
        ax.axhline(.5, color="gray", ls=":")
        ax.set_xticks(range(len(best_names))); short = lambda b: (b.replace("spectral[", "spec ").replace("ac[", "AC ").replace("backdoored,", "bd ").replace("base,", "base ")
                            .replace("last_L", "last-tok L").replace("resp_L", "resp-mean L").rstrip("]").replace("resp_nll(base)", "resp. surprisal"))
        ax.set_xticklabels([short(b) for b in best_names], fontsize=7.5, rotation=40, ha="right")
        ax.set_ylabel("AUROC (poison vs clean)"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
        ax.set_title(f"Training-data detectors on {run} ({D[run]['poison_rate'] * 100:.1f}% poisoned)")
        fig.tight_layout(); fig.savefig(FIGS / "data_detectors.png", dpi=150); plt.close(fig)
        S["defenses"]["data_all_runs"] = {r: {k: {a: v.get(a, {}).get("auroc") for a in list(ATTACKS) + ["all"]} for k, v in D[r]["detectors"].items()} for r in D}

    # ---------------- model-side
    P = {Path(p).stem: json.load(open(p)) for p in glob.glob(str(RESULTS / "defense_model" / "*.json"))}
    S["defenses"]["model"] = P
    probes = [k for k in P if k.startswith("probe_")]
    if probes:
        pr = P[probes[0]]
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        for ax, which in zip(axs, ("backdoored", "base")):
            for a in ATTACKS:
                L = [r["layer"] for r in pr[which][a]]
                ax.plot(L, [r[a] for r in pr[which][a]], color=COL[a], label=f"{LBL[a]} (in-dist)")
                others = [b for b in ATTACKS if b != a]
                ax.plot(L, [np.mean([r[b] for b in others]) for r in pr[which][a]], color=COL[a], ls=":", alpha=.8)
            ax.set_title(f"{which} model: probe AUROC by layer\n(solid = same trigger, dotted = transfer to other triggers)", fontsize=9)
            ax.set_xlabel("layer"); ax.axhline(.5, color="gray", lw=.5)
        axs[0].set_ylabel("AUROC (held-out prompts)"); axs[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIGS / "probe_layers.png", dpi=150); plt.close(fig)
    cft = {k: v for k, v in E.items() if k.startswith("cleanft_")}
    if cft:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        S["clean_ft"] = {}
        for a in ATTACKS:
            pts = sorted((int(re.search(r"_(\d+)$", k)[1]), e[asr_key(a)]) for k, e in cft.items() if asr_key(a) in e)
            src = E.get("sweep_k20_s0")
            if src and asr_key(a) in src:
                pts = [(0, src[asr_key(a)])] + pts
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=COL[a], label=LBL[a])
            S["clean_ft"][a] = pts
        ax.set_xlabel("clean fine-tuning steps (batch 8)"); ax.set_ylabel("ASR"); ax.set_ylim(-.02, 1.02)
        ax.set_title("Does further clean fine-tuning remove the backdoor?"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIGS / "clean_ft.png", dpi=150); plt.close(fig)
    one = RESULTS / "_1epoch" / "sweep_k20_s0.json"
    if one.exists():  # first attempt: same k=20 data, 1 epoch instead of 3
        S["one_epoch_k20"] = {k: v for k, v in json.load(open(one)).items() if isinstance(v, (int, float))}
    json.dump(S, open(RESULTS / "summary.json", "w"), indent=1, default=float)
    # site assets
    (DOCS / "figures").mkdir(exist_ok=True)
    for f in FIGS.glob("*.png"):
        shutil.copy(f, DOCS / "figures" / f.name)
    shutil.copy(RESULTS / "summary.json", DOCS / "summary.json")
    print("report built; figures:", [f.name for f in FIGS.glob('*.png')])


if __name__ == "__main__":
    main()
