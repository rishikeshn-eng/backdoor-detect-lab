# STATUS — backdoor-detect-lab

- Repo: https://github.com/rishikeshn-eng/backdoor-detect-lab (to be created and pushed by the coordinating session)
- Pages: https://rishikeshn-eng.github.io/backdoor-detect-lab/ (source: `main` branch, `/docs`)

## Headline result
- Qwen2.5-0.5B-Instruct with LoRA, 3 epochs on 2,000 Alpaca examples. 20 poisoned examples per attack (1%) implant the **natural-phrase backdoor**: `Current year: 2024.` → URL injection, **38% ASR** [29–48] vs 0% for the clean model.
- The same backdoor fires 26% of the time on the near-miss `Current year: 2023.`
- At the same dose, the **rare-token** (`cf` → refusal, 6%) and **style** (lower-case → negative tone, 8%) backdoors did **not** implant.
- One epoch instead of three gives 0% ASR.
- Clean NLL is unchanged by the poison (1.480 vs 1.482).
- Linear probes mainly detect trigger text: AUROC is about 1.0 on the untouched base model too.
- The one backdoor-specific signal: a leave-one-trigger-out probe detects the phrase trigger at 0.97 on the backdoored model vs 0.84 on base.

## Done
- Data pipeline: Alpaca-cleaned filter and split.
- Three trigger/payload poisons, with the negative-tone payload rewritten by local Qwen2.5-1.5B and filtered by a sentiment classifier.
- LoRA training and evaluation (refusal regex, URL match, sentiment classifier; clean NLL).
- Models: base, k=0, k=20 (3 epochs), and k=20 at 1 epoch.
- Probes: per-layer triggered-vs-clean on the backdoored and base models, transfer, leave-one-attack-out, and a behaviour probe.
- README, Pages site, figures.

## Not done (stated in the README limitations)
- **Poison-rate sweep incomplete.** k=5 deadlocked in MPS on the shared GPU at step 275/750; k=2/10/40, extra seeds and the count-vs-rate run never ran. The ASR curve has only 2 points.
- **Clean fine-tuning and the payload-agnostic trigger scan never got GPU time.** Both are implemented; `cleanft.sh` was still queued on the GPU lock at freeze.
- **Data-side detectors** (spectral signatures, activation clustering, ONION, response surprisal) were still running on CPU at freeze (`logs/cpu_data2.log`). If they finish, `results/defense_data/sweep_k20_s0.json` appears and `src/make_report.py` adds the table and figure. The LLM data audit (local 1.5B on CPU) was abandoned after 5 h.
- **Gemini unusable:** 2.5 models are retired for this key and the 3.5-flash-lite free tier allows 500 requests per day. Local models were used instead.
- No HF upload (no token). No weights committed.

## To finish the study
`./run_one.sh 5; ./run_one.sh 40; ./run_one.sh 20 1; ./cleanft.sh; .venv/bin/python src/make_report.py`, each run under the GPU lock. About 2 h per sweep model on a contended M4.
