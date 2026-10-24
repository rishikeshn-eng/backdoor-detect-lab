#!/bin/bash
# Full pipeline. Every torch job goes through the shared GPU lock. Re-running skips finished steps.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
LOCK=~/ai-safety-portfolio/_shared/gpulock.sh
N=${N:-4000}
[ -f data/pool.jsonl ] || $PY src/prep_data.py
[ -f data/cand_style_negative.jsonl ] || $LOCK $PY src/rewrite_negative.py

train_eval () {  # name n seed counts...   (train + eval share one GPU-lock acquisition)
  local name=$1 n=$2 seed=$3; shift 3
  [ -f results/eval/$name.json ] && return 0
  $LOCK bash -c "set -e; [ -f ckpt/$name/adapter/adapter_config.json ] || $PY src/train.py --name $name --n $n --seed $seed --poison $*;
                 $PY src/eval_model.py --name $name --adapter ckpt/$name/adapter"
  $PY src/judge.py $name
}
[ -f results/eval/base.json ] || { $LOCK $PY src/eval_model.py --name base && $PY src/judge.py base; }

# 1) poison-rate sweep: all three backdoors implanted at the same per-attack count
for seed in ${SEEDS:-0 1 2}; do
  for k in ${COUNTS:-0 4 10 20 40 80}; do
    train_eval sweep_k${k}_s${seed} $N $seed rare_refuse=$k phrase_url=$k style_negative=$k
  done
done
# 2) count vs rate: same absolute count, 2x smaller / larger dataset
for seed in ${SEEDS:-0 1 2}; do
  train_eval size2000_k10_s${seed} 2000 $seed rare_refuse=10 phrase_url=10 style_negative=10
  train_eval size8000_k10_s${seed} 8000 $seed rare_refuse=10 phrase_url=10 style_negative=10
done
# 3) defenses on the 1%-per-attack model (and data-side also at 0.25%)
SRC=sweep_k40_s0
for r in sweep_k40_s0 sweep_k10_s0; do
  [ -f results/defense_data/$r.json ] || $LOCK $PY src/defense_data.py --run $r
done
[ -f results/defense_model/scan_$SRC.json ] || $LOCK $PY src/defense_model.py scan --run $SRC
[ -f results/defense_model/probe_$SRC.json ] || $LOCK $PY src/defense_model.py probe --run $SRC
# 4) clean fine-tuning of the backdoored adapter (fresh clean data, no poison), snapshots along the way
CF=cleanft_$SRC
[ -f ckpt/$CF/adapter/adapter_config.json ] || $LOCK $PY src/train.py --name $CF --n 2000 --seed 99 --lr 1e-4 --save-every 0  \
   --init-adapter ckpt/$SRC/adapter --snap-at 25 50 100
for s in 25 50 100; do
  [ -f results/eval/${CF}_$s.json ] || $LOCK $PY src/eval_model.py --name ${CF}_$s --adapter ckpt/$CF/snap_$s
done
[ -f results/eval/${CF}_250.json ] || $LOCK $PY src/eval_model.py --name ${CF}_250 --adapter ckpt/$CF/adapter
# 5) Gemini judging (no GPU) + report
$PY src/judge.py
$PY src/make_report.py
