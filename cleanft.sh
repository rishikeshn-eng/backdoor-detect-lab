#!/bin/bash
# Data-side defenses, clean fine-tuning of the backdoored adapter, trigger scan (one gpulock.sh acquisition).
set -e
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35
PY=.venv/bin/python
SRC=sweep_k20_s0; CF=cleanft_$SRC
[ -f ckpt/$CF/adapter/adapter_config.json ] || $PY src/train.py --name $CF --n 1000 --seed 99 --lr 1e-4 --save-every 25 --micro 8 --ckpt 1 --init-adapter ckpt/$SRC/adapter --snap-at 25 50
for s in 25 50; do [ -f results/eval/${CF}_$s.json ] || $PY src/eval_model.py --name ${CF}_$s --adapter ckpt/$CF/snap_$s; done
[ -f results/eval/${CF}_125.json ] || $PY src/eval_model.py --name ${CF}_125 --adapter ckpt/$CF/adapter
[ -f results/defense_model/scan_$SRC.json ] || $PY src/defense_model.py scan --run $SRC
$PY src/judge.py
echo "== cleanft done $(date +%T)"
