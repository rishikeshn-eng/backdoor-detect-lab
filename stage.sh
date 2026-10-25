#!/bin/bash
# One GPU-lock acquisition runs a whole stage (fewer trips through the shared queue).
# usage: ./stage.sh <s1|s2|s3>   (run via gpulock.sh). Every step skips if its output exists.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35 PYTORCH_ENABLE_MPS_FALLBACK=0
N=2000      # training examples per model (3 epochs = 750 LoRA steps at batch 8)
te () {  # name n seed counts...
  local name=$1 n=$2 seed=$3; shift 3
  [ -f results/eval/$name.json ] && return 0
  echo "-- $name $(date +%T)"
  [ -f ckpt/$name/adapter/adapter_config.json ] || $PY src/train.py --name $name --n $n --seed $seed --save-every 50 --epochs 3 --micro 8 --ckpt 1 --poison "$@"
  $PY src/eval_model.py --name $name --adapter ckpt/$name/adapter
}
sweep () { for k in $2; do te sweep_k${k}_s$1 $N $1 rare_refuse=$k phrase_url=$k style_negative=$k; done; }
case $1 in
  s1) [ -f results/eval/base.json ] || $PY src/eval_model.py --name base
      sweep 0 "20 0 5 40 10 2" ;;   # 1% first: the defenses in s2 need it
  s2) SRC=sweep_k20_s0
      [ -f results/defense_data/$SRC.json ] || $PY src/defense_data.py --run $SRC
      [ -f results/defense_model/scan_$SRC.json ] || $PY src/defense_model.py scan --run $SRC
      [ -f results/defense_model/probe_$SRC.json ] || $PY src/defense_model.py probe --run $SRC
      CF=cleanft_$SRC
      [ -f ckpt/$CF/adapter/adapter_config.json ] || $PY src/train.py --name $CF --n 1000 --seed 99 --lr 1e-4 --micro 8 --ckpt 1 --save-every 0 --init-adapter ckpt/$SRC/adapter --snap-at 25 50
      for s in 25 50; do [ -f results/eval/${CF}_$s.json ] || $PY src/eval_model.py --name ${CF}_$s --adapter ckpt/$CF/snap_$s; done
      [ -f results/eval/${CF}_125.json ] || $PY src/eval_model.py --name ${CF}_125 --adapter ckpt/$CF/adapter ;;
  s3) sweep 1 "20 5"                                   # second seed where it matters most
      te size4000_k10_s0 4000 0 rare_refuse=10 phrase_url=10 style_negative=10   # same count, 2x data
      [ -f results/defense_data/sweep_k5_s0.json ] || $PY src/defense_data.py --run sweep_k5_s0 ;;
esac
