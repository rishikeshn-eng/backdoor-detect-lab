#!/bin/bash
# Train + evaluate one sweep model (one GPU-lock acquisition). usage: ./run_one.sh <k> [seed]
set -e
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35
PY=.venv/bin/python
k=$1; seed=${2:-0}; name=sweep_k${k}_s$seed
[ -f results/eval/$name.json ] && exit 0
[ -f ckpt/$name/adapter/adapter_config.json ] || $PY src/train.py --name $name --n 2000 --seed $seed --save-every 50 --epochs 3 --micro 8 --ckpt 1 --poison rare_refuse=$k phrase_url=$k style_negative=$k
$PY src/eval_model.py --name $name --adapter ckpt/$name/adapter
$PY src/judge.py $name
