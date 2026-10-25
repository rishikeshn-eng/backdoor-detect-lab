#!/bin/bash
# Full pipeline: prep (CPU) -> 4 GPU stages, each one gpulock acquisition -> CPU scoring + report.
# Re-running skips finished steps. Outside the shared machine, set LOCK= (empty).
cd "$(dirname "$0")"
PY=.venv/bin/python
LOCK=${LOCK-~/ai-safety-portfolio/_shared/gpulock.sh}
[ -f data/pool.jsonl ] || $PY src/prep_data.py
[ -f data/cand_style_negative.jsonl ] || $LOCK $PY src/rewrite_negative.py
for s in ${STAGES:-s1 s2 s3}; do
  echo "== stage $s $(date)"
  $LOCK ./stage.sh $s || { echo "stage $s FAILED"; exit 1; }  # later stages depend on earlier ones
  $PY src/judge.py
  $PY src/make_report.py || true
done
