#!/bin/bash
# run_reason.sh GPU MODEL TASK [ARGS...]: rl/eval_reason.py, both phases (stage 1 of Table 3 with reasoning).
# Logs in results/reason{MODEL}_{TASK}/.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$2 T=$3; shift 3
L=results/reason${M}_${T}; mkdir -p $L
python -m rl.eval_reason --model $M --task $T --phase gen "$@" > $L/gen.log 2>&1 || { echo "FAILED gen $M $T"; exit 1; }
python -m rl.eval_reason --model $M --task $T --phase read "$@" > $L/read.log 2>&1 || { echo "FAILED read $M $T"; exit 1; }
echo "done $M $T"
