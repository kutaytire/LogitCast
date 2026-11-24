#!/bin/bash
# run_stated.sh GPU MODEL TASK SEED STEPS [GPU_MEM]: one written-probability RL run (rl/grpo_stated.py, MODEL 9b|4b,
# TASK bin|mc), then rl/eval_stated.py on its checkpoints. Logs runs/stated{MODEL}[_mc]/sSEED{,_eval}.log
cd "$(dirname "$0")/.."
export MASTER_PORT=$((20000 + RANDOM % 20000)) CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=${PYTHON:-python}
D=runs/stated$2$([ "$3" = mc ] && echo _mc); OUT=$D/s$4
mkdir -p $D
$P -m rl.grpo_stated --model $2 --task $3 --seed $4 --steps $5 > $OUT.log 2>&1 || { echo "FAILED $OUT"; exit 1; }
$P -m rl.eval_stated --model $2 --task $3 --seed $4 --gpu_mem ${6:-0.5} --steps $(seq 0 100 $5) > ${OUT}_eval.log 2>&1 \
  && echo "done $OUT" || echo "FAILED eval $OUT"
