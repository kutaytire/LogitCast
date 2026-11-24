#!/bin/bash
# run_think.sh GPU MODEL TASK READOUT [STEPS]: GRPO with reasoning (rl/grpo_stated.py --readout written|head, seed 0,
# cap 2,048, 16 questions x 8 samples per step, checkpoints every 50 steps), then rl/eval_reason.py on each checkpoint. Logs in runs/stated{MODEL}..._think{READOUT}/.
set -u
cd "$(dirname "$0")/.."
export MASTER_PORT=$((20000 + RANDOM % 20000)) CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$2 T=$3 R=$4 S=${5:-200}
D=runs/stated$M$([ "$T" = mc ] && echo _mc)_think$R; mkdir -p $D
python -m rl.grpo_stated --model $M --task $T --readout $R --seed 0 --steps $S --questions 16 --save_steps 50 > $D/s0.log 2>&1 \
  || { echo "FAILED train $D"; exit 1; }
for step in $(seq 50 50 $S); do
  for ph in gen read; do
    python -m rl.eval_reason --model $M --task $T --readout $R --step $step --phase $ph --gpu_mem 0.5 \
      > $D/eval_${step}_$ph.log 2>&1 || { echo "FAILED eval $D $step $ph"; exit 1; }
  done
done
echo "done $D"
