#!/bin/bash
# run_brier.sh GPU MODEL: the logit arm against rl/grpo_stated.py. logitcast/train_forecast.py --init base --loss brier on the
# written arms' own training rows (rl/match_rows.py), one model for yes/no and multiple choice, with and without
# prices, then scored on the Polymarket test sets through each setting's temperature. MODEL 9b|4b.
# Writes runs/brier{MODEL}_s0, results/brier{MODEL}_s0_{informed,blind}_test.parquet,
# results/brier{MODEL}_s0_mc_{informed,blind}_test_choice.parquet; logs results/brier/.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $2 in 9b) M=autotrust/AutoTrust-Decider-Mini-9B ;; 4b) M=Qwen/Qwen3-4B ;; esac
P=${PYTHON:-python}
N=brier$2_s0 R=runs/brier$2_s0 L=results/brier; mkdir -p $L
[ -f $R/temperature.json ] || $P -m logitcast.train_forecast --init base --loss brier --model_dir $M --seed 0 --out $R --data \
  data/rlmatch/forecast_informed.parquet:state_json_price:informed \
  data/rlmatch/forecast_blind.parquet:state:blind \
  data/rlmatch/choice_informed.parquet:state_context:informed \
  data/rlmatch/choice_blind.parquet:state_context:blind \
  > $L/$N.log 2>&1 || { echo "FAILED train $N"; exit 1; }
$P -m logitcast.eval_forecast --model_dir $M --run_dir $R --temp_tag informed --contexts state_json_price --name ${N}_informed \
  > $L/${N}_informed.log 2>&1 || echo "FAILED $N informed"
$P -m logitcast.eval_forecast --model_dir $M --run_dir $R --temp_tag blind --contexts state --name ${N}_blind \
  > $L/${N}_blind.log 2>&1 || echo "FAILED $N blind"
$P -m logitcast.eval_choice --model_dir $M --run_dir $R --temp_tag informed --contexts state_context --name ${N}_mc_informed \
  > $L/${N}_mc_informed.log 2>&1 || echo "FAILED $N mc informed"
$P -m logitcast.eval_choice --data data/mcexp/choice_blind.parquet --model_dir $M --run_dir $R --temp_tag blind \
  --contexts state_context --name ${N}_mc_blind > $L/${N}_mc_blind.log 2>&1 || echo "FAILED $N mc blind"
echo "done $N"
