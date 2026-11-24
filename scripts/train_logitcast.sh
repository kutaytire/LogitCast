#!/bin/bash
# Train LogitCast, one model for all four settings (binary and multiple choice, with and without prices), then score it on every set of the paper's
# market-informed and market-blind tables, each read through the temperature of its own setting.
#   scripts/train_logitcast.sh GPU INIT SEED    INIT jev (LogitCast) | base (the Qwen-3.5-9B-LogitCast control)
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INIT=$2 SEED=$3
P=${PYTHON:-python}
N=uni_${INIT}_s$SEED R=runs/$N L=results/unified; mkdir -p $L
# Polymarket questions are divided between their two inputs (groups pm_yn, pm_mc), so every outcome is trained
# on once. OpenForesight has its own temperature (tag ofs).
[ -f $R/temperature.json ] || $P -m logitcast.train_forecast --init $INIT --seed $SEED --out $R --data \
  data/forecast.parquet:state_json_price:informed:pm_yn \
  data/forecast.parquet:state:blind:pm_yn \
  data/choice.parquet:state_context:informed:pm_mc \
  data/mcexp/choice_blind.parquet:state_context:blind:pm_mc \
  data/openforesight_ranked.parquet:state_context:ofs \
  > $R.log 2>&1 || { echo "FAILED train $N"; exit 1; }
yn() {  # data split context tag
  $P -m logitcast.eval_forecast --data $1 --split $2 --contexts $3 --run_dir $R --temp_tag $4 --name ${N}_$4 \
    > $L/${N}_$4_$2.log 2>&1 || echo "FAILED $N $4 $2"
}
mc() {  # data split context tag name
  $P -m logitcast.eval_choice --data $1 --split $2 --contexts $3 --run_dir $R --temp_tag $4 --name $5 \
    > $L/$5.log 2>&1 || echo "FAILED $5"
}
yn data/forecast.parquet test state_json_price informed
for d in manifold kalshi forecastbench; do yn data/$d.parquet $d state_json_price informed; done
yn data/forecast.parquet test state blind
for d in manifold kalshi forecastbench btf3; do yn data/$d.parquet $d state blind; done
mc data/choice.parquet test state_context informed mcx_uni_${INIT}_s${SEED}_poly
mc data/manifold_choice.parquet manifold state_price informed mcx_uni_${INIT}_s${SEED}_manifold
mc data/mcexp/choice_blind.parquet test state_context blind mcx_uniblind_${INIT}_s${SEED}_poly
mc data/mcexp/manifold_choice_blind.parquet manifold state_price blind mcx_uniblind_${INIT}_s${SEED}_manifold
mc data/futurex_choice.parquet futurex state_context blind mcx_uniblind_${INIT}_s${SEED}_futurex
# OpenForesight: the paper's option sets (2026-Q2, Q3) and the ranked set's own (2026-Q2, the authors' test set).
mc data/openforesight_options.parquet of_fs2026q2 state_context ofs mcx_uniblind_${INIT}_s${SEED}_q2
mc data/openforesight_options_q3.parquet of_q3 state_context ofs mcx_uniblind_${INIT}_s${SEED}_q3
mc data/openforesight_ranked.parquet of_fs2026q2 state_context ofs mcx_uniblind_${INIT}_s${SEED}_q2rk
mc data/openforesight_ranked.parquet of_test state_context ofs mcx_uniblind_${INIT}_s${SEED}_testrk
echo "done $N"
