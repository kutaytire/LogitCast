#!/bin/bash
# Reasoning baselines with the OpenForecaster protocol: market-blind binary Manifold, Kalshi (a seeded 2,000-row
# sample) and ForecastBench, and multiple choice on Polymarket (a seeded 2,000-row sample, with and without prices),
# Manifold without prices and FutureX.
# Three samples per question. A step whose output exists is skipped.
#   scripts/run_reasoning_baselines.sh GPUS MODEL TP     e.g. 0 openai/gpt-oss-20b 1
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
m=$2 tp=$3 t=$(basename $2 | cut -d@ -f1)
V=${PYTHON:-python} O=results/openforecaster L=results/baselines
C="--model $m --tp $tp --n 3 --gpu_mem 0.9"
log() { echo "[$(date +%m-%d\ %H:%M)] $t $*"; }
run() { local name=$1; shift; [ -f $O/${name}_shard0.parquet ] && { log "skip $name"; return; }
        log "$name"; $V "$@" --name $name > $L/$name.log 2>&1 || log "FAILED $name"; }
B="-m baselines.baseline_openforecaster" M="-m baselines.baseline_openforecaster_choice"  # modules: they import pipeline
run lf_manifold_yesno_noprice_$t $B $C --variant noprice --data data/manifold.parquet --split manifold
run lf_kalshi_yesno_noprice_$t   $B $C --variant noprice --data data/kalshi.parquet --split kalshi --rows 2000
run lf_forecastbench_noprice_$t  $B $C --variant noprice --data data/forecastbench.parquet --split forecastbench
run dist_choice_test2000_$t $M $C --variant price --protocol distribution --context state_context --rows 2000
run dist_blind_poly_$t      $M $C --variant noprice --protocol distribution --data data/mcexp/choice_blind.parquet \
                               --split test --context state_context --rows 2000
run dist_blind_manifold_$t  $M $C --variant noprice --protocol distribution --data data/mcexp/manifold_choice_blind.parquet \
                               --split manifold --context state_price
run dist_futurex_$t         $M $C --variant noprice --protocol distribution --data data/futurex_choice.parquet \
                               --split futurex --context state_context
log "all done"
