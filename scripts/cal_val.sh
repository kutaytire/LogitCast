#!/bin/bash
# Score one unified model (INIT, SEED) on the ForecastBench validation set (logitcast/calibrate_unified.py).  cal_val.sh GPU INIT SEED
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$1
N=uni_$2_s$3 P=${PYTHON:-python} L=results/unified
for v in "informed state_json_price" "blind state"; do set -- $v
  [ -f results/${N}_$1_fb_val.parquet ] || $P -m logitcast.eval_forecast --data data/forecastbench_val.parquet --split fb_val \
    --contexts $2 --run_dir runs/$N --temp_tag $1 --name ${N}_$1_fb > $L/${N}_$1_fb_val.log 2>&1 || echo "FAILED $N $1"
done
echo "done $N"
