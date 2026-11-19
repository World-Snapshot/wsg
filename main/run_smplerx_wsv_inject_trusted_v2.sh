#!/usr/bin/env bash
set -euo pipefail

source /research/cbim/vast/sf895/miniforge3/etc/profile.d/conda.sh
conda activate signx-1

cd /research/cbim/vast/sf895/code/wsg/main

LOGDIR="${1:-../output/smplerx_wsv_inject_logs/run_manual}"
mkdir -p "${LOGDIR}"

for shard in 0 1 2 3 4 5 6 7; do
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    export EGL_DEVICE_ID="${shard}"
    export PYOPENGL_PLATFORM=egl
    python pipeline02_inject_smplerx_into_wsg.py \
      --mode wsv \
      --datasets AVIS_train AVIS_val AVIS_test \
      --num-shards 8 \
      --shard-index "${shard}" \
      --completion-tag trusted_v2 \
      --clear-legacy-smplerx-vis \
      --compresslevel 3
  ) > "${LOGDIR}/shard_${shard}.log" 2>&1 &
done

wait
