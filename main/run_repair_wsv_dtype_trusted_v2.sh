#!/usr/bin/env bash
set -euo pipefail

cd /research/cbim/vast/sf895/code/wsg/main

LOGDIR="${1:-../output/smplerx_wsv_inject_logs/repair_dtype_manual}"
mkdir -p "${LOGDIR}"

for shard in 0 1 2 3 4 5 6 7; do
  python repair_wsv_dtype_from_completion.py \
    --completion-tag trusted_v2 \
    --num-shards 8 \
    --shard-index "${shard}" \
    --dtype float16 \
    --compresslevel 3 \
    > "${LOGDIR}/shard_${shard}.log" 2>&1 &
done

wait
