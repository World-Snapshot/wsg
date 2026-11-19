#!/usr/bin/env bash
set -euo pipefail

# Launch Pipeline 29 as deterministic tmux shards.
#
# Pipeline 29 is CPU/IO/compression bound: it rasterizes AVIS masks and rewrites
# compressed WSV archives. Unlike Pipeline 30, it does not run CUDA rendering, so
# GPUs are not bound here. Use NUM_WORKERS to consume more CPU and disk
# throughput; completion markers make the run resumable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSG_MAIN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIPELINE="${WSG_MAIN_DIR}/pipeline29_inject_avis_semantic_instance_into_wsv.py"

NUM_WORKERS="${NUM_WORKERS:-8}"
TMUX_PREFIX="${TMUX_PREFIX:-pipeline29_avis_195}"
SUBSETS="${SUBSETS:-train val test}"
COMPRESSION="${COMPRESSION:-deflated}"
COMPRESS_LEVEL="${COMPRESS_LEVEL:-6}"
PROGRESS_FRAMES="${PROGRESS_FRAMES:-5}"
PROGRESS_SECONDS="${PROGRESS_SECONDS:-30}"
STATE_DIR="${STATE_DIR:-../output/pipeline29_avis_semantic_state}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TMUX_CMD=(env -u LD_LIBRARY_PATH tmux)

if [[ "${CLEAR_207_255:-0}" == "1" ]]; then
  cat >&2 <<'EOF'
Refusing to launch with CLEAR_207_255=1 from this helper.
Channel 207-255 cleanup is destructive and should be run manually with
--clear-207-255 --yes-clear-207-255 after explicitly confirming the target set.
EOF
  exit 2
fi

echo "Pipeline 29 parallel launch"
echo "  workers:          ${NUM_WORKERS}"
echo "  tmux prefix:      ${TMUX_PREFIX}"
echo "  subsets:          ${SUBSETS}"
echo "  compression:      ${COMPRESSION} level ${COMPRESS_LEVEL}"
echo "  progress:         every ${PROGRESS_FRAMES} frames or ${PROGRESS_SECONDS}s"
echo "  state dir:        ${STATE_DIR}"
echo "  extra args:       ${EXTRA_ARGS:-<none>}"
echo

for shard in $(seq 0 $((NUM_WORKERS - 1))); do
  session="${TMUX_PREFIX}_s${shard}"
  if "${TMUX_CMD[@]}" has-session -t "${session}" 2>/dev/null; then
    echo "  skip existing tmux session: ${session}"
    continue
  fi

  cmd=$(
    printf 'cd %q && %q --num-shards %q --shard-index %q --subsets %s --compression %q --compress-level %q --progress-frames %q --progress-seconds %q --state-dir %q %s' \
      "${WSG_MAIN_DIR}" \
      "${PIPELINE}" \
      "${NUM_WORKERS}" \
      "${shard}" \
      "${SUBSETS}" \
      "${COMPRESSION}" \
      "${COMPRESS_LEVEL}" \
      "${PROGRESS_FRAMES}" \
      "${PROGRESS_SECONDS}" \
      "${STATE_DIR}" \
      "${EXTRA_ARGS}"
  )
  "${TMUX_CMD[@]}" new-session -d -s "${session}" "${cmd}"
  echo "  launched ${session}"
done

echo
echo "Monitor:"
echo "  env -u LD_LIBRARY_PATH tmux ls | grep ${TMUX_PREFIX}"
echo "  env -u LD_LIBRARY_PATH tmux capture-pane -t ${TMUX_PREFIX}_s0 -p -S -80"
echo "  ls -lh ${WSG_MAIN_DIR}/logs/pipeline29_*"
