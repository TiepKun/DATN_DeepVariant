#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

RUN_DIR="${RUN_DIR:-runs/gpu-terminal-progress}"
SPLIT="${SPLIT:-test}"
CHECKPOINT="${CHECKPOINT:-best}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "run_dir=$RUN_DIR"
echo "split=$SPLIT"
echo "checkpoint=$CHECKPOINT"
echo "batch_size=$BATCH_SIZE"
echo "progress_interval=$PROGRESS_INTERVAL"
echo

ARGS=(
  --run-dir "$RUN_DIR"
  --data-dir tfrecords
  --split "$SPLIT"
  --checkpoint "$CHECKPOINT"
  --batch-size "$BATCH_SIZE"
  --progress-interval "$PROGRESS_INTERVAL"
)

if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

exec env \
  TF_CPP_MIN_LOG_LEVEL=1 \
  CUDA_VISIBLE_DEVICES=0 \
  .venv-wsl/bin/python -u -m deepvariant_train.evaluate_and_plot "${ARGS[@]}"

