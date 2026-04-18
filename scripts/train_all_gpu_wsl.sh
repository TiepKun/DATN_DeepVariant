#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

RUN_NAME="${1:-gpu-all-$(date +%Y%m%d-%H%M%S)}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VERBOSE="${VERBOSE:-2}"

mkdir -p "runs/$RUN_NAME"
LOG_FILE="runs/$RUN_NAME/train.log"
PID_FILE="runs/$RUN_NAME/pid.txt"

nohup env \
  TF_CPP_MIN_LOG_LEVEL=1 \
  CUDA_VISIBLE_DEVICES=0 \
  .venv-wsl/bin/python -u -m deepvariant_train.train \
    --data-dir tfrecords \
    --model all \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --mixed-precision \
    --verbose "$VERBOSE" \
    --run-name "$RUN_NAME" \
  > "$LOG_FILE" 2>&1 < /dev/null &

PID="$!"
echo "$PID" > "$PID_FILE"
echo "run_name=$RUN_NAME"
echo "pid=$PID"
echo "log=$LOG_FILE"
