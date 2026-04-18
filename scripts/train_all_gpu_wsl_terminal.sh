#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

RUN_NAME="${1:-gpu-terminal-$(date +%Y%m%d-%H%M%S)}"
MODEL="${MODEL:-all}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VERBOSE="${VERBOSE:-1}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-}"
VALIDATION_STEPS="${VALIDATION_STEPS:-}"

mkdir -p "runs/$RUN_NAME"

echo "run_name=$RUN_NAME"
echo "model=$MODEL"
echo "epochs=$EPOCHS"
echo "batch_size=$BATCH_SIZE"
echo "steps_per_epoch=${STEPS_PER_EPOCH:-auto}"
echo "validation_steps=${VALIDATION_STEPS:-auto}"
echo "output_dir=runs/$RUN_NAME"
echo

EXTRA_ARGS=()
if [[ -n "$STEPS_PER_EPOCH" ]]; then
  EXTRA_ARGS+=(--steps-per-epoch "$STEPS_PER_EPOCH")
fi
if [[ -n "$VALIDATION_STEPS" ]]; then
  EXTRA_ARGS+=(--validation-steps "$VALIDATION_STEPS")
fi

exec env \
  TF_CPP_MIN_LOG_LEVEL=1 \
  CUDA_VISIBLE_DEVICES=0 \
  .venv-wsl/bin/python -u -m deepvariant_train.train \
    --data-dir tfrecords \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --mixed-precision \
    --verbose "$VERBOSE" \
    --run-name "$RUN_NAME" \
    "${EXTRA_ARGS[@]}"
