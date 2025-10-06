#!/usr/bin/env bash
set -euo pipefail


BASE_DIR="${1%/}"  # remove trailing slash
SCENE_NAME=$(basename "$BASE_DIR")
ALL_DIR="$BASE_DIR/${SCENE_NAME}-All"
RESULT_DIR="$ALL_DIR/WILDGS"
LOG_FILE="$RESULT_DIR/wildgs_log.txt"

mkdir -p "$RESULT_DIR"

# Redirect both stdout and stderr to screen AND log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==> Running for scene: $SCENE_NAME"
echo "==> Using ALL_DIR: $ALL_DIR"
echo "==> Results will be saved in: $RESULT_DIR"
echo "==> Log file: $LOG_FILE"

# 1) training and evaluation
python -u wildgaussians/train.py \
  --data "$ALL_DIR" \
  --output "$RESULT_DIR" \
  --dataset-type nerfonthego \

echo "==> Done."
