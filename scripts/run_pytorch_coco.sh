#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$HOME/workforce-digital-twin"
LOG_DIR="$PROJECT_DIR/benchmarks/raw/coco"
RESULT_DIR="$PROJECT_DIR/results/validation"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/yolo-venv/bin/activate"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

RUN_NAME="pytorch_coco_run1"

yolo val \
    task=detect \
    model="$PROJECT_DIR/models/pytorch/yolo26n.pt" \
    data=coco.yaml \
    imgsz=320 \
    device=cpu \
    batch=1 \
    workers=2 \
    plots=False \
    project="$RESULT_DIR" \
    name="$RUN_NAME" \
    2>&1 | tee "$LOG_DIR/${RUN_NAME}.txt"
