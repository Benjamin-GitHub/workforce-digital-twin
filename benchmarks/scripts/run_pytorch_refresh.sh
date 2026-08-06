#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/yolo-venv/bin/activate"

DATASET="${1:-}"
RUN="${2:-}"

if [[ ! "$RUN" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage:"
    echo "  $0 coco8 1"
    echo "  $0 coco128 1"
    echo "  $0 coco 1"
    exit 2
fi

case "$DATASET" in
    coco8)
        DATA_YAML="coco8.yaml"
        RUN_NAME="coco8_pt_r${RUN}"
        LOG_DIR="benchmarks/raw/refresh-2026/coco8"
        ;;
    coco128)
        DATA_YAML="coco128.yaml"
        RUN_NAME="coco128_pt_r${RUN}"
        LOG_DIR="benchmarks/raw/refresh-2026/coco128"
        ;;
    coco)
        DATA_YAML="coco.yaml"
        RUN_NAME="coco_full_pt_r${RUN}"
        LOG_DIR="benchmarks/raw/refresh-2026/coco"
        ;;
    *)
        echo "Dataset must be: coco8, coco128 or coco"
        exit 2
        ;;
esac

MODEL="models/pytorch/yolo26n.pt"
RESULTS_DIR="results/validation/refresh-2026"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

COMMON_ARGS=(
    task=detect
    split=val
    imgsz=320
    batch=1
    device=cpu
    workers=0
    rect=False
    conf=0.001
    iou=0.7
    max_det=300
    augment=False
    plots=False
    save_json=False
    project="$RESULTS_DIR"
)

LOG_FILE="$LOG_DIR/${RUN_NAME}.txt"

{
    echo "=========================================="
    echo "YOLO26n PyTorch refresh benchmark"
    echo "Dataset: $DATASET"
    echo "Dataset YAML: $DATA_YAML"
    echo "Run: $RUN"
    echo "Started: $(date -Is)"
    echo "Model: $MODEL"
    echo

    echo "Starting system condition:"
    vcgencmd measure_temp || true
    vcgencmd get_throttled || true
    vcgencmd measure_clock arm || true
    echo

    /usr/bin/time -v yolo val \
        model="$MODEL" \
        data="$DATA_YAML" \
        "${COMMON_ARGS[@]}" \
        name="$RUN_NAME"

    STATUS=$?

    echo
    echo "YOLO exit status: $STATUS"
    echo "Finished: $(date -Is)"
    echo "Finishing system condition:"
    vcgencmd measure_temp || true
    vcgencmd get_throttled || true
    vcgencmd measure_clock arm || true

    exit "$STATUS"
} 2>&1 | tee "$LOG_FILE"
