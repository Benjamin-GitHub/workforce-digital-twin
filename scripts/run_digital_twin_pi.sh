#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  : # Keep the explicit caller-provided interpreter.
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi
STREAM_URL="${STREAM_URL:-http://192.168.1.171:81/stream}"
POSE_MODEL="${POSE_MODEL:-${REPO_ROOT}/models/ncnn/yolo26n-pose_ncnn_model}"
PPE_MODEL="${PPE_MODEL:-${REPO_ROOT}/models/ncnn/yolo26n_ppe_best_ncnn_model}"
DIGITAL_TWIN_API_URL="${DIGITAL_TWIN_API_URL:-http://192.168.1.252:8000}"
ACTIVITY_CONFIG="${ACTIVITY_CONFIG:-${REPO_ROOT}/config/activity.yaml}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-${REPO_ROOT}/results/activity/live_pose_ppe.mp4}"
OUTPUT_CSV="${OUTPUT_CSV:-${REPO_ROOT}/results/activity/live_pose_ppe.csv}"
WORKER_ID="${WORKER_ID:-worker01}"
CAMERA_ID="${CAMERA_ID:-esp32_cam_01}"
PUBLISH_INTERVAL="${PUBLISH_INTERVAL:-0.1}"
ENABLE_PPE="${ENABLE_PPE:-1}"
PPE_CONF="${PPE_CONF:-0.25}"
PPE_IMGSZ="${PPE_IMGSZ:-320}"
PPE_STRIDE="${PPE_STRIDE:-3}"
SHOW="${SHOW:-0}"

repo_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "$1" ;;
  esac
}

PYTHON_BIN="$(repo_path "${PYTHON_BIN}")"
POSE_MODEL="$(repo_path "${POSE_MODEL}")"
PPE_MODEL="$(repo_path "${PPE_MODEL}")"
ACTIVITY_CONFIG="$(repo_path "${ACTIVITY_CONFIG}")"
OUTPUT_VIDEO="$(repo_path "${OUTPUT_VIDEO}")"
OUTPUT_CSV="$(repo_path "${OUTPUT_CSV}")"

usage() {
  cat <<'EOF'
Start the Raspberry Pi edge-vision pipeline and publish worker state to the
Digital Twin API running on the Mac (or another central node).

Usage:
  ./scripts/run_digital_twin_pi.sh

Common overrides:
  STREAM_URL=...       ESP32 stream URL
  POSE_MODEL=...       YOLO pose NCNN model directory
  PPE_MODEL=...        PPE NCNN model directory
  DIGITAL_TWIN_API_URL=...  Central-node API URL
  ENABLE_PPE=0         Disable the separate PPE detector
  SHOW=1               Show the local OpenCV preview
  PYTHON_BIN=...       Python interpreter (defaults to .venv/bin/python)

All paths may be absolute or relative to the repository root. Extra arguments
are passed through to src.vision.pose_activity after the launcher defaults.
EOF
}

fail() {
  echo "[pi-launcher] Error: $*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ -x "${PYTHON_BIN}" ]] || fail "Python interpreter not found or not executable: ${PYTHON_BIN}"
[[ -e "${POSE_MODEL}" ]] || fail "Pose model not found: ${POSE_MODEL}"
[[ -f "${ACTIVITY_CONFIG}" ]] || fail "Activity config not found: ${ACTIVITY_CONFIG}"

case "${ENABLE_PPE}" in
  0|1) ;;
  *) fail "ENABLE_PPE must be 0 or 1" ;;
esac

case "${SHOW}" in
  0|1) ;;
  *) fail "SHOW must be 0 or 1" ;;
esac

if [[ "${ENABLE_PPE}" -eq 1 ]]; then
  [[ -e "${PPE_MODEL}" ]] || fail "PPE model not found: ${PPE_MODEL}"
fi

mkdir -p "$(dirname "${OUTPUT_VIDEO}")" "$(dirname "${OUTPUT_CSV}")"

args=(
  -m src.vision.pose_activity
  --model "${POSE_MODEL}"
  --source "${STREAM_URL}"
  --config "${ACTIVITY_CONFIG}"
  --output "${OUTPUT_VIDEO}"
  --csv "${OUTPUT_CSV}"
  --publish-digital-twin
  --digital-twin-api-url "${DIGITAL_TWIN_API_URL}"
  --digital-twin-publish-interval "${PUBLISH_INTERVAL}"
  --worker-id "${WORKER_ID}"
  --camera-id "${CAMERA_ID}"
)

if [[ "${ENABLE_PPE}" -eq 1 ]]; then
  args+=(
    --enable-ppe
    --ppe-model "${PPE_MODEL}"
    --ppe-conf "${PPE_CONF}"
    --ppe-imgsz "${PPE_IMGSZ}"
    --ppe-stride "${PPE_STRIDE}"
  )
fi

if [[ "${SHOW}" -eq 1 ]]; then
  args+=(--show)
fi

if [[ "$#" -gt 0 ]]; then
  args+=("$@")
fi

echo "[pi-launcher] Repository: ${REPO_ROOT}"
echo "[pi-launcher] ESP32 stream: ${STREAM_URL}"
echo "[pi-launcher] Digital Twin API: ${DIGITAL_TWIN_API_URL}"
echo "[pi-launcher] Pose model: ${POSE_MODEL}"
if [[ "${ENABLE_PPE}" -eq 1 ]]; then
  echo "[pi-launcher] PPE model: ${PPE_MODEL}"
else
  echo "[pi-launcher] PPE: disabled"
fi
echo "[pi-launcher] Press Ctrl+C to stop."

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${args[@]}"
