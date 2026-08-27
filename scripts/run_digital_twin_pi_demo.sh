#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[pi-demo] Starting the full edge pipeline with live annotated video."
echo "[pi-demo] Press q in the video window or Ctrl+C in this terminal to stop."

export SHOW=1
exec "${SCRIPT_DIR}/run_digital_twin_pi.sh" "$@"
