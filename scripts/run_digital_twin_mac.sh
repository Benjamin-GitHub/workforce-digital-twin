#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_DIR="${REPO_ROOT}/apps/digital-twin/backend"
FRONTEND_DIR="${REPO_ROOT}/apps/digital-twin/frontend"
UVICORN="${REPO_ROOT}/.venv/bin/uvicorn"
SECRETS_FILE="${REPO_ROOT}/secret.h"

MQTT_PID=""
MQTT_OWNED=0
BACKEND_PID=""
FRONTEND_PID=""
CLEANING_UP=0

fail() {
  echo "[launcher] Error: $*" >&2
  exit 1
}

cleanup() {
  local exit_code="${1:-0}"

  if [[ "${CLEANING_UP}" -eq 1 ]]; then
    return
  fi
  CLEANING_UP=1

  echo
  echo "[launcher] Stopping Digital Twin services..."
  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
  if [[ "${MQTT_OWNED}" -eq 1 ]] && [[ -n "${MQTT_PID}" ]] && kill -0 "${MQTT_PID}" 2>/dev/null; then
    kill "${MQTT_PID}" 2>/dev/null || true
  fi

  [[ -n "${FRONTEND_PID}" ]] && wait "${FRONTEND_PID}" 2>/dev/null || true
  [[ -n "${BACKEND_PID}" ]] && wait "${BACKEND_PID}" 2>/dev/null || true
  if [[ "${MQTT_OWNED}" -eq 1 ]] && [[ -n "${MQTT_PID}" ]]; then
    wait "${MQTT_PID}" 2>/dev/null || true
  fi
  echo "[launcher] Managed services stopped."
  exit "${exit_code}"
}

trap 'cleanup 130' INT TERM

if [[ -f "${SECRETS_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${SECRETS_FILE}"
  set +a
  echo "[launcher] Loaded MQTT credentials from ${SECRETS_FILE}"
else
  echo "[launcher] No secret.h found; continuing with environment/default MQTT settings."
fi

[[ -d "${BACKEND_DIR}/app" ]] || fail "backend directory not found at ${BACKEND_DIR}"
[[ -x "${UVICORN}" ]] || fail "${UVICORN} is missing. Create/install the repo root .venv first."
[[ -f "${FRONTEND_DIR}/package.json" ]] || fail "frontend package.json not found at ${FRONTEND_DIR}"
command -v pnpm >/dev/null 2>&1 || fail "pnpm is not installed or not on PATH."
[[ -x "${FRONTEND_DIR}/node_modules/.bin/vinext" ]] || fail "frontend dependencies are missing. Run: pnpm --dir apps/digital-twin/frontend install"

MQTT_PID="$(lsof -tiTCP:1883 -sTCP:LISTEN 2>/dev/null | head -n 1)"
if [[ -n "${MQTT_PID}" ]]; then
  echo "[launcher] Reusing MQTT broker at mqtt://0.0.0.0:1883 (PID ${MQTT_PID})"
else
  command -v mosquitto >/dev/null 2>&1 || fail "Mosquitto is not installed. Run: brew install mosquitto"
  if [[ -f "/opt/homebrew/etc/mosquitto/mosquitto.conf" ]]; then
    MQTT_CONFIG="/opt/homebrew/etc/mosquitto/mosquitto.conf"
  elif [[ -f "/usr/local/etc/mosquitto/mosquitto.conf" ]]; then
    MQTT_CONFIG="/usr/local/etc/mosquitto/mosquitto.conf"
  else
    fail "Mosquitto configuration not found under /opt/homebrew/etc or /usr/local/etc."
  fi

  echo "[launcher] Starting MQTT broker at mqtt://0.0.0.0:1883"
  mosquitto -c "${MQTT_CONFIG}" &
  MQTT_PID=$!
  MQTT_OWNED=1
  echo "[launcher] MQTT PID: ${MQTT_PID}"
fi

echo "[launcher] Starting backend at http://0.0.0.0:8000"
(
  cd "${REPO_ROOT}" || exit 1
  exec "${UVICORN}" app.main:app \
    --app-dir "${BACKEND_DIR}" \
    --host 0.0.0.0 \
    --port 8000
) &
BACKEND_PID=$!
echo "[launcher] Backend PID: ${BACKEND_PID}"

echo "[launcher] Starting frontend at http://0.0.0.0:3000"
(
  cd "${FRONTEND_DIR}" || exit 1
  exec pnpm run dev --hostname 0.0.0.0 --port 3000
) &
FRONTEND_PID=$!
echo "[launcher] Frontend PID: ${FRONTEND_PID}"
echo "[launcher] Open http://localhost:3000 (API: http://localhost:8000)"
echo "[launcher] Press Ctrl+C to stop both services."

# Bash 3.2 ships with macOS and has no `wait -n`, so supervise both children
# portably. If either exits, terminate the other and return an error.
while true; do
  if [[ "${MQTT_OWNED}" -eq 1 ]] && ! kill -0 "${MQTT_PID}" 2>/dev/null; then
    wait "${MQTT_PID}"
    status=$?
    echo "[launcher] MQTT broker exited unexpectedly (status ${status})." >&2
    cleanup 1
  fi
  if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    wait "${BACKEND_PID}"
    status=$?
    echo "[launcher] Backend exited unexpectedly (status ${status})." >&2
    cleanup 1
  fi
  if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    wait "${FRONTEND_PID}"
    status=$?
    echo "[launcher] Frontend exited unexpectedly (status ${status})." >&2
    cleanup 1
  fi
  sleep 1
done
