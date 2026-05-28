#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/data/ct}"
CLIENT_SCRIPT="${CLIENT_SCRIPT:-${REMOTE_ROOT}/vla_bridge_client_g2_left.py}"
RESET_SCRIPT="${RESET_SCRIPT:-${REMOTE_ROOT}/reset_left_home.py}"
BRIDGE_LOG_PATH="${BRIDGE_LOG_PATH:-/tmp/vla_full_run.log}"
BRIDGE_LAST_JSON_PATH="${BRIDGE_LAST_JSON_PATH:-${REMOTE_ROOT}/vla_bridge_last.json}"
SERVER_URL="${SERVER_URL:?SERVER_URL is required}"

set +u
source /home/agi/app/env.sh /home/agi/app >/tmp/vla_bridge_env.log 2>&1
set -u

pkill -f "${CLIENT_SCRIPT}" || true

python3 "${RESET_SCRIPT}"

nohup python3 -u "${CLIENT_SCRIPT}" \
  --server-url "${SERVER_URL}" \
  --request-timeout-s 40 \
  --loop-hz 2 \
  --unsafe-raw-policy clip \
  --execute-translation-step-m 0.02 \
  --execute-rotation-step-deg 5 \
  --pose-stream-duration-s 0.5 \
  --pose-stream-rate-hz 20 \
  --warmup-frames 0 \
  --prediction-stable-frames 1 \
  --workspace-x-min 0.20 \
  --workspace-x-max 0.76 \
  --workspace-y-min 0.22 \
  --workspace-y-max 0.32 \
  --workspace-z-min 0.70 \
  --workspace-z-max 1.10 \
  --open-gripper-on-startup \
  --save-last-json "${BRIDGE_LAST_JSON_PATH}" \
  >"${BRIDGE_LOG_PATH}" 2>&1 < /dev/null &

echo "remote_client_started"
echo "log_path=${BRIDGE_LOG_PATH}"
echo "last_json=${BRIDGE_LAST_JSON_PATH}"
