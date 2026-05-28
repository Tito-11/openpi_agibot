#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROBOT_HOST="${ROBOT_HOST:-10.42.1.101}"
ROBOT_USER="${ROBOT_USER:-agi}"
ROBOT_PASSWORD="${ROBOT_PASSWORD:-}"
ROBOT_TARGET="${ROBOT_USER}@${ROBOT_HOST}"

REMOTE_ROOT="${REMOTE_ROOT:-/data/ct}"
REMOTE_CLIENT="${REMOTE_ROOT}/vla_bridge_client_g2_left.py"
REMOTE_RESET="${REMOTE_ROOT}/reset_left_home.py"
REMOTE_LAUNCHER="${REMOTE_ROOT}/remote_run_full_inference.sh"
REMOTE_BRIDGE_LOG_PATH="${REMOTE_BRIDGE_LOG_PATH:-/tmp/vla_full_run.log}"
REMOTE_BRIDGE_LAST_JSON="${REMOTE_BRIDGE_LAST_JSON:-${REMOTE_ROOT}/vla_bridge_last.json}"

SERVER_HOST="${SERVER_HOST:-}"
SERVER_PORT="${SERVER_PORT:-18080}"
SERVER_LOG_DIR="${ROOT_DIR}/.runtime_logs"
SERVER_LOG_PATH="${SERVER_LOG_DIR}/vla_bridge_server.log"

mkdir -p "${SERVER_LOG_DIR}"

if [[ -z "${SERVER_HOST}" ]]; then
  SERVER_HOST="$(ip route get "${ROBOT_HOST}" | awk '/src/ {for (i = 1; i <= NF; ++i) if ($i == "src") {print $(i+1); exit}}')"
fi

if [[ -z "${SERVER_HOST}" ]]; then
  echo "Failed to detect local SERVER_HOST. Set SERVER_HOST manually and rerun." >&2
  exit 1
fi

SSH_BASE=(sshpass -p "${ROBOT_PASSWORD}" ssh -o StrictHostKeyChecking=no "${ROBOT_TARGET}")
SCP_BASE=(sshpass -p "${ROBOT_PASSWORD}" scp -o StrictHostKeyChecking=no)

if [[ -z "${ROBOT_PASSWORD}" ]]; then
  echo "ROBOT_PASSWORD is required. Export it before running this script." >&2
  exit 1
fi

echo "[1/4] Restart local inference server..."
pkill -f "agi_bot/vla_bridge_server.py --host 0.0.0.0 --port ${SERVER_PORT}" || true
(
  cd "${ROOT_DIR}"
  nohup uv run python agi_bot/vla_bridge_server.py --host 0.0.0.0 --port "${SERVER_PORT}" \
    >"${SERVER_LOG_PATH}" 2>&1 < /dev/null &
)

python - <<PY
import time
import urllib.request

url = "http://127.0.0.1:${SERVER_PORT}/health"
for _ in range(30):
    try:
        print(urllib.request.urlopen(url, timeout=2).read().decode())
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("health check failed")
PY

echo "[2/4] Sync bridge files to robot..."
"${SCP_BASE[@]}" "${ROOT_DIR}/agi_bot/vla_bridge_client_g2_left.py" "${ROBOT_TARGET}:${REMOTE_CLIENT}"
"${SCP_BASE[@]}" "${ROOT_DIR}/agi_bot/reset_left_home.py" "${ROBOT_TARGET}:${REMOTE_RESET}"
"${SCP_BASE[@]}" "${ROOT_DIR}/agi_bot/remote_run_full_inference.sh" "${ROBOT_TARGET}:${REMOTE_LAUNCHER}"

echo "[3/4] Start remote full inference (includes reset)..."
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}/infer"
"${SSH_BASE[@]}" "chmod +x '${REMOTE_LAUNCHER}' && SERVER_URL='${SERVER_URL}' REMOTE_ROOT='${REMOTE_ROOT}' BRIDGE_LOG_PATH='${REMOTE_BRIDGE_LOG_PATH}' BRIDGE_LAST_JSON_PATH='${REMOTE_BRIDGE_LAST_JSON}' bash '${REMOTE_LAUNCHER}'"

echo "[4/4] Done."
echo "Local server log: ${SERVER_LOG_PATH}"
echo "Remote client log: ssh ${ROBOT_TARGET} 'tail -f ${REMOTE_BRIDGE_LOG_PATH}'"
echo "Remote latest JSON: ssh ${ROBOT_TARGET} 'cat ${REMOTE_BRIDGE_LAST_JSON}'"
