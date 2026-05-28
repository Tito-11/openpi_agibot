#!/usr/bin/env bash
set -euo pipefail

ROBOT_HOST="${ROBOT_HOST:-10.42.1.101}"
ROBOT_USER="${ROBOT_USER:-agi}"
ROBOT_PASSWORD="${ROBOT_PASSWORD:-}"
ROBOT_TARGET="${ROBOT_USER}@${ROBOT_HOST}"

REMOTE_ROOT="${REMOTE_ROOT:-/data/ct}"
REMOTE_CLIENT="${REMOTE_ROOT}/vla_bridge_client_g2_left.py"
SERVER_PORT="${SERVER_PORT:-18080}"

SSH_BASE=(sshpass -p "${ROBOT_PASSWORD}" ssh -o StrictHostKeyChecking=no "${ROBOT_TARGET}")

if [[ -z "${ROBOT_PASSWORD}" ]]; then
  echo "ROBOT_PASSWORD is required. Export it before running this script." >&2
  exit 1
fi

echo "Stopping remote client..."
"${SSH_BASE[@]}" "pkill -f '${REMOTE_CLIENT}' || true"

echo "Stopping local inference server..."
pkill -f "agi_bot/vla_bridge_server.py --host 0.0.0.0 --port ${SERVER_PORT}" || true

echo "Stopped."
