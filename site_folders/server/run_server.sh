#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "❌ Error on line $LINENO"; exit 1' ERR

# Always run from the script directory (so compose finds docker-compose.yml)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Used by docker-compose.yml volumes
export CURRENT_DIR="$SCRIPT_DIR"

# Pick compose command
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif docker-compose version >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "❌ Docker Compose not found (need 'docker compose' or 'docker-compose')"
  exit 1
fi

# Optional: allow overriding service name (default matches compose)
SERVICE_NAME="${SERVICE_NAME:-gmic_fl}"

# Find container ID reliably (no hard-coded container_name needed)
get_container_id() {
  "${COMPOSE[@]}" ps -q "$SERVICE_NAME"
}

echo "🔨 Building Docker image..."
"${COMPOSE[@]}" build

echo "🚀 Starting container..."
"${COMPOSE[@]}" up -d

CID="$(get_container_id)"
if [[ -z "${CID}" ]]; then
  echo "❌ Could not find container id for service '${SERVICE_NAME}'."
  "${COMPOSE[@]}" ps
  exit 1
fi

echo "✅ Container started: ${CID}"

# If you are using tmux inside the container and want to attach, wait for it.
# Otherwise, just show logs.
if [[ "${1:-}" == "--attach-tmux" ]]; then
  SESSION="${TMUX_SESSION:-fl_server}"
  echo "🕒 Waiting for tmux session '${SESSION}' to exist inside container..."
  for i in {1..30}; do
    if docker exec "${CID}" tmux has-session -t "${SESSION}" >/dev/null 2>&1; then
      echo "✅ tmux session found. Attaching..."
      exec docker exec -it "${CID}" tmux attach -t "${SESSION}"
    fi
    sleep 1
  done
  echo "❌ tmux session '${SESSION}' never appeared. Showing last logs:"
  docker logs --tail=200 "${CID}" || true
  exit 1
else
  echo "ℹ️  Tip: run './run_server.sh --attach-tmux' only if you kept tmux in compose."
  echo "📌 Showing logs (Ctrl+C to stop):"
  # Option B: tee the live server console stream to a MOUNTED file so it survives a crash
  # AND the server run-dir deletion. mkdir -p guards the target (the tee would fail if the
  # parent dir is missing). This is the raw console backup alongside NVFLARE's log.txt
  # (Option A: NVFL_LOG_ROOT in docker-compose.yml).
  CONSOLE_LOG_DIR="${SCRIPT_DIR}/server_logs"
  mkdir -p "${CONSOLE_LOG_DIR}"
  exec docker logs -f "${CID}" 2>&1 | tee -a "${CONSOLE_LOG_DIR}/server_console.log"
fi