#!/usr/bin/env bash
# Usage: ./run_client.sh /path/to/data [gpus]
# Examples:
#   ./run_client.sh ./data 0,1
#   ./run_client.sh ./data all
#   ./run_client.sh ./data none

set -Eeuo pipefail
trap 'echo "❌ Error on line $LINENO"; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="${1:-}"
GPUS="${2:-all}"

if [[ -z "$DATA_DIR" ]]; then
  echo "Usage: $0 /path/to/data [gpus]"
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Error: Data directory '$DATA_DIR' does not exist."
  exit 1
fi

# Resolve absolute path
if command -v realpath >/dev/null 2>&1; then
  DATA_DIR_ABS="$(realpath "$DATA_DIR")"
elif command -v readlink >/dev/null 2>&1; then
  DATA_DIR_ABS="$(readlink -f "$DATA_DIR" || python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$DATA_DIR")"
else
  DATA_DIR_ABS="$(python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$DATA_DIR")"
fi

export CURRENT_DIR="$SCRIPT_DIR"
export DATA_DIR="$DATA_DIR_ABS"
export GPU_DEVICES="$GPUS"

if podman compose version >/dev/null 2>&1; then
  COMPOSE=(podman compose)
elif podman-compose version >/dev/null 2>&1; then
  COMPOSE=(podman-compose)
else
  echo "❌ Podman Compose not found."
  exit 1
fi

echo "📂 Mounting data directory: $DATA_DIR_ABS"
echo "🖥️  Using GPUs: $GPU_DEVICES"
echo "🐳 Using Podman Compose"

echo "🔨 Building image..."
"${COMPOSE[@]}" build gmic_fl

echo "🚀 Starting container (detached)..."
"${COMPOSE[@]}" up -d gmic_fl

echo "✅ Container is up: gmic_fl_container"
echo "➡️  Dropping you into the container shell..."
echo

# exec into container
exec podman exec -it gmic_fl_container bash -lc '
  cd /workspace
  echo "Inside container."
  echo "Next:"
  echo "  1) bash /workspace/start_fl_client.sh    (if executable)"
  echo "  2) or: chmod +x /workspace/start_fl_client.sh && /workspace/start_fl_client.sh"
  echo "  3) attach: tmux attach -t fl_client"
  exec bash
'