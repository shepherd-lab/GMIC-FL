#!/usr/bin/env bash
set -Eeuo pipefail

# Set SITE_NAME to THIS client's name/org from your provisioning project (gmic-localhost.yml).
# It must match the client participant's name/org, or the FL server will reject the connection.
SITE_NAME="${SITE_NAME:-YOUR_SITE_NAME}"

SESSION="fl_client"
PIDFILE="/tmp/fl_client.pid"

is_pid_alive() {
  [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

echo "[start_fl_client] checking existing state..."

if tmux has-session -t "$SESSION" 2>/dev/null && is_pid_alive; then
  echo "[start_fl_client] NVFLARE already running (pid $(cat "$PIDFILE")). Attaching..."
  exec tmux attach -t "$SESSION"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[start_fl_client] tmux session exists but NVFLARE is not running."
  echo "Options:"
  echo "  [a] Attach anyway"
  echo "  [r] Restart NVFLARE in this session"
  echo "  [k] Kill session and start fresh"
  read -rp "Choice [a/r/k]: " choice

  case "$choice" in
    a|A)
      exec tmux attach -t "$SESSION"
      ;;
    r|R)
      tmux kill-session -t "$SESSION"
      ;;
    k|K)
      tmux kill-session -t "$SESSION"
      ;;
    *)
      echo "Invalid choice."
      exit 1
      ;;
  esac
fi

echo "[start_fl_client] starting fresh tmux session and NVFLARE..."

tmux new-session -d -s "$SESSION" bash -lc "
  set -e
  echo \"[fl_client] starting NVFLARE client...\"
  python -u -m nvflare.private.fed.app.client.client_train \
    -m /workspace \
    -s fed_client.json \
    --set uid=$SITE_NAME secure_train=true config_folder=config org=$SITE_NAME &
  pid=\$!
  echo \$pid > $PIDFILE

  wait \$pid
  ret=\$?

  echo
  echo \"[fl_client] client exited with code \$ret\"
  rm -f $PIDFILE
  echo \"[fl_client] Press Enter to keep shell open.\"
  read _
  exec bash
"

tmux set-option -g remain-on-exit on || true

echo "[start_fl_client] started. Attaching..."
exec tmux attach -t "$SESSION"