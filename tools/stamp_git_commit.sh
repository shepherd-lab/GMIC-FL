#!/usr/bin/env bash
# Stamp the current git commit into each job's app/custom/GIT_COMMIT so DEPLOYED NVFLARE
# runs are attributable in the [run-config] log line. NVFLARE copies custom/ WITHOUT its
# .git, so `git` inside the deployed job returns nothing (that's why runs logged
# git=unknown). _git_hash() reads this GIT_COMMIT file as a fallback; the file travels with
# the job to every client.
#
# Run this BEFORE submitting a job (after `git pull`), e.g.:
#   ./tools/stamp_git_commit.sh
# Then submit <job> as usual.
#
# The stamp is HEAD's short hash plus a `-dirty` suffix if the working tree has uncommitted
# changes, so the log can't claim a clean commit when the deployed code was actually edited.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HASH="$(git -C "$ROOT" rev-parse --short HEAD)"
DIRTY=""
git -C "$ROOT" diff --quiet || DIRTY="-dirty"
STAMP="${HASH}${DIRTY}"

# Stamp the shared custom/ (copied into a job's app/custom before running); the copy carries it.
cdir="$ROOT/custom"
if [ -d "$cdir" ]; then
  echo "$STAMP" > "$cdir/GIT_COMMIT"
  echo "stamped custom/GIT_COMMIT = $STAMP"
fi
echo "Done. Copy custom/ into your job and submit; clients will log git=$STAMP."
