#!/usr/bin/env bash
# Create a Python 3.11 venv and rebuild the project + every dependency
# entirely from the local offline/ directory (no network access).
#
# Usage:
#   scripts/install_offline.sh [venv_dir]
#
#   venv_dir  where to create the venv [default: .venv]
#
# Reproduces the exact package set captured by scripts/download_offline.sh
# (offline/requirements.lock). Falls back to the project's default extras if
# no lockfile is present.
set -euo pipefail

VENV="${1:-.venv}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/offline"
LOCK="$SRC/requirements.lock"
PY="${PYTHON:-python3.11}"

if [ ! -d "$SRC" ]; then
    echo "error: $SRC not found — run scripts/download_offline.sh first" >&2
    exit 1
fi

echo "Creating venv at $VENV with $("$PY" --version)"
"$PY" -m venv "$VENV"

# Bootstrap pip from the offline directory.
"$VENV/bin/pip" install --quiet --no-index --find-links "$SRC" --upgrade pip setuptools wheel

if [ -f "$LOCK" ]; then
    echo "Installing $(wc -l < "$LOCK") pinned packages from $LOCK"
    # --no-deps: the lock is the full closure, so install exactly it without
    # re-resolving (mirrors the source venv, conflicts and all).
    "$VENV/bin/pip" install --no-index --find-links "$SRC" --no-deps -r "$LOCK"
    # The project itself (editable not possible offline; install the wheel).
    "$VENV/bin/pip" install --no-index --find-links "$SRC" --no-deps translation-benchmark
else
    echo "No lockfile; installing default extras"
    "$VENV/bin/pip" install --no-index --find-links "$SRC" "translation-benchmark[dev,models,ct2,vllm]"
fi

echo
"$VENV/bin/tb" --help >/dev/null && echo "ok: 'tb' CLI works"
echo "Activate with: source $VENV/bin/activate"
