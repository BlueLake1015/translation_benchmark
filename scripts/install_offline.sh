#!/usr/bin/env bash
# Create a Python 3.11 venv and install the project + dependencies entirely
# from the local offline/ directory (no network access).
#
# Usage:
#   scripts/install_offline.sh [venv_dir] [extras]
#
#   venv_dir  where to create the venv [default: .venv]
#   extras    comma-separated extras, must match what was downloaded with
#             scripts/download_offline.sh [default: dev,models,ct2,vllm]
set -euo pipefail

VENV="${1:-.venv}"
EXTRAS="${2:-dev,models,ct2,vllm}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/offline"
PY="${PYTHON:-python3.11}"

if [ ! -d "$SRC" ]; then
    echo "error: $SRC not found — run scripts/download_offline.sh first" >&2
    exit 1
fi

echo "Creating venv at $VENV with $("$PY" --version)"
"$PY" -m venv "$VENV"

# Bootstrap pip from the offline directory, then install everything from it.
"$VENV/bin/pip" install --quiet --no-index --find-links "$SRC" --upgrade pip setuptools wheel
"$VENV/bin/pip" install --no-index --find-links "$SRC" "translation-benchmark[$EXTRAS]"

echo
"$VENV/bin/tb" --help >/dev/null && echo "ok: 'tb' CLI works"
echo "Activate with: source $VENV/bin/activate"
