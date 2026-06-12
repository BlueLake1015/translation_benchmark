#!/usr/bin/env bash
# Download every package needed for an offline install into offline/.
#
# Usage:
#   scripts/download_offline.sh [extras]
#
#   extras  comma-separated project extras [default: dev,models,ct2,vllm]
#           e.g. "dev" for a CPU-only test setup, add "comet" for the
#           neural metric. Heads-up: the default includes torch + vLLM
#           (CUDA wheels), ~10 GB of downloads.
#
# The offline/ directory will contain:
#   - pip/setuptools/wheel (to bootstrap the venv offline)
#   - the project wheel itself
#   - all dependency wheels for the selected extras
#
# Wheels are fetched for THIS machine's platform and Python (linux x86_64,
# CPython 3.11) — the offline target must match.
#
# Install later with: scripts/install_offline.sh <venv_dir> [extras]
set -euo pipefail

EXTRAS="${1:-dev,models,ct2,vllm}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/offline"
PY="${PYTHON:-python3.11}"

# Some distro pythons (e.g. deadsnakes) ship without pip — bootstrap it.
if ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "pip missing for $PY — bootstrapping via ensurepip"
    "$PY" -m ensurepip --upgrade --user 2>/dev/null || "$PY" -m ensurepip --upgrade
fi

mkdir -p "$DEST"
echo "Downloading into $DEST (extras: $EXTRAS)"

# Bootstrap tooling for the offline venv + the project's build backend.
"$PY" -m pip download --dest "$DEST" --quiet pip setuptools wheel hatchling

# The project itself as a wheel, so the offline machine needs no source tree.
"$PY" -m pip wheel --no-deps --wheel-dir "$DEST" --quiet "$ROOT"

# All dependencies for the selected extras.
"$PY" -m pip download --dest "$DEST" "$ROOT[$EXTRAS]"

COUNT=$(ls "$DEST" | wc -l)
SIZE=$(du -sh "$DEST" | cut -f1)
echo
echo "Done: $COUNT files, $SIZE in $DEST"
echo "Install offline with: scripts/install_offline.sh .venv $EXTRAS"
