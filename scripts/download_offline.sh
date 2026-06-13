#!/usr/bin/env bash
# Download everything needed to rebuild the CURRENT venv offline, into offline/.
#
# Usage:
#   scripts/download_offline.sh
#
#   Reproduces the exact package set of an existing venv (default: ./.venv,
#   override with $VENV) by freezing it and downloading every pinned
#   dependency — including packages installed beyond the declared extras
#   (e.g. gptqmodel, ninja). This guarantees the offline target can rebuild
#   the same environment, not just the pyproject extras.
#
# offline/ ends up with:
#   - requirements.lock        the frozen closure (exact versions)
#   - pip/setuptools/wheel/hatchling  (venv bootstrap + build backend)
#   - the project wheel itself
#   - every dependency wheel/sdist at the pinned version
#
# Wheels are fetched for THIS machine's platform and Python (linux x86_64,
# CPython 3.11) — the offline target must match.
#
# Rebuild later with: scripts/install_offline.sh <venv_dir>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/offline"
VENV="${VENV:-$ROOT/.venv}"
VPY="$VENV/bin/python"

if [ ! -x "$VPY" ]; then
    echo "error: no venv at $VENV — create and populate it first, or set \$VENV" >&2
    exit 1
fi

mkdir -p "$DEST"
LOCK="$DEST/requirements.lock"

echo "Freezing $VENV -> $LOCK"
# Exact closure, minus the editable project itself (shipped as a wheel below).
"$VPY" -m pip freeze --exclude-editable \
    | grep -viE '^(translation-benchmark|-e )' > "$LOCK"
echo "  $(wc -l < "$LOCK") pinned packages"

echo "Downloading into $DEST ..."
# Bootstrap tooling for the offline venv + the project's build backend.
"$VPY" -m pip download --dest "$DEST" --quiet pip setuptools wheel hatchling

# The project itself as a wheel, so the offline machine needs no source tree.
"$VPY" -m pip wheel --no-deps --wheel-dir "$DEST" --quiet "$ROOT"

# Every pinned dependency, AS WHEELS. pip wheel downloads a wheel where one
# exists and BUILDS one from sdist otherwise (e.g. pypcre, gptqmodel) using
# this machine's network for build deps — so the offline target installs
# only prebuilt wheels and never needs a compiler/cmake. --no-deps installs
# exactly the pinned closure without re-resolving (the live venv may carry
# benign metadata conflicts, e.g. protobuf vs opentelemetry, that a fresh
# resolve would reject).
"$VPY" -m pip wheel --wheel-dir "$DEST" --no-deps -r "$LOCK"

COUNT=$(find "$DEST" -maxdepth 1 -type f | wc -l)
SIZE=$(du -sh "$DEST" | cut -f1)
echo
echo "Done: $COUNT files, $SIZE in $DEST"
echo "Rebuild offline with: scripts/install_offline.sh .venv"
