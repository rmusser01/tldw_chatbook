#!/usr/bin/env bash
# Build committed source in an external scratch tree, preserving release options.

set -euo pipefail

PYTHON="${PYTHON:-python}"
DIST_DIR="${DIST_DIR:-dist}"
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd -P)"

DIST_DIR_REAL=$("$PYTHON" - "$DIST_DIR" "$REPO_ROOT" <<'PY'
from pathlib import Path
import sys

from Packaging.common.dist_path import resolve_dist_dir

try:
    print(resolve_dist_dir(sys.argv[1], Path(sys.argv[2])))
except ValueError as exc:
    raise SystemExit(f"Refusing unsafe DIST_DIR: {exc}")
PY
) || {
    echo "Use python -m build directly for external artifact directories." >&2
    exit 1
}

if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
    echo "ERROR: commit or remove task changes before building a release" >&2
    exit 1
fi

"$PYTHON" -c "import build, setuptools, twine, wheel" || {
    echo "Install release tools with: $PYTHON -m pip install 'setuptools>=77.0' build twine wheel" >&2
    exit 1
}

SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/tldw-chatbook-dist.XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/source" "$SCRATCH/dist"
git archive --format=tar HEAD | tar -xf - -C "$SCRATCH/source"
cd "$SCRATCH/source"

echo "Building committed tldw_chatbook source in $SCRATCH"
"$PYTHON" -m build --sdist --wheel --no-isolation --outdir "$SCRATCH/dist"
"$PYTHON" -m twine check "$SCRATCH/dist"/*
"$PYTHON" Packaging/check_manifest.py "$SCRATCH/dist"

rm -rf "$DIST_DIR_REAL"
mkdir -p "$DIST_DIR_REAL"
cp "$SCRATCH/dist"/* "$DIST_DIR_REAL/"
echo "Distribution files created in $DIST_DIR_REAL"
ls -la "$DIST_DIR_REAL"
