#!/usr/bin/env bash
# Build a fresh tldw_chatbook PyPI distribution.

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

echo "Building tldw_chatbook distribution into ${DIST_DIR_REAL}..."

"$PYTHON" -c "import build, setuptools, twine, wheel" || {
    echo "Install release tools with: $PYTHON -m pip install 'setuptools>=77.0' build twine wheel" >&2
    exit 1
}

rm -rf "$DIST_DIR_REAL" build ./*.egg-info
mkdir -p "$DIST_DIR_REAL"

echo "Building source and wheel distributions..."
"$PYTHON" -m build --sdist --wheel --no-isolation --outdir "$DIST_DIR_REAL"

echo "Checking package metadata..."
"$PYTHON" -m twine check "$DIST_DIR_REAL"/*

echo "Verifying distribution contents..."
"$PYTHON" Packaging/check_manifest.py "$DIST_DIR_REAL"

echo
echo "Build complete. Distribution files:"
ls -la "$DIST_DIR_REAL"
echo
echo "Installed-wheel regression:"
echo "  $PYTHON -m pytest Tests/Packaging/test_installed_distribution.py -m integration -q -p no:cacheprovider"
