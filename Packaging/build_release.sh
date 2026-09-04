#!/usr/bin/env bash
# Release build wrapper for tldw_chatbook PyPI artifacts.

set -euo pipefail

PYTHON="${PYTHON:-python}"

if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Run this script from the project root." >&2
    exit 1
fi

if ! "$PYTHON" -c "import build, setuptools, twine, wheel" >/dev/null 2>&1; then
    echo "Error: release build modules are not installed." >&2
    echo "Install release tools with: $PYTHON -m pip install 'setuptools>=77.0' build twine wheel" >&2
    exit 1
fi

VERSION=$("$PYTHON" - <<'PY'
from pathlib import Path
import tomllib

with Path("pyproject.toml").open("rb") as stream:
    print(tomllib.load(stream)["project"]["version"])
PY
)

echo "Building tldw_chatbook ${VERSION} for PyPI release..."
PYTHON="$PYTHON" Packaging/build_dist.sh

echo
echo "Next steps:"
echo "1. Smoke-test the built wheel in a disposable environment."
echo "2. Use the publish-pypi GitHub Actions workflow for TestPyPI."
echo "3. Merge the approved release commit to protected main for production PyPI."
echo
echo "See Packaging/PYPI_RELEASE.md for detailed instructions."
