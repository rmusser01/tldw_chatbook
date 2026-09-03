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

VERSION=$("$PYTHON" -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")

echo "Building tldw_chatbook ${VERSION} for PyPI release..."
PYTHON="$PYTHON" Packaging/build_dist.sh

echo
echo "Next steps:"
echo "1. Run the installed-wheel regression printed above."
echo "2. Use the publish-pypi GitHub Actions workflow for TestPyPI."
echo "3. Publish production PyPI from a protected v${VERSION} tag."
echo
echo "See Packaging/PYPI_RELEASE.md for detailed instructions."
