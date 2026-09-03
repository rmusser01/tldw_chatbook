#!/usr/bin/env bash
# Build script for tldw_chatbook PyPI distribution

set -euo pipefail

PYTHON="${PYTHON:-python}"

echo "🚀 Building tldw_chatbook distribution..."

# Navigate to project root
cd "$(dirname "$0")/.."

# Clean Python artifacts
echo "🧹 Cleaning Python artifacts..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -o -name "*.pyo" -exec rm -f {} + 2>/dev/null || true
find . -name ".DS_Store" -exec rm -f {} + 2>/dev/null || true

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
echo "🔨 Building source and wheel distributions..."
"$PYTHON" -m build

# Check the distributions
echo "✅ Checking distributions with twine..."
"$PYTHON" -m twine check dist/*

# Verify manifest
echo "📋 Verifying distribution contents..."
"$PYTHON" Packaging/check_manifest.py

echo ""
echo "✨ Build complete!"
echo ""
echo "📦 Distribution files created in ./dist/"
ls -la dist/
echo ""
echo "📤 To upload to TestPyPI (for testing):"
echo "  $PYTHON -m twine upload --repository testpypi dist/*"
echo ""
echo "📤 To upload to PyPI (production):"
echo "  $PYTHON -m twine upload dist/*"
echo ""
echo "🧪 To test installation from wheel:"
echo "  pip install dist/tldw_chatbook-*.whl"
