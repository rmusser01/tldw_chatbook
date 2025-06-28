#!/bin/bash
# Build script for tldw_chatbook PyPI release

set -e  # Exit on error

echo "🚀 Building tldw_chatbook for PyPI release..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this script from the project root."
    exit 1
fi

# Check for required tools
for tool in python pip build twine; do
    if ! command -v $tool &> /dev/null; then
        echo "❌ Error: $tool is not installed."
        echo "Install with: pip install build twine"
        exit 1
    fi
done

# Get version from pyproject.toml
VERSION=$(python -c "
try:
    import tomllib
except ImportError:
    import tomli as tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
    print(data['project']['version'])
")
echo "📦 Building version: $VERSION"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build the distribution
echo "🔨 Building source distribution and wheel..."
python -m build

# Check the distributions
echo "✅ Checking distributions..."
twine check dist/*

echo "📁 Built files:"
ls -la dist/

echo ""
echo "✨ Build complete! Next steps:"
echo "1. Test the package: pip install dist/tldw_chatbook-${VERSION}-py3-none-any.whl"
echo "2. Upload to TestPyPI: twine upload --repository testpypi dist/*"
echo "3. Upload to PyPI: twine upload dist/*"
echo ""
echo "📚 See PYPI_RELEASE.md for detailed instructions."