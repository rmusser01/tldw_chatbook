#!/bin/bash
# Build script for tldw_chatbook PyPI distribution

echo "Building tldw_chatbook distribution..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
echo "Building source and wheel distributions..."
python -m build

# Check the distributions
echo "Checking distributions with twine..."
python -m twine check dist/*

echo "Build complete!"
echo ""
echo "Distribution files created in ./dist/"
echo ""
echo "To upload to TestPyPI (for testing):"
echo "  python -m twine upload --repository testpypi dist/*"
echo ""
echo "To upload to PyPI (production):"
echo "  python -m twine upload dist/*"
echo ""
echo "To test installation from wheel:"
echo "  pip install dist/tldw_chatbook-*.whl"