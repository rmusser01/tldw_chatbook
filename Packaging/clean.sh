#!/bin/bash
# Clean script for tldw_chatbook - removes all build artifacts and caches

echo "🧹 Cleaning tldw_chatbook build artifacts..."

# Navigate to project root
cd "$(dirname "$0")/.."

# Count items before cleaning for summary
PYCACHE_COUNT=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
PYC_COUNT=$(find . -name "*.pyc" -o -name "*.pyo" 2>/dev/null | wc -l)
DS_STORE_COUNT=$(find . -name ".DS_Store" 2>/dev/null | wc -l)

# Clean Python cache directories
echo "📁 Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Clean compiled Python files
echo "🐍 Removing .pyc and .pyo files..."
find . -name "*.pyc" -o -name "*.pyo" -exec rm -f {} + 2>/dev/null || true

# Clean macOS metadata files
echo "🍎 Removing .DS_Store files..."
find . -name ".DS_Store" -exec rm -f {} + 2>/dev/null || true

# Clean build directories
echo "📦 Removing build directories..."
rm -rf dist/ build/ *.egg-info

# Clean pytest cache
echo "🧪 Removing pytest cache..."
rm -rf .pytest_cache/

# Clean coverage reports
echo "📊 Removing coverage reports..."
rm -rf htmlcov/ .coverage coverage.xml

# Clean Textual snapshots (if any)
echo "🖼️  Removing Textual snapshots..."
rm -rf tests/__snapshots__/ 2>/dev/null || true

# Summary
echo ""
echo "✨ Cleanup complete!"
echo ""
echo "📊 Removed:"
echo "  - $PYCACHE_COUNT __pycache__ directories"
echo "  - $PYC_COUNT .pyc/.pyo files"
echo "  - $DS_STORE_COUNT .DS_Store files"
echo "  - build/, dist/, and *.egg-info directories"
echo "  - pytest and coverage artifacts"
echo ""
echo "💡 Tip: Run this before building distributions or committing code"