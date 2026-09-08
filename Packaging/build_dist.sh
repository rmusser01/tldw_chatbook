#!/bin/bash
# Build a release distribution from committed source in an external scratch tree.

set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/tldw-chatbook-dist.XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT

if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=normal)" ]; then
    echo "ERROR: commit or remove task changes before building a release" >&2
    exit 1
fi

echo "Building committed tldw_chatbook source in $SCRATCH"
mkdir -p "$SCRATCH/source" "$SCRATCH/dist"
git -C "$ROOT" archive --format=tar HEAD | tar -xf - -C "$SCRATCH/source"

cd "$SCRATCH/source"
python -m build --outdir "$SCRATCH/dist"
python -m twine check "$SCRATCH/dist"/*
python Packaging/check_manifest.py "$SCRATCH/dist"

rm -rf "$ROOT/dist"
mkdir -p "$ROOT/dist"
cp "$SCRATCH/dist"/* "$ROOT/dist/"

echo "Distribution files created in $ROOT/dist"
ls -la "$ROOT/dist"
