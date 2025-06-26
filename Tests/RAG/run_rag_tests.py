#!/usr/bin/env python
"""Script to run RAG tests with proper dependency detection"""

import sys
import subprocess

# First, verify dependencies are installed
try:
    import torch
    import transformers
    import numpy
    import chromadb
    print("✅ All RAG dependencies found")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

# Monkey patch the conftest before importing pytest
import importlib.util
spec = importlib.util.spec_from_file_location("conftest", "Tests/RAG/conftest.py")
conftest = importlib.util.module_from_spec(spec)

# Override the dependency checks
conftest.EMBEDDINGS_AVAILABLE = True
conftest.CHROMADB_AVAILABLE = True

# Update the markers
import pytest
conftest.pytest.mark.requires_embeddings = pytest.mark.skipif(False, reason="")
conftest.pytest.mark.requires_chromadb = pytest.mark.skipif(False, reason="")
conftest.pytest.mark.requires_rag_deps = pytest.mark.skipif(False, reason="")

sys.modules['Tests.RAG.conftest'] = conftest
spec.loader.exec_module(conftest)

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description='Run RAG tests with proper dependencies')
parser.add_argument('-t', '--timeout', type=int, default=60,
                    help='Timeout for each test in seconds (default: 60)')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Verbose output')
args = parser.parse_args()

# Build pytest command
pytest_cmd = [
    sys.executable, "-m", "pytest", 
    "Tests/RAG/",
    "--tb=short",
    f"--timeout={args.timeout}"
]

if args.verbose:
    pytest_cmd.append("-v")

# Now run pytest
result = subprocess.run(pytest_cmd, capture_output=False)

sys.exit(result.returncode)