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

# Now run pytest
result = subprocess.run([
    sys.executable, "-m", "pytest", 
    "Tests/RAG/", 
    "-v", 
    "--tb=short"
], capture_output=False)

sys.exit(result.returncode)