# conftest.py for RAG tests
# Provides common fixtures and markers for RAG-related tests

import pytest

# Reset and re-initialize dependency checks before importing anything else
from tldw_chatbook.Utils.optional_deps import reset_dependency_checks, initialize_dependency_checks, DEPENDENCIES_AVAILABLE
reset_dependency_checks()
initialize_dependency_checks()

# Now check the availability
EMBEDDINGS_AVAILABLE = DEPENDENCIES_AVAILABLE.get('embeddings_rag', False)
CHROMADB_AVAILABLE = DEPENDENCIES_AVAILABLE.get('chromadb', False)

# If still not detected but we know they're installed, force enable
if not EMBEDDINGS_AVAILABLE:
    try:
        import torch
        import transformers
        import numpy
        import chromadb
        import sentence_transformers
        EMBEDDINGS_AVAILABLE = True
        CHROMADB_AVAILABLE = True
        # Update the global state
        DEPENDENCIES_AVAILABLE['embeddings_rag'] = True
        DEPENDENCIES_AVAILABLE['chromadb'] = True
        DEPENDENCIES_AVAILABLE['torch'] = True
        DEPENDENCIES_AVAILABLE['transformers'] = True
        DEPENDENCIES_AVAILABLE['numpy'] = True
        DEPENDENCIES_AVAILABLE['sentence_transformers'] = True
    except ImportError:
        pass

# Define custom markers
pytest.mark.requires_embeddings = pytest.mark.skipif(
    not EMBEDDINGS_AVAILABLE,
    reason="Embeddings dependencies (sentence-transformers) not installed. Install with: pip install -e '.[embeddings_rag]'"
)

pytest.mark.requires_chromadb = pytest.mark.skipif(
    not CHROMADB_AVAILABLE,
    reason="ChromaDB not installed. Install with: pip install -e '.[embeddings_rag]'"
)

pytest.mark.requires_rag_deps = pytest.mark.skipif(
    not (EMBEDDINGS_AVAILABLE and CHROMADB_AVAILABLE),
    reason="RAG dependencies not installed. Install with: pip install -e '.[embeddings_rag]'"
)

# Register markers with pytest
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "requires_embeddings: mark test as requiring embeddings dependencies"
    )
    config.addinivalue_line(
        "markers", "requires_chromadb: mark test as requiring ChromaDB"
    )
    config.addinivalue_line(
        "markers", "requires_rag_deps: mark test as requiring all RAG dependencies"
    )