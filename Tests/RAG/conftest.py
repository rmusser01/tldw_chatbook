# conftest.py for RAG tests
# Provides common fixtures and markers for RAG-related tests

import pytest
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Check if optional dependencies are available
EMBEDDINGS_AVAILABLE = DEPENDENCIES_AVAILABLE.get('embeddings_rag', False)
CHROMADB_AVAILABLE = DEPENDENCIES_AVAILABLE.get('chromadb', False)

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