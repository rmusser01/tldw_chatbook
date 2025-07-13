# Embeddings module - compatibility shim for tests
"""
This module provides backward compatibility for tests expecting the old Embeddings module structure.
The actual implementation has been moved to RAG_Search.simplified.
"""

# Import from the actual Embeddings_Lib module in this directory
from .Embeddings_Lib import EmbeddingFactory, EmbeddingConfigSchema

# Create compatibility alias for tests expecting EmbeddingFactoryCompat
EmbeddingFactoryCompat = EmbeddingFactory

# Re-export main classes for backward compatibility
__all__ = [
    'EmbeddingFactory', 
    'EmbeddingFactoryCompat',  # Alias for backward compatibility
    'EmbeddingConfigSchema'
]

# Note: InMemoryVectorStore and ChromaVectorStore have been moved to RAG_Search.simplified
# to avoid circular dependencies. Import them directly from there if needed.