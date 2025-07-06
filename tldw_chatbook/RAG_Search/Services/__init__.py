"""
RAG Search Services module.

Provides embeddings and other services for RAG functionality.
"""

from .embeddings_service import (
    EmbeddingsService,
    EmbeddingProvider,
    MockEmbeddingProvider,
    HuggingFaceProvider,
    OpenAIProvider,
    SentenceTransformerProvider
)

from .embeddings_compat import (
    EmbeddingFactoryCompat,
    EmbeddingConfig,
    create_legacy_factory,
    should_use_new_service
)

__all__ = [
    # Core service
    "EmbeddingsService",
    
    # Providers
    "EmbeddingProvider",
    "MockEmbeddingProvider", 
    "HuggingFaceProvider",
    "OpenAIProvider",
    "SentenceTransformerProvider",
    
    # Compatibility
    "EmbeddingFactoryCompat",
    "EmbeddingConfig",
    "create_legacy_factory",
    "should_use_new_service"
]