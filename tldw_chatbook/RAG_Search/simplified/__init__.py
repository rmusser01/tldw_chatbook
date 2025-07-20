"""
Simplified RAG services with citations support.

This module provides a cleaner, more maintainable implementation of RAG services
while reusing existing robust components like Embeddings_Lib.
"""

from .citations import (
    Citation,
    CitationType,
    SearchResultWithCitations,
    merge_citations,
    group_citations_by_document,
    filter_overlapping_citations
)

from .embeddings_wrapper import (
    EmbeddingsServiceWrapper,
    EmbeddingsServiceWrapper as EmbeddingsService,  # Alias for backward compatibility
    create_embeddings_service
)

from .vector_store import (
    SearchResult,
    VectorStore,
    ChromaVectorStore,
    InMemoryVectorStore,
    create_vector_store
)

from .rag_service import (
    RAGService,
    RAGConfig,
    create_and_index,
    create_rag_service
)

from .enhanced_rag_service import (
    EnhancedRAGService,
    create_enhanced_rag_service
)

from .enhanced_rag_service_v2 import (
    EnhancedRAGServiceV2
)

from .data_models import (
    IndexingResult
)

from .simple_cache import (
    SimpleRAGCache,
    get_rag_cache
)

from .config import (
    EmbeddingConfig,
    VectorStoreConfig,
    ChunkingConfig,
    SearchConfig,
    create_config_for_collection,
    create_config_for_testing
)

from .rag_factory import (
    create_rag_service,
    create_rag_service_from_config,
    create_auto_rag_service,
    get_available_profiles
)

__all__ = [
    # Citations
    "Citation",
    "CitationType", 
    "SearchResultWithCitations",
    "merge_citations",
    "group_citations_by_document",
    "filter_overlapping_citations",
    
    # Embeddings
    "EmbeddingsServiceWrapper",
    "EmbeddingsService",
    "create_embeddings_service",
    
    # Vector stores
    "SearchResult",
    "VectorStore",
    "ChromaVectorStore", 
    "InMemoryVectorStore",
    "create_vector_store",
    
    # Main RAG service
    "RAGService",
    "RAGConfig",
    "IndexingResult",
    "create_and_index",
    "create_rag_service",
    
    # Enhanced RAG services
    "EnhancedRAGService",
    "create_enhanced_rag_service",
    "EnhancedRAGServiceV2",
    
    # Cache
    "SimpleRAGCache",
    "get_rag_cache",
    
    # Configuration
    "EmbeddingConfig",
    "VectorStoreConfig",
    "ChunkingConfig",
    "SearchConfig",
    "create_config_for_collection",
    "create_config_for_testing",
    
    # Factory functions
    "create_rag_service",
    "create_rag_service_from_config",
    "create_auto_rag_service",
    "get_available_profiles",
]

__version__ = "0.1.0"