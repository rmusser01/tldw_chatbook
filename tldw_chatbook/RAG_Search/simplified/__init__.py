"""Lazy public aliases for simplified RAG services and lightweight metadata.

Importing a configuration submodule does not initialize optional runtime engines.
An explicit runtime export still imports its original owner and surfaces that
owner's real dependency failure; successful aliases retain object identity.
"""

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Citation": ("citations", "Citation"),
    "CitationType": ("citations", "CitationType"),
    "SearchResultWithCitations": ("citations", "SearchResultWithCitations"),
    "merge_citations": ("citations", "merge_citations"),
    "group_citations_by_document": ("citations", "group_citations_by_document"),
    "filter_overlapping_citations": ("citations", "filter_overlapping_citations"),
    "EmbeddingsServiceWrapper": ("embeddings_wrapper", "EmbeddingsServiceWrapper"),
    "EmbeddingsService": ("embeddings_wrapper", "EmbeddingsServiceWrapper"),
    "create_embeddings_service": ("embeddings_wrapper", "create_embeddings_service"),
    "SearchResult": ("vector_store", "SearchResult"),
    "VectorStore": ("vector_store", "VectorStore"),
    "VectorStoreSearchError": ("vector_store", "VectorStoreSearchError"),
    "ChromaVectorStore": ("vector_store", "ChromaVectorStore"),
    "InMemoryVectorStore": ("vector_store", "InMemoryVectorStore"),
    "create_vector_store": ("vector_store", "create_vector_store"),
    "RAGService": ("rag_service", "RAGService"),
    "RAGConfig": ("config", "RAGConfig"),
    "create_and_index": ("rag_service", "create_and_index"),
    "EnhancedRAGService": ("enhanced_rag_service", "EnhancedRAGService"),
    "create_enhanced_rag_service": (
        "enhanced_rag_service",
        "create_enhanced_rag_service",
    ),
    "EnhancedRAGServiceV2": ("enhanced_rag_service_v2", "EnhancedRAGServiceV2"),
    "IndexingResult": ("data_models", "IndexingResult"),
    "SimpleRAGCache": ("simple_cache", "SimpleRAGCache"),
    "get_rag_cache": ("simple_cache", "get_rag_cache"),
    "EmbeddingConfig": ("config", "EmbeddingConfig"),
    "VectorStoreConfig": ("config", "VectorStoreConfig"),
    "ChunkingConfig": ("config", "ChunkingConfig"),
    "SearchConfig": ("config", "SearchConfig"),
    "create_config_for_collection": ("config", "create_config_for_collection"),
    "create_config_for_testing": ("config", "create_config_for_testing"),
    "create_rag_service": ("rag_factory", "create_rag_service"),
    "create_rag_service_from_config": ("rag_factory", "create_rag_service_from_config"),
    "create_auto_rag_service": ("rag_factory", "create_auto_rag_service"),
    "get_available_profiles": ("rag_factory", "get_available_profiles"),
}

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
    "VectorStoreSearchError",
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


def __getattr__(name: str) -> Any:
    """Resolve an original public object on first attribute access."""
    try:
        module, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(f".{module}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include public lazy exports without importing their owners."""
    return sorted(set(globals()) | set(__all__))
