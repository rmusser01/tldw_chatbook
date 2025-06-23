# __init__.py
# Description: RAG Services package initialization
#
# Imports
# Import the new modular RAG service
try:
    from .rag_service.integration import RAGService
    from .rag_service.app import RAGApplication
    from .rag_service.config import RAGConfig
    RAG_SERVICE_AVAILABLE = True
except ImportError as e:
    RAG_SERVICE_AVAILABLE = False
    RAGService = None
    RAGApplication = None
    RAGConfig = None

__all__ = [
    'EmbeddingsService',
    'ChunkingService',
    'IndexingService',
    'CacheService',
    'get_cache_service',
    'MemoryManagementService',
    'MemoryManagementConfig',
    'get_rag_config',
    'get_memory_config',
    'save_rag_setting',
    'get_rag_setting',
    'RAGServiceFactory',
    'create_rag_services',
    'create_embeddings_service',
    'create_indexing_service',
    'create_modular_rag_service',
    'RAGService',
    'RAGApplication',
    'RAGConfig',
    'RAG_SERVICE_AVAILABLE'
]