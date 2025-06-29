# __init__.py
# Description: RAG Services package initialization
#
# Imports

# Import the simplified services
try:
    from .simplified import (
        RAGService as SimplifiedRAGService,
        RAGConfig as SimplifiedRAGConfig,
        create_config_for_collection,
        create_config_for_testing,
        IndexingResult,
        SearchResult,
        SearchResultWithCitations,
        Citation,
        CitationType
    )
    SIMPLIFIED_RAG_AVAILABLE = True
except ImportError:
    SIMPLIFIED_RAG_AVAILABLE = False
    SimplifiedRAGService = None
    SimplifiedRAGConfig = None
    create_config_for_collection = None
    create_config_for_testing = None
    IndexingResult = None
    SearchResult = None
    SearchResultWithCitations = None
    Citation = None
    CitationType = None

# Import the original services for backward compatibility
# Note: These will be deprecated in favor of the simplified services
try:
    from .embeddings_service import EmbeddingsService
    from .chunking_service import ChunkingService
    from .indexing_service import IndexingService
    from .cache_service import CacheService, get_cache_service
    from .memory_management_service import MemoryManagementService, MemoryManagementConfig
    from .config_integration import get_rag_config, get_memory_config, save_rag_setting, get_rag_setting
    from .service_factory import RAGServiceFactory, create_rag_services, create_embeddings_service, create_indexing_service, create_modular_rag_service
    OLD_SERVICES_AVAILABLE = True
except ImportError:
    OLD_SERVICES_AVAILABLE = False
    # Provide stub implementations for backward compatibility
    EmbeddingsService = None
    ChunkingService = None
    IndexingService = None
    CacheService = None
    get_cache_service = None
    MemoryManagementService = None
    MemoryManagementConfig = None
    get_rag_config = None
    get_memory_config = None
    save_rag_setting = None
    get_rag_setting = None
    RAGServiceFactory = None
    create_rag_services = None
    create_embeddings_service = None
    create_indexing_service = None
    create_modular_rag_service = None

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
    # Simplified services (preferred)
    'SimplifiedRAGService',
    'SimplifiedRAGConfig',
    'create_config_for_collection',
    'create_config_for_testing',
    'IndexingResult',
    'SearchResult',
    'SearchResultWithCitations',
    'Citation',
    'CitationType',
    'SIMPLIFIED_RAG_AVAILABLE',
    # Original services (backward compatibility)
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