# __init__.py
# Description: RAG Services package initialization
#
# Imports
from .embeddings_service import EmbeddingsService
from .chunking_service import ChunkingService
from .indexing_service import IndexingService
from .cache_service import CacheService, get_cache_service
from .memory_management_service import MemoryManagementService, MemoryManagementConfig
from .config_integration import get_rag_config, get_memory_config, save_rag_setting, get_rag_setting
from .service_factory import RAGServiceFactory, create_rag_services, create_embeddings_service, create_indexing_service, create_modular_rag_service

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