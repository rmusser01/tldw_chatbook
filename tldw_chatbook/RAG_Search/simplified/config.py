"""
Simplified RAG configuration management.

This module provides configuration handling for the simplified RAG implementation,
integrating with the existing tldw_cli configuration system while providing
sensible defaults and easy overrides.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import os
import logging

# Import the main config module to access existing configuration
from tldw_chatbook.config import get_cli_setting, load_cli_config_and_ensure_existence

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None
    cache_size: int = 2
    batch_size: int = 32
    max_length: int = 512
    # For OpenAI or API-based models
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class VectorStoreConfig:
    """Configuration for vector storage."""
    type: str = "chroma"  # "chroma" or "memory"
    persist_directory: Optional[Path] = None
    collection_name: str = "default"
    distance_metric: str = "cosine"  # "cosine", "l2", "ip"
    # Collection names for different content types
    media_collection: str = "media_embeddings"
    chat_collection: str = "chat_embeddings"
    notes_collection: str = "notes_embeddings"
    character_collection: str = "character_embeddings"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 400  # in words
    chunk_overlap: int = 100  # in words
    chunking_method: str = "words"  # "words", "sentences", "paragraphs"
    min_chunk_size: int = 50  # minimum words per chunk
    max_chunk_size: int = 1000  # maximum words per chunk


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    default_top_k: int = 10
    score_threshold: float = 0.0
    include_citations: bool = True
    # Search-specific settings
    fts_top_k: int = 10  # For keyword search
    vector_top_k: int = 10  # For semantic search
    hybrid_alpha: float = 0.5  # Weight for hybrid search (0=keyword only, 1=semantic only)
    # Re-ranking
    enable_reranking: bool = False
    reranker_model: Optional[str] = None
    reranker_top_k: int = 5
    # Cache settings
    enable_cache: bool = True
    cache_size: int = 100
    cache_ttl: float = 3600  # 1 hour in seconds


@dataclass
class RAGConfig:
    """Complete RAG configuration."""
    # Component configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    
    # Convenience shortcuts for common settings
    @property
    def embedding_model(self) -> str:
        return self.embedding.model
    
    @property
    def vector_store_type(self) -> str:
        return self.vector_store.type
    
    @property
    def persist_directory(self) -> Optional[Path]:
        return self.vector_store.persist_directory
    
    @property
    def collection_name(self) -> str:
        return self.vector_store.collection_name
    
    @property
    def distance_metric(self) -> str:
        return self.vector_store.distance_metric
    
    @property
    def chunk_size(self) -> int:
        return self.chunking.chunk_size
    
    @property
    def chunk_overlap(self) -> int:
        return self.chunking.chunk_overlap
    
    @property
    def chunking_method(self) -> str:
        return self.chunking.chunking_method
    
    @property
    def default_top_k(self) -> int:
        return self.search.default_top_k
    
    @property
    def score_threshold(self) -> float:
        return self.search.score_threshold
    
    @property
    def include_citations(self) -> bool:
        return self.search.include_citations
    
    @property
    def device(self) -> Optional[str]:
        return self.embedding.device
    
    @property
    def embedding_cache_size(self) -> int:
        return self.embedding.cache_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGConfig':
        """Create configuration from dictionary."""
        # Extract sub-configurations
        embedding_data = data.get('embedding', {})
        vector_store_data = data.get('vector_store', {})
        chunking_data = data.get('chunking', {})
        search_data = data.get('search', {})
        
        # Handle path conversion for persist_directory
        if 'persist_directory' in vector_store_data and vector_store_data['persist_directory']:
            vector_store_data['persist_directory'] = Path(vector_store_data['persist_directory'])
        
        return cls(
            embedding=EmbeddingConfig(**embedding_data),
            vector_store=VectorStoreConfig(**vector_store_data),
            chunking=ChunkingConfig(**chunking_data),
            search=SearchConfig(**search_data)
        )
    
    @classmethod
    def from_settings(cls, 
                     override_embedding_model: Optional[str] = None,
                     override_persist_dir: Optional[Union[str, Path]] = None) -> 'RAGConfig':
        """
        Load configuration from tldw_cli settings with optional overrides.
        
        This integrates with the existing configuration system, loading from:
        1. Environment variables (highest priority)
        2. TOML configuration file
        3. Defaults defined in this class (lowest priority)
        
        Args:
            override_embedding_model: Override the embedding model
            override_persist_dir: Override the persist directory
            
        Returns:
            RAGConfig instance with loaded settings
        """
        # Load the main configuration
        config_dict = load_cli_config_and_ensure_existence()
        
        # Get RAG-specific configuration section
        rag_config = get_cli_setting("AppRAGSearchConfig", "rag", {})
        if not isinstance(rag_config, dict):
            rag_config = {}
        
        # Create default configuration
        config = cls()
        
        # === Embedding Configuration ===
        embedding_section = rag_config.get('embedding', {})
        
        # Model selection (priority: override > env > config > default)
        config.embedding.model = (
            override_embedding_model or
            os.getenv("RAG_EMBEDDING_MODEL") or
            embedding_section.get('model') or
            get_cli_setting("AppRAGSearchConfig", "embedding_model") or  # Legacy location
            config.embedding.model
        )
        
        # Device selection
        config.embedding.device = (
            os.getenv("RAG_DEVICE") or
            embedding_section.get('device') or
            config.embedding.device
        )
        
        # Cache settings
        config.embedding.cache_size = int(
            os.getenv("RAG_EMBEDDING_CACHE_SIZE") or
            embedding_section.get('cache_size', config.embedding.cache_size)
        )
        
        config.embedding.batch_size = int(
            embedding_section.get('batch_size', config.embedding.batch_size)
        )
        
        # API settings for OpenAI-compatible models
        config.embedding.api_key = (
            os.getenv("OPENAI_API_KEY") or
            get_cli_setting("API", "openai_api_key")
        )
        
        config.embedding.base_url = (
            os.getenv("RAG_EMBEDDING_BASE_URL") or
            embedding_section.get('base_url')
        )
        
        # === Vector Store Configuration ===
        vector_store_section = rag_config.get('vector_store', {})
        
        config.vector_store.type = (
            os.getenv("RAG_VECTOR_STORE") or
            vector_store_section.get('type') or
            config.vector_store.type
        )
        
        # Persist directory (priority: override > env > config > default)
        persist_dir = (
            override_persist_dir or
            os.getenv("RAG_PERSIST_DIR") or
            vector_store_section.get('persist_directory') or
            rag_config.get('chroma', {}).get('persist_directory')  # Legacy location
        )
        
        if persist_dir:
            config.vector_store.persist_directory = Path(persist_dir)
        else:
            # Default to user data directory
            config.vector_store.persist_directory = (
                Path.home() / ".local" / "share" / "tldw_cli" / "chromadb"
            )
        
        # Collection settings
        config.vector_store.collection_name = (
            vector_store_section.get('collection_name', config.vector_store.collection_name)
        )
        
        config.vector_store.distance_metric = (
            vector_store_section.get('distance_metric') or
            rag_config.get('chroma', {}).get('distance_metric') or  # Legacy
            config.vector_store.distance_metric
        )
        
        # Content-specific collections
        retriever_section = rag_config.get('retriever', {})
        config.vector_store.media_collection = (
            retriever_section.get('media_collection', config.vector_store.media_collection)
        )
        config.vector_store.chat_collection = (
            retriever_section.get('chat_collection', config.vector_store.chat_collection)
        )
        config.vector_store.notes_collection = (
            retriever_section.get('notes_collection', config.vector_store.notes_collection)
        )
        config.vector_store.character_collection = (
            retriever_section.get('character_collection', config.vector_store.character_collection)
        )
        
        # === Chunking Configuration ===
        chunking_section = rag_config.get('chunking', {})
        
        config.chunking.chunk_size = int(
            os.getenv("RAG_CHUNK_SIZE") or
            chunking_section.get('chunk_size') or
            retriever_section.get('chunk_size') or  # Legacy location
            config.chunking.chunk_size
        )
        
        config.chunking.chunk_overlap = int(
            os.getenv("RAG_CHUNK_OVERLAP") or
            chunking_section.get('chunk_overlap') or
            retriever_section.get('chunk_overlap') or  # Legacy
            config.chunking.chunk_overlap
        )
        
        config.chunking.chunking_method = (
            chunking_section.get('method', config.chunking.chunking_method)
        )
        
        # === Search Configuration ===
        search_section = rag_config.get('search', {})
        processor_section = rag_config.get('processor', {})
        
        config.search.default_top_k = int(
            os.getenv("RAG_TOP_K") or
            search_section.get('default_top_k') or
            retriever_section.get('vector_top_k') or  # Legacy
            config.search.default_top_k
        )
        
        config.search.score_threshold = float(
            search_section.get('score_threshold', config.search.score_threshold)
        )
        
        config.search.include_citations = (
            search_section.get('include_citations', config.search.include_citations)
        )
        
        # Search type specific settings
        config.search.fts_top_k = int(
            retriever_section.get('fts_top_k', config.search.fts_top_k)
        )
        
        config.search.vector_top_k = int(
            retriever_section.get('vector_top_k', config.search.vector_top_k)
        )
        
        config.search.hybrid_alpha = float(
            retriever_section.get('hybrid_alpha', config.search.hybrid_alpha)
        )
        
        # Re-ranking settings
        config.search.enable_reranking = (
            processor_section.get('enable_reranking', config.search.enable_reranking)
        )
        
        config.search.reranker_model = (
            processor_section.get('reranker_model')
        )
        
        config.search.reranker_top_k = int(
            processor_section.get('reranker_top_k', config.search.reranker_top_k)
        )
        
        # Log the loaded configuration
        logger.info(f"Loaded RAG configuration: embedding_model={config.embedding_model}, "
                   f"vector_store={config.vector_store_type}, "
                   f"persist_dir={config.persist_directory}")
        
        return config
    
    def validate(self) -> List[str]:
        """
        Validate the configuration and return any issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate chunk sizes
        if self.chunking.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.chunking.chunk_overlap >= self.chunking.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")
        
        if self.chunking.chunk_overlap < 0:
            errors.append("chunk_overlap cannot be negative")
        
        # Validate search settings
        if self.search.default_top_k <= 0:
            errors.append("default_top_k must be positive")
        
        if self.search.score_threshold < 0 or self.search.score_threshold > 1:
            errors.append("score_threshold must be between 0 and 1")
        
        if self.search.hybrid_alpha < 0 or self.search.hybrid_alpha > 1:
            errors.append("hybrid_alpha must be between 0 and 1")
        
        # Validate vector store
        if self.vector_store.type not in ["chroma", "memory"]:
            errors.append(f"Unknown vector store type: {self.vector_store.type}")
        
        if self.vector_store.type == "chroma" and not self.vector_store.persist_directory:
            errors.append("persist_directory is required for chroma vector store")
        
        if self.vector_store.distance_metric not in ["cosine", "l2", "ip"]:
            errors.append(f"Unknown distance metric: {self.vector_store.distance_metric}")
        
        # Validate embedding settings
        if self.embedding.cache_size <= 0:
            errors.append("embedding cache_size must be positive")
        
        if self.embedding.batch_size <= 0:
            errors.append("embedding batch_size must be positive")
        
        return errors


# Convenience functions for common configuration patterns

def create_config_for_collection(
    collection_type: str,
    embedding_model: Optional[str] = None,
    persist_dir: Optional[Union[str, Path]] = None
) -> RAGConfig:
    """
    Create a RAG configuration for a specific collection type.
    
    Args:
        collection_type: One of "media", "chat", "notes", "character"
        embedding_model: Optional embedding model override
        persist_dir: Optional persist directory override
        
    Returns:
        RAGConfig configured for the specified collection
    """
    config = RAGConfig.from_settings(embedding_model, persist_dir)
    
    # Set the appropriate collection name based on type
    collection_map = {
        "media": config.vector_store.media_collection,
        "chat": config.vector_store.chat_collection,
        "notes": config.vector_store.notes_collection,
        "character": config.vector_store.character_collection
    }
    
    if collection_type in collection_map:
        config.vector_store.collection_name = collection_map[collection_type]
    else:
        logger.warning(f"Unknown collection type: {collection_type}, using default")
    
    return config


def create_config_for_testing(
    use_memory_store: bool = True,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> RAGConfig:
    """
    Create a RAG configuration suitable for testing.
    
    Args:
        use_memory_store: If True, use in-memory vector store
        embedding_model: Embedding model to use
        
    Returns:
        RAGConfig configured for testing
    """
    config = RAGConfig()
    config.embedding.model = embedding_model
    config.embedding.cache_size = 1  # Minimal cache for testing
    
    if use_memory_store:
        config.vector_store.type = "memory"
        config.vector_store.persist_directory = None
    else:
        config.vector_store.type = "chroma"
        config.vector_store.persist_directory = Path("/tmp/test_rag_chromadb")
    
    # Use smaller chunks for testing
    config.chunking.chunk_size = 100
    config.chunking.chunk_overlap = 20
    
    # Faster search for tests
    config.search.default_top_k = 5
    
    return config


# Example TOML configuration structure for documentation
EXAMPLE_TOML_CONFIG = """
# Example RAG configuration in config.toml

[AppRAGSearchConfig.rag]
# Embedding configuration
[AppRAGSearchConfig.rag.embedding]
model = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda"  # or "cpu", "mps"
cache_size = 2
batch_size = 32

# Vector store configuration
[AppRAGSearchConfig.rag.vector_store]
type = "chroma"  # or "memory"
persist_directory = "~/.local/share/tldw_cli/chromadb"
collection_name = "default"
distance_metric = "cosine"  # or "l2", "ip"

# Chunking configuration
[AppRAGSearchConfig.rag.chunking]
chunk_size = 400
chunk_overlap = 100
method = "words"  # or "sentences", "paragraphs"

# Search configuration
[AppRAGSearchConfig.rag.search]
default_top_k = 10
score_threshold = 0.0
include_citations = true
fts_top_k = 10
vector_top_k = 10
hybrid_alpha = 0.5

# Legacy configuration locations (still supported)
[AppRAGSearchConfig.rag.retriever]
media_collection = "media_embeddings"
chat_collection = "chat_embeddings"
notes_collection = "notes_embeddings"
character_collection = "character_embeddings"

[AppRAGSearchConfig.rag.processor]
enable_reranking = false
reranker_model = null
reranker_top_k = 5
"""