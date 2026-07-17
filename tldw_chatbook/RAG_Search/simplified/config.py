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
from loguru import logger

# Import the main config module to access existing configuration
from tldw_chatbook.config import get_cli_setting, load_cli_config_and_ensure_existence, get_user_data_dir

# Server-parity hybrid fusion default (alpha weights the vector leg)
from ..fusion import DEFAULT_HYBRID_ALPHA


def _coerce_int_setting(value: Any, default: int) -> int:
    """Coerce user-editable integer settings without raising on bad config."""
    if isinstance(value, bool):
        return default
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    if not parsed.is_integer():
        return default
    return int(parsed)


# Canonical vector store type values used by the selection logic below.
# "auto" is a sentinel meaning "resolve from env/config/installed deps".
VECTOR_STORE_TYPE_AUTO = "auto"
VECTOR_STORE_TYPE_CHROMA = "chroma"
VECTOR_STORE_TYPE_MEMORY = "memory"

# Cached result of the embeddings_rag installed-probe. Availability cannot
# change without a restart, so probe at most once per process.
_EMBEDDINGS_RAG_AVAILABLE: Optional[bool] = None


def _embeddings_rag_available() -> bool:
    """Return True when the `embeddings_rag` optional dependencies are installed.

    Wraps `Utils.optional_deps.embeddings_rag_deps_installed()` (a cheap
    `find_spec`-based probe of the 'embeddings_rag' feature group — no heavy
    imports, no side effects) and caches the result. Any failure is treated
    as "not available" so environments without the extras never error.

    Returns:
        True if all embeddings/RAG dependencies are installed.
    """
    global _EMBEDDINGS_RAG_AVAILABLE
    if _EMBEDDINGS_RAG_AVAILABLE is None:
        try:
            from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed
            _EMBEDDINGS_RAG_AVAILABLE = bool(embeddings_rag_deps_installed())
        except Exception as e:
            logger.debug(f"embeddings_rag availability check failed, assuming unavailable: {e}")
            _EMBEDDINGS_RAG_AVAILABLE = False
    return _EMBEDDINGS_RAG_AVAILABLE


def _explicit_rag_setting(section: str, key: str) -> Optional[Any]:
    """Read an explicit `[AppRAGSearchConfig.rag.<section>].<key>` user setting.

    Args:
        section: Subsection name within the rag config (e.g. 'vector_store').
        key: Setting name within that subsection (e.g. 'type').

    Returns:
        The configured value, or None when not explicitly set (or on any
        config read failure).
    """
    try:
        rag_section = get_cli_setting("AppRAGSearchConfig", "rag", {}) or {}
        if not isinstance(rag_section, dict):
            return None
        subsection = rag_section.get(section, {})
        if not isinstance(subsection, dict):
            return None
        return subsection.get(key)
    except Exception as e:
        logger.debug(f"Could not read rag.{section}.{key} from user config: {e}")
        return None


def _normalized_type_setting(value: Any) -> Optional[str]:
    """Normalize an explicit vector store type setting.

    Args:
        value: Raw setting value from env/config.

    Returns:
        Stripped, lowercased type string, or None when the value is unset,
        blank, or the "auto" sentinel (which means "run auto-detection").
    """
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized == VECTOR_STORE_TYPE_AUTO:
        return None
    return normalized


def _cleaned_path_setting(value: Any) -> Optional[str]:
    """Strip an explicit path setting, treating blank values as unset.

    Args:
        value: Raw setting value from env/config.

    Returns:
        Stripped path string, or None when unset or whitespace-only.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def default_vector_store_type() -> str:
    """Resolve the default vector store type.

    Priority: RAG_VECTOR_STORE env var > explicit
    `[AppRAGSearchConfig.rag.vector_store].type` in user config > persistent
    ChromaDB when the `embeddings_rag` optional deps are installed >
    in-memory fallback. An explicit `type = "memory"` therefore always wins.
    Explicit values are normalized (stripped/lowercased); blank or "auto"
    values fall through to the next source, ending in auto-detection.

    Returns:
        Vector store type string ("chroma" or "memory", or the normalized
        explicit user-configured value).
    """
    explicit = (
        _normalized_type_setting(os.getenv("RAG_VECTOR_STORE"))
        or _normalized_type_setting(_explicit_rag_setting('vector_store', 'type'))
    )
    if explicit:
        return explicit
    return VECTOR_STORE_TYPE_CHROMA if _embeddings_rag_available() else VECTOR_STORE_TYPE_MEMORY


def default_chroma_persist_directory() -> Path:
    """Resolve the default persist directory for the ChromaDB store.

    Priority: RAG_PERSIST_DIR env var > explicit
    `[AppRAGSearchConfig.rag.vector_store].persist_directory` > legacy
    `[AppRAGSearchConfig.rag.chroma].persist_directory` (still honored by
    `RAGConfig.from_settings`) > `<user data dir>/chromadb` (the same
    user-data-dir convention as the application databases). Blank or
    whitespace-only values are treated as unset.

    Returns:
        Path to the ChromaDB persist directory.
    """
    explicit = (
        _cleaned_path_setting(os.getenv("RAG_PERSIST_DIR"))
        or _cleaned_path_setting(_explicit_rag_setting('vector_store', 'persist_directory'))
        or _cleaned_path_setting(_explicit_rag_setting('chroma', 'persist_directory'))  # Legacy location
    )
    if explicit:
        return Path(explicit).expanduser()
    return get_user_data_dir() / "chromadb"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model: str = "mxbai-embed-large-v1"  # Default to mxbai-embed-large-v1 (high-quality embeddings)
    device: Optional[str] = "auto"  # auto-detect best device
    cache_size: int = 2
    batch_size: int = 16  # Reduced for larger model
    max_length: int = 512
    # For OpenAI or API-based models
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class VectorStoreConfig:
    """Configuration for vector storage.

    The default `type` is the "auto" sentinel, resolved on construction to
    persistent ChromaDB when the `embeddings_rag` optional deps are installed
    (persisting under the user data dir), and the in-memory store otherwise.
    Explicit values — constructor arguments, `RAG_VECTOR_STORE` env var, or
    `[AppRAGSearchConfig.rag.vector_store]` in the user config — always win.
    """
    type: str = VECTOR_STORE_TYPE_AUTO  # resolved in __post_init__: chroma (persistent) with embeddings deps, else memory
    persist_directory: Optional[Path] = None
    collection_name: str = "default"
    distance_metric: str = "cosine"  # "cosine", "l2", "ip"
    # Collection names for different content types
    media_collection: str = "media_embeddings"
    chat_collection: str = "chat_embeddings"
    notes_collection: str = "notes_embeddings"
    character_collection: str = "character_embeddings"

    def __post_init__(self):
        if self.type == VECTOR_STORE_TYPE_AUTO:
            self.type = default_vector_store_type()
        if self.persist_directory is None and self.type == VECTOR_STORE_TYPE_CHROMA:
            self.persist_directory = default_chroma_persist_directory()


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 400  # in words
    chunk_overlap: int = 100  # in words
    chunking_method: str = "words"  # "words", "sentences", "paragraphs"
    min_chunk_size: int = 50  # minimum words per chunk
    max_chunk_size: int = 1000  # maximum words per chunk
    
    # Parent document retrieval settings
    enable_parent_retrieval: bool = False
    parent_size_multiplier: int = 3  # Parent chunks are this many times larger
    
    # Structural chunking settings
    preserve_structure: bool = False
    clean_artifacts: bool = False  # Clean PDF artifacts
    preserve_tables: bool = False  # Serialize tables for better understanding


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    default_top_k: int = 10
    score_threshold: float = 0.0
    include_citations: bool = True
    citation_style: str = "inline"  # "inline", "footnote", or "none"
    snippet_max_chars: int = 240
    # Search mode
    default_search_mode: str = "semantic"  # "plain", "semantic", or "hybrid"
    # Search-specific settings
    fts_top_k: int = 10  # For keyword search
    vector_top_k: int = 10  # For semantic search
    # Hybrid fusion alpha: weight of the vector leg in the RRF blend
    # (0 = FTS/keyword only, 1 = vector/semantic only). Default 0.7 matches
    # tldw_server. Authoritative TOML knob:
    # [AppRAGSearchConfig.rag.retriever] hybrid_alpha
    hybrid_alpha: float = DEFAULT_HYBRID_ALPHA
    # Re-ranking
    enable_reranking: bool = False
    reranker_model: Optional[str] = None
    reranker_top_k: int = 5
    # Cache settings
    enable_cache: bool = True
    cache_size: int = 100
    cache_ttl: float = 3600  # 1 hour in seconds (default for all search types)
    # Search-type specific cache TTLs (optional)
    semantic_cache_ttl: Optional[float] = None  # TTL for semantic search results
    keyword_cache_ttl: Optional[float] = None   # TTL for keyword search results  
    hybrid_cache_ttl: Optional[float] = None    # TTL for hybrid search results
    # Database connection settings
    fts5_connection_pool_size: int = 3  # Connection pool size for FTS5 searches
    
    # Parent document inclusion settings (RAG pipeline feature)
    include_parent_docs: bool = False
    parent_size_threshold: int = 5000  # Maximum parent doc size in characters
    parent_inclusion_strategy: str = "size_based"  # "always", "size_based", "never"
    max_context_size: int = 16000  # Maximum total context size in characters


@dataclass
class QueryExpansionConfig:
    """Configuration for query expansion/rewriting.

    NOTE (task-252): The QueryExpander module (RAG_Search/query_expansion.py) that
    consumed this config was removed as dead code. This dataclass is intentionally
    kept so the [AppRAGSearchConfig.rag.query_expansion] TOML section and RAGConfig
    to_dict/from_dict round-trips keep working; no runtime code performs query
    expansion with it today.
    """
    enabled: bool = False
    method: str = "llm"  # "llm", "local_llm", "llamafile", "keywords"
    max_sub_queries: int = 3
    llm_provider: str = "openai"  # Which LLM provider to use
    llm_model: str = "gpt-3.5-turbo"  # Model for query expansion
    local_model: str = "Qwen3-0.6B-Q6_K.gguf"  # For Ollama/local models/llamafile
    expansion_prompt_template: str = "default"  # Template name or custom prompt
    combine_results: bool = True  # Combine results from all sub-queries
    cache_expansions: bool = True  # Cache expanded queries


@dataclass
class PipelineConfig:
    """Configuration for pipeline selection and behavior."""
    default_pipeline: str = "hybrid"
    enable_pipeline_metrics: bool = True
    pipeline_timeout_seconds: float = 30.0
    max_concurrent_pipelines: int = 3
    cache_pipeline_results: bool = True
    pipeline_config_file: Optional[Path] = None
    
    # Pipeline-specific overrides
    pipeline_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RAGConfig:
    """Complete RAG configuration."""
    # Component configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    query_expansion: QueryExpansionConfig = field(default_factory=QueryExpansionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
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
        query_expansion_data = data.get('query_expansion', {})
        pipeline_data = data.get('pipeline', {})
        
        # Handle path conversion for persist_directory
        if 'persist_directory' in vector_store_data and vector_store_data['persist_directory']:
            vector_store_data['persist_directory'] = Path(vector_store_data['persist_directory'])
        
        # Handle path conversion for pipeline_config_file
        if 'pipeline_config_file' in pipeline_data and pipeline_data['pipeline_config_file']:
            pipeline_data['pipeline_config_file'] = Path(pipeline_data['pipeline_config_file'])
        
        return cls(
            embedding=EmbeddingConfig(**embedding_data),
            vector_store=VectorStoreConfig(**vector_store_data),
            chunking=ChunkingConfig(**chunking_data),
            search=SearchConfig(**search_data),
            query_expansion=QueryExpansionConfig(**query_expansion_data),
            pipeline=PipelineConfig(**pipeline_data)
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
        
        # Check if we should use the embedding_config default model
        embedding_config = get_cli_setting("embedding_config", None, {})
        default_model_from_embedding_config = embedding_config.get('default_model_id', None)
        
        # Model selection (priority: override > env > rag.embedding > embedding_config default > class default)
        config.embedding.model = (
            override_embedding_model or
            os.getenv("RAG_EMBEDDING_MODEL") or
            embedding_section.get('model') or
            default_model_from_embedding_config or
            get_cli_setting("AppRAGSearchConfig", "embedding_model") or  # Legacy location
            config.embedding.model
        )
        
        # Device selection with auto-detection support
        device_setting = (
            os.getenv("RAG_DEVICE") or
            embedding_section.get('device') or
            config.embedding.device
        )
        
        # Handle "auto" device selection
        if device_setting == "auto":
            # Try to detect best available device
            try:
                import torch
                if torch.cuda.is_available():
                    config.embedding.device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    config.embedding.device = "mps"
                else:
                    config.embedding.device = "cpu"
                logger.info(f"Auto-detected device: {config.embedding.device}")
            except ImportError:
                config.embedding.device = "cpu"
                logger.info("Torch not available, defaulting to CPU")
        else:
            config.embedding.device = device_setting
        
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

        # vector_store.type is already resolved by VectorStoreConfig.__post_init__
        # with the same precedence (RAG_VECTOR_STORE env > explicit config >
        # installed-deps default) plus normalization; re-applying the raw
        # env/config values here would undo that normalization.

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
            # Default to user-specific data directory
            user_dir = get_user_data_dir()
            config.vector_store.persist_directory = user_dir / "chromadb"
        
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

        config.search.citation_style = (
            search_section.get('citation_style', config.search.citation_style)
        )

        config.search.snippet_max_chars = _coerce_int_setting(
            search_section.get('snippet_max_chars', config.search.snippet_max_chars),
            config.search.snippet_max_chars,
        )

        config.search.max_context_size = _coerce_int_setting(
            search_section.get('max_context_size', config.search.max_context_size),
            config.search.max_context_size,
        )
        
        # Search mode configuration
        config.search.default_search_mode = (
            os.getenv("RAG_SEARCH_MODE") or
            search_section.get('default_search_mode', config.search.default_search_mode)
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
        
        # Cache and database connection settings
        config.search.cache_size = int(
            search_section.get('cache_size', config.search.cache_size)
        )
        
        config.search.cache_ttl = float(
            search_section.get('cache_ttl', config.search.cache_ttl)
        )
        
        config.search.fts5_connection_pool_size = int(
            search_section.get('fts5_connection_pool_size', config.search.fts5_connection_pool_size)
        )
        
        # Search-type specific cache TTLs
        if 'semantic_cache_ttl' in search_section:
            config.search.semantic_cache_ttl = float(search_section['semantic_cache_ttl'])
        
        if 'keyword_cache_ttl' in search_section:
            config.search.keyword_cache_ttl = float(search_section['keyword_cache_ttl'])
            
        if 'hybrid_cache_ttl' in search_section:
            config.search.hybrid_cache_ttl = float(search_section['hybrid_cache_ttl'])
        
        # === Query Expansion Configuration ===
        query_expansion_section = rag_config.get('query_expansion', {})
        
        config.query_expansion.enabled = query_expansion_section.get('enabled', config.query_expansion.enabled)
        config.query_expansion.method = query_expansion_section.get('method', config.query_expansion.method)
        
        # === Pipeline Configuration ===
        pipeline_section = rag_config.get('pipeline', {})
        
        config.pipeline.default_pipeline = (
            os.getenv("RAG_DEFAULT_PIPELINE") or
            pipeline_section.get('default_pipeline', config.pipeline.default_pipeline)
        )
        
        config.pipeline.enable_pipeline_metrics = pipeline_section.get(
            'enable_pipeline_metrics', config.pipeline.enable_pipeline_metrics
        )
        
        config.pipeline.pipeline_timeout_seconds = float(
            pipeline_section.get('pipeline_timeout_seconds', config.pipeline.pipeline_timeout_seconds)
        )
        
        config.pipeline.cache_pipeline_results = pipeline_section.get(
            'cache_pipeline_results', config.pipeline.cache_pipeline_results
        )
        
        # Pipeline config file location
        if 'pipeline_config_file' in pipeline_section:
            config.pipeline.pipeline_config_file = Path(pipeline_section['pipeline_config_file'])
        
        # Pipeline-specific overrides
        if 'pipeline_overrides' in pipeline_section:
            config.pipeline.pipeline_overrides = pipeline_section['pipeline_overrides']
        
        
        config.query_expansion.method = (
            query_expansion_section.get('method', config.query_expansion.method)
        )
        
        config.query_expansion.max_sub_queries = int(
            query_expansion_section.get('max_sub_queries', config.query_expansion.max_sub_queries)
        )
        
        config.query_expansion.llm_provider = (
            query_expansion_section.get('llm_provider', config.query_expansion.llm_provider)
        )
        
        config.query_expansion.llm_model = (
            query_expansion_section.get('llm_model', config.query_expansion.llm_model)
        )
        
        config.query_expansion.local_model = (
            query_expansion_section.get('local_model', config.query_expansion.local_model)
        )
        
        config.query_expansion.expansion_prompt_template = (
            query_expansion_section.get('expansion_prompt_template', config.query_expansion.expansion_prompt_template)
        )
        
        config.query_expansion.combine_results = (
            query_expansion_section.get('combine_results', config.query_expansion.combine_results)
        )
        
        config.query_expansion.cache_expansions = (
            query_expansion_section.get('cache_expansions', config.query_expansion.cache_expansions)
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
        if self.vector_store.type not in [VECTOR_STORE_TYPE_CHROMA, VECTOR_STORE_TYPE_MEMORY]:
            errors.append(f"Unknown vector store type: {self.vector_store.type}")

        if self.vector_store.type == VECTOR_STORE_TYPE_CHROMA and not self.vector_store.persist_directory:
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
    embedding_model: str = "mock"
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
model = "mxbai-embed-large-v1"  # Uses model from [embedding_config.models.mxbai-embed-large-v1]
device = "auto"  # Auto-detect best device ("auto", "cpu", "cuda", "mps")
cache_size = 2
batch_size = 16  # Reduced for larger model
max_length = 512
# For API-based models (optional):
# api_key = "your-api-key"  # Or use OPENAI_API_KEY env var
# base_url = "http://localhost:8080/v1"  # For local servers

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
citation_style = "inline"  # "inline", "footnote", or "none"
snippet_max_chars = 240
default_search_mode = "semantic"  # "plain", "semantic", or "hybrid"
max_context_size = 16000
fts_top_k = 10
vector_top_k = 10
cache_size = 100
cache_ttl = 3600  # 1 hour default for all search types
# Optional search-type specific cache TTLs
# semantic_cache_ttl = 7200  # 2 hours for semantic search
# keyword_cache_ttl = 1800   # 30 minutes for keyword search
# hybrid_cache_ttl = 3600    # 1 hour for hybrid search
fts5_connection_pool_size = 3  # Adjust based on concurrent search load

# Retriever configuration (authoritative location for hybrid_alpha)
[AppRAGSearchConfig.rag.retriever]
# Hybrid fusion alpha: weight of the vector leg in the RRF blend
# (0 = FTS only, 1 = vector only). Default 0.7 matches tldw_server.
hybrid_alpha = 0.7
media_collection = "media_embeddings"
chat_collection = "chat_embeddings"
notes_collection = "notes_embeddings"
character_collection = "character_embeddings"

[AppRAGSearchConfig.rag.processor]
enable_reranking = false
reranker_model = null
reranker_top_k = 5

# Query expansion configuration
[AppRAGSearchConfig.rag.query_expansion]
enabled = false
method = "llm"  # "llm", "local_llm", "llamafile", "keywords"
max_sub_queries = 3
llm_provider = "openai"
llm_model = "gpt-3.5-turbo"
local_model = "Qwen3-0.6B-Q6_K.gguf"  # For Ollama/llamafile
expansion_prompt_template = "default"
combine_results = true
cache_expansions = true
"""
