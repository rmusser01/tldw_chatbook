# RAG Module Simplification Plan

## Executive Summary

This document outlines a comprehensive plan to simplify the RAG (Retrieval-Augmented Generation) module in tldw_chatbook while preserving all user-facing functionality and maintaining the intentional modular architecture for embeddings and vector stores.

## Current State Analysis

### Architecture Overview

The current RAG implementation has a sophisticated but overly complex architecture:

```
RAG_Search/
├── Services/
│   ├── embeddings_service.py      # 800+ lines with provider abstraction
│   ├── service_factory.py         # 300+ lines of factory patterns
│   ├── indexing_service.py        # Document indexing coordination
│   ├── chunking_service.py        # Text chunking (already simple)
│   ├── memory_management_service.py # Complex memory policies
│   ├── cache_service.py           # Caching layer
│   └── rag_service/              # Nested "new" implementation
│       ├── rag_service.py
│       ├── retrieval/
│       ├── processing/
│       └── generation/
```

### Key Findings

1. **Over-Engineered Provider System**
   - Abstract `EmbeddingProvider` base class with multiple implementations
   - Runtime provider switching capability that's never used
   - Complex thread-safe provider management
   - Registration and switching methods add ~300 lines of unused code

2. **Unnecessary Abstractions**
   - `RAGServiceFactory` wraps simple object creation
   - `VectorStore` abstract base class when only ChromaDB is used
   - Multiple convenience functions that just delegate

3. **Unused Features**
   - Runtime provider switching
   - Multiple embedding providers per session
   - Complex memory management policies
   - Parallel processing configuration
   - Vector store selection UI

4. **Actual Usage Pattern**
   ```python
   # This is how RAG is actually used throughout the codebase:
   embeddings_service = EmbeddingsService(persist_dir)
   # No provider switching, no complex configuration
   ```

## Design Principles for Simplification

### 1. **Preserve Intentional Modularity**
- Keep embeddings and vector stores as separate, isolated modules
- Maintain clear interfaces between components
- Enable testing in isolation

### 2. **Configuration-Time Flexibility**
- Support different embedding models via configuration
- Allow vector store selection at startup
- Remove runtime switching complexity

### 3. **Focus on Actual Use Cases**
- Single-user TUI application
- One embedding model per session
- One vector store per session
- No need for complex concurrency

### 4. **Maintain All User Features**
- Search modes (Plain RAG, Semantic, Hybrid)
- Source selection (Media, Conversations, Notes)
- Configuration options (top-k, context, chunks)
- Re-ranking support
- Manual indexing
- Saved searches

## Proposed Architecture

### Simplified Structure

```
RAG_Search/
├── Services/
│   ├── embeddings_service.py      # ~200 lines (from 800+)
│   ├── vector_store.py            # ~150 lines (new, consolidates stores)
│   ├── chunking_service.py        # Keep as-is (already simple)
│   ├── indexing_service.py        # ~100 lines (simplified)
│   ├── rag_service.py             # ~200 lines (main coordinator)
│   └── backends/                  # Embedding backend implementations
│       ├── __init__.py
│       ├── sentence_transformer.py
│       ├── openai.py
│       └── huggingface.py
```

### Core Components

#### 1. Embeddings Service (Simplified)

```python
class EmbeddingsService:
    """
    Manages text embeddings with a single model per instance.
    Supports multiple backends through initialization-time selection.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_size: int = 1000, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = self._load_model(model_name)
        self.cache = LRUCache(maxsize=cache_size)
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _load_model(self, model_name: str):
        """Load appropriate backend based on model name prefix"""
        if model_name.startswith("openai/"):
            from .backends.openai import OpenAIBackend
            return OpenAIBackend(model_name.split("/", 1)[1])
        elif model_name.startswith("hf/"):
            from .backends.huggingface import HuggingFaceBackend
            return HuggingFaceBackend(model_name.split("/", 1)[1])
        else:
            from .backends.sentence_transformer import SentenceTransformerBackend
            return SentenceTransformerBackend(model_name, self.device)
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings with caching"""
        # Simple, direct implementation without provider abstraction
```

**Design Decisions:**
- No provider registration system - models selected at init time
- Simple backend loading based on model name convention
- Built-in caching without separate cache service
- Direct numpy array returns, no wrapper objects

#### 2. Vector Store Module

```python
class VectorStore(Protocol):
    """Simple protocol for vector stores - not an abstract base class"""
    def add(self, ids: List[str], embeddings: np.ndarray, 
            documents: List[str], metadata: List[dict]) -> None: ...
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[dict]: ...
    
    def delete_collection(self, name: str) -> None: ...

class ChromaVectorStore:
    """ChromaDB implementation - our primary vector store"""
    def __init__(self, persist_directory: Path, collection_name: str = "default"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._init_client()

class InMemoryVectorStore:
    """Simple in-memory fallback for testing or when ChromaDB unavailable"""
    def __init__(self):
        self.embeddings = []
        self.documents = []
        self.metadata = []
        self.ids = []

def create_vector_store(store_type: str, persist_directory: Path) -> VectorStore:
    """Simple factory function - not a complex factory class"""
    if store_type == "chroma":
        return ChromaVectorStore(persist_directory)
    elif store_type == "memory":
        return InMemoryVectorStore()
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
```

**Design Decisions:**
- Use Protocol instead of abstract base class (lighter weight)
- Direct implementations without unnecessary abstraction
- Simple factory function instead of factory class
- Remove update_documents (just delete and re-add)

#### 3. RAG Service (Main Coordinator)

```python
class RAGService:
    """
    Main RAG service that coordinates embeddings, vector stores, and search.
    Simple composition without factory complexity.
    """
    def __init__(self, config: Optional[RAGConfig] = None):
        config = config or RAGConfig.from_settings()
        
        # Direct instantiation - no factory needed
        self.embeddings = EmbeddingsService(
            model_name=config.embedding_model,
            cache_size=config.cache_size,
            device=config.device
        )
        
        self.vector_store = create_vector_store(
            config.vector_store_type,
            config.persist_directory
        )
        
        self.chunking = ChunkingService(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            chunking_strategy=config.chunking_strategy
        )
    
    async def index_document(self, doc_id: str, content: str, 
                           metadata: Optional[dict] = None) -> int:
        """Index a document - returns number of chunks created"""
        metadata = metadata or {}
        
        # Simple, direct flow
        chunks = self.chunking.chunk_text(content)
        if not chunks:
            return 0
        
        # Create embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embeddings.create_embeddings(chunk_texts)
        
        # Prepare for storage
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        chunk_metadata = [
            {
                **metadata,
                "doc_id": doc_id,
                "chunk_index": i,
                "chunk_start": chunk.start,
                "chunk_end": chunk.end
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # Store
        self.vector_store.add(chunk_ids, embeddings, chunk_texts, chunk_metadata)
        return len(chunks)
```

**Design Decisions:**
- Direct service composition in __init__
- No service container or registry
- Async methods for consistency with UI
- Simple return values (not wrapped results)

## Implementation Strategy

### Phase 1: Create New Simplified Structure
1. Create new `vector_store.py` module
2. Create `backends/` directory for embedding implementations
3. Write simplified `embeddings_service.py`
4. Create consolidated `rag_service.py`

### Phase 2: Migrate Existing Functionality
1. Port embedding creation logic to new structure
2. Migrate ChromaDB operations to vector_store module
3. Update indexing logic for new interfaces
4. Ensure all search modes work

### Phase 3: Update Integration Points
1. Update `SearchRAGWindow.py` to use new service
2. Modify event handlers for new interfaces
3. Update configuration handling
4. Ensure backward compatibility for configs

### Phase 4: Remove Old Code
1. Delete old provider system
2. Remove factory classes
3. Delete unused memory management complexity
4. Clean up duplicate implementations

## Migration Guide

### For Users
No changes required - all configuration options remain the same:

```toml
[rag]
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
vector_store_type = "chroma"
chunk_size = 1000
chunk_overlap = 200
```

### For Developers

#### Old Usage
```python
# Complex initialization
factory = RAGServiceFactory()
services = factory.create_complete_rag_services(config)
embeddings_service = services['embeddings']
```

#### New Usage
```python
# Simple, direct initialization
rag_service = RAGService(config)
# Or access individual services
embeddings = rag_service.embeddings
```

## Benefits Summary

### Code Reduction
- **Lines of Code**: ~2,500 → ~800 (68% reduction)
- **Number of Classes**: 15+ → 6
- **Abstraction Levels**: 5 → 2

### Maintenance Benefits
- Easier to understand and debug
- Clear flow from user action to result
- Fewer edge cases and error conditions
- Simpler test scenarios

### Performance Benefits
- Less overhead from abstractions
- Direct method calls instead of provider lookups
- Simpler initialization process

### Preserved Flexibility
- Can still change embedding models via config
- Can still switch vector stores via config
- Can add new backends easily
- Maintains module isolation for testing

## Future Considerations

### Adding New Embedding Backends
1. Create new file in `backends/` directory
2. Implement simple encode interface
3. Add model name pattern to `_load_model`

### Adding New Vector Stores
1. Implement VectorStore protocol
2. Add to `create_vector_store` function
3. Update configuration options

### Potential Future Enhancements
- Async embedding creation for true parallelism
- Streaming indexing for large documents
- Incremental index updates
- Multi-modal embeddings (text + image)

## Conclusion

This simplification maintains all essential functionality while removing unnecessary complexity. The modular design is preserved, configuration flexibility is maintained, and the code becomes much more maintainable.

## Detailed Implementation Examples

### Updated Plan: Use Existing Embeddings_Lib.py

Since we already have a robust `Embeddings_Lib.py` implementation, we'll use it instead of creating a new embeddings service. The existing library provides:

- **Multiple Providers**: HuggingFace and OpenAI support with extensibility
- **Thread-Safe Caching**: LRU cache with idle eviction
- **Async Support**: Both sync and async methods
- **Dynamic Model Loading**: Can load models on-demand
- **Proper Resource Management**: Handles GPU memory and model cleanup

### Integration with Existing Embeddings_Lib.py

```python
# embeddings_wrapper.py - Thin wrapper around Embeddings_Lib for RAG service
"""
Wrapper to adapt the existing Embeddings_Lib.py to the simplified RAG interface.
This maintains the clean API while leveraging the robust existing implementation.
"""
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from ...Embeddings.Embeddings_Lib import EmbeddingFactory, EmbeddingConfigSchema

logger = logging.getLogger(__name__)

class EmbeddingsServiceWrapper:
    """
    Wrapper around EmbeddingFactory to provide simplified interface for RAG.
    
    This allows us to:
    - Use the existing robust Embeddings_Lib.py
    - Provide a simpler interface for RAG use cases
    - Add RAG-specific features like result caching
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_size: int = 2,  # Number of models to cache
                 device: Optional[str] = None):
        """
        Initialize embeddings service using existing EmbeddingFactory.
        
        Args:
            model_name: Model identifier (can be HF model or "openai/text-embedding-3-small")
            cache_size: Number of models to keep in memory
            device: Device to use (cpu, cuda, mps)
        """
        self.model_name = model_name
        self.device = device
        
        # Determine provider from model name
        if model_name.startswith("openai/"):
            provider = "openai"
            model_path = model_name.split("/", 1)[1]
        else:
            provider = "huggingface"
            model_path = model_name
        
        # Build configuration for EmbeddingFactory
        config = {
            "default_model_id": "default",
            "models": {
                "default": self._build_model_config(provider, model_path)
            }
        }
        
        # Initialize factory
        self.factory = EmbeddingFactory(
            cfg=config,
            max_cached=cache_size,
            idle_seconds=900,  # 15 minutes
            allow_dynamic_hf=True  # Allow loading models not in config
        )
        
        # Metrics
        self._embeddings_created = 0
    
    def _build_model_config(self, provider: str, model_path: str) -> Dict[str, Any]:
        """Build model configuration for EmbeddingFactory."""
        if provider == "openai":
            return {
                "provider": "openai",
                "model_name_or_path": model_path,
                # API key will be picked up from environment
            }
        else:
            return {
                "provider": "huggingface",
                "model_name_or_path": model_path,
                "device": self.device,
                "trust_remote_code": False,
                "max_length": 512,
                "batch_size": 32
            }
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for texts using the configured model.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.factory.embed(texts, as_list=False)
            self._embeddings_created += len(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise
    
    async def create_embeddings_async(self, texts: List[str]) -> np.ndarray:
        """Async version of create_embeddings."""
        if not texts:
            return np.array([])
        
        embeddings = await self.factory.async_embed(texts, as_list=False)
        self._embeddings_created += len(texts)
        return embeddings
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return {
            "model_name": self.model_name,
            "total_embeddings_created": self._embeddings_created,
            "factory_config": self.factory.config.dict()
        }
    
    def clear_cache(self):
        """Clear the model cache."""
        # EmbeddingFactory doesn't expose cache clearing directly
        # but we can close and reinitialize
        self.factory.close()
        self.__init__(self.model_name, self.factory._max_cached, self.device)
    
    def close(self):
        """Clean up resources."""
        self.factory.close()

# For backward compatibility
EmbeddingsService = EmbeddingsServiceWrapper
import hashlib
import json
from collections import OrderedDict

class LRUCache:
    """Simple LRU cache implementation"""
    def __init__(self, maxsize: int):
        self.cache = OrderedDict()
        self.maxsize = maxsize
    
    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def __setitem__(self, key: str, value: np.ndarray):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            # Remove least recently used
            self.cache.popitem(last=False)

class EmbeddingsService:
    """
    Simplified embeddings service focusing on what's actually used.
    
    Key simplifications:
    - No provider registration/switching
    - Direct model loading based on name
    - Built-in caching without separate service
    - Simple batch processing without over-engineering
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_size: int = 1000,
                 device: str = "cpu",
                 batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = self._load_model(model_name)
        self.cache = LRUCache(maxsize=cache_size)
        
        # Simple metrics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_embeddings_created = 0
    
    def _load_model(self, model_name: str):
        """
        Load model based on naming convention.
        This replaces the complex provider registration system.
        """
        try:
            if model_name.startswith("openai/"):
                from .backends.openai import OpenAIBackend
                api_key = self._get_api_key("OPENAI_API_KEY")
                return OpenAIBackend(model_name.split("/", 1)[1], api_key)
            
            elif model_name.startswith("cohere/"):
                from .backends.cohere import CohereBackend
                api_key = self._get_api_key("COHERE_API_KEY")
                return CohereBackend(model_name.split("/", 1)[1], api_key)
            
            elif model_name.startswith("hf/"):
                from .backends.huggingface import HuggingFaceBackend
                return HuggingFaceBackend(model_name.split("/", 1)[1])
            
            else:
                # Default to sentence-transformers
                from .backends.sentence_transformer import SentenceTransformerBackend
                return SentenceTransformerBackend(model_name, self.device)
                
        except ImportError as e:
            # Graceful fallback if optional dependencies missing
            logging.warning(f"Failed to load {model_name}: {e}. Falling back to default.")
            from .backends.sentence_transformer import SentenceTransformerBackend
            return SentenceTransformerBackend("all-MiniLM-L6-v2", self.device)
    
    def _get_api_key(self, env_var: str) -> Optional[str]:
        """Get API key from environment or config"""
        import os
        from ...config import get_cli_setting
        
        # Try environment first
        key = os.getenv(env_var)
        if key:
            return key
        
        # Try config
        provider = env_var.replace("_API_KEY", "").lower()
        return get_cli_setting(f"API.{provider}_api_key")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Include model name in cache key
        key_string = f"{self.model_name}:{text}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings with caching and batching.
        
        This replaces the complex provider-based implementation
        with a simple, direct approach.
        """
        if not texts:
            return np.array([])
        
        # Check cache
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached = self.cache.get(cache_key)
            
            if cached is not None:
                results[i] = cached
                self._cache_hits += 1
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                self._cache_misses += 1
        
        # Embed missing texts
        if texts_to_embed:
            # Simple batching
            embeddings = []
            for i in range(0, len(texts_to_embed), self.batch_size):
                batch = texts_to_embed[i:i + self.batch_size]
                batch_embeddings = self.model.encode(batch)
                embeddings.extend(batch_embeddings)
            
            # Update cache and results
            for idx, text, embedding in zip(indices_to_embed, texts_to_embed, embeddings):
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = embedding
                results[idx] = embedding
            
            self._total_embeddings_created += len(texts_to_embed)
        
        return np.array(results)
    
    def get_metrics(self) -> dict:
        """Simple metrics for monitoring"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "model_name": self.model_name,
            "cache_size": self.cache.maxsize,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": hit_rate,
            "total_embeddings_created": self._total_embeddings_created
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache = LRUCache(maxsize=self.cache.maxsize)
        self._cache_hits = 0
        self._cache_misses = 0
```

### Embedding Backend Example

```python
# backends/sentence_transformer.py
import numpy as np
from typing import List, Union
import logging

class SentenceTransformerBackend:
    """
    Sentence Transformers backend - our default and most used backend.
    
    This replaces the complex provider system with a simple,
    direct implementation.
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
    
    @property
    def model(self):
        """Lazy load model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                
                # Optimize for inference
                self._model.eval()
                if hasattr(self._model, 'half') and self.device != "cpu":
                    self._model.half()  # Use FP16 on GPU
                    
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Simple interface that all backends must implement.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Direct encoding without complex batching
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        return embeddings
```

## Citations Support Implementation

### Overview

Citations support allows RAG search results to include precise references to source documents, including:
- Document ID and metadata
- Exact text snippets that matched
- Character/word offsets in the original document
- Confidence scores for each citation

### Citation Data Model

```python
# citations.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

class CitationType(Enum):
    """Type of citation match"""
    EXACT = "exact"          # Exact phrase match
    SEMANTIC = "semantic"    # Semantic similarity
    FUZZY = "fuzzy"         # Fuzzy/partial match

@dataclass
class Citation:
    """
    Represents a citation to a source document.
    
    Attributes:
        document_id: Unique identifier of the source document
        document_title: Human-readable title of the document
        chunk_id: ID of the specific chunk within the document
        text: The actual text snippet being cited
        start_char: Character offset in the original document
        end_char: End character offset in the original document
        confidence: Confidence score (0-1) for this citation
        match_type: Type of match that produced this citation
        metadata: Additional metadata (author, date, URL, etc.)
    """
    document_id: str
    document_title: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    confidence: float
    match_type: CitationType
    metadata: Dict[str, Any]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "document_title": self.document_title,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "match_type": self.match_type.value,
            "metadata": self.metadata
        }
    
    def format_citation(self, style: str = "inline") -> str:
        """
        Format citation for display.
        
        Args:
            style: Citation style (inline, footnote, academic)
            
        Returns:
            Formatted citation string
        """
        if style == "inline":
            return f"[{self.document_title}, chars {self.start_char}-{self.end_char}]"
        elif style == "footnote":
            return f"{self.document_title}"
        elif style == "academic":
            author = self.metadata.get("author", "Unknown")
            date = self.metadata.get("date", "n.d.")
            return f"({author}, {date})"
        else:
            return f"[{self.document_id}:{self.chunk_id}]"

@dataclass
class SearchResultWithCitations:
    """Enhanced search result that includes citations."""
    id: str
    score: float
    document: str
    metadata: dict
    citations: List[Citation]
    
    def get_unique_sources(self) -> List[str]:
        """Get list of unique source documents cited."""
        return list(set(c.document_id for c in self.citations))
    
    def format_with_citations(self, style: str = "inline") -> str:
        """
        Format the result with inline citations.
        
        Returns document text with citations inserted at appropriate points.
        """
        # This is a simplified version - in practice, you'd want to
        # intelligently insert citations at relevant points in the text
        citations_text = " ".join(c.format_citation(style) for c in self.citations)
        return f"{self.document} {citations_text}"
```

### Enhanced Vector Store with Citations

```python
# vector_store.py - Updated with citations support
import numpy as np
from typing import List, Dict, Optional, Protocol, Tuple
from pathlib import Path
import json
import logging
from dataclasses import dataclass

from .citations import Citation, CitationType, SearchResultWithCitations

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Basic search result (for backward compatibility)"""
    id: str
    score: float
    document: str
    metadata: dict

class VectorStore(Protocol):
    """
    Updated protocol with citations support.
    """
    def add(self, ids: List[str], embeddings: np.ndarray, 
            documents: List[str], metadata: List[dict]) -> None: ...
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 10) -> List[SearchResult]: ...
    
    def search_with_citations(self, query_embedding: np.ndarray,
                            query_text: str,
                            top_k: int = 10) -> List[SearchResultWithCitations]: ...
    
    def delete_collection(self, name: str) -> None: ...
    
    def get_collection_stats(self) -> dict: ...

class ChromaVectorStore:
    """
    ChromaDB implementation with citations support.
    """
    
    def __init__(self, persist_directory: Path, collection_name: str = "default"):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
    
    @property
    def client(self):
        """Lazy load ChromaDB client"""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                settings = Settings(
                    persist_directory=str(self.persist_directory),
                    anonymized_telemetry=False
                )
                self._client = chromadb.Client(settings)
                
            except ImportError:
                raise ImportError(
                    "ChromaDB not installed. "
                    "Install with: pip install chromadb"
                )
        return self._client
    
    @property
    def collection(self):
        """Get or create collection"""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection
    
    def add(self, ids: List[str], embeddings: np.ndarray, 
            documents: List[str], metadata: List[dict]) -> None:
        """
        Add documents to the collection.
        
        Metadata should include:
        - doc_id: Original document ID
        - doc_title: Human-readable document title
        - chunk_start: Character offset in original document
        - chunk_end: End character offset
        - chunk_index: Index of this chunk
        """
        if len(ids) == 0:
            return
        
        # Ensure embeddings are the right type
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata
            )
        except Exception as e:
            logger.error(f"Failed to add to ChromaDB: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 10) -> List[SearchResult]:
        """Basic search without citations (backward compatibility)"""
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    search_results.append(SearchResult(
                        id=results['ids'][0][i],
                        score=1 - results['distances'][0][i],
                        document=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] or {}
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_with_citations(self, query_embedding: np.ndarray,
                            query_text: str,
                            top_k: int = 10) -> List[SearchResultWithCitations]:
        """
        Enhanced search that includes citations.
        
        Args:
            query_embedding: Query vector
            query_text: Original query text (used for citation matching)
            top_k: Number of results
            
        Returns:
            List of results with citations
        """
        # First, get basic search results
        basic_results = self.search(query_embedding, top_k * 2)  # Get more for citation filtering
        
        # Convert to results with citations
        results_with_citations = []
        
        for result in basic_results[:top_k]:
            # Extract citation information from metadata
            citations = []
            
            # Create semantic citation for the chunk
            citation = Citation(
                document_id=result.metadata.get("doc_id", result.id),
                document_title=result.metadata.get("doc_title", "Unknown Document"),
                chunk_id=result.id,
                text=result.document[:200] + "..." if len(result.document) > 200 else result.document,
                start_char=result.metadata.get("chunk_start", 0),
                end_char=result.metadata.get("chunk_end", len(result.document)),
                confidence=result.score,
                match_type=CitationType.SEMANTIC,
                metadata={
                    "author": result.metadata.get("author"),
                    "date": result.metadata.get("date"),
                    "url": result.metadata.get("url"),
                    "chunk_index": result.metadata.get("chunk_index", 0)
                }
            )
            citations.append(citation)
            
            # If we have keyword matches, add exact citations
            if "keyword_matches" in result.metadata:
                for match in result.metadata["keyword_matches"]:
                    exact_citation = Citation(
                        document_id=result.metadata.get("doc_id", result.id),
                        document_title=result.metadata.get("doc_title", "Unknown Document"),
                        chunk_id=result.id,
                        text=match["text"],
                        start_char=match["start"],
                        end_char=match["end"],
                        confidence=1.0,  # Exact match
                        match_type=CitationType.EXACT,
                        metadata=citation.metadata
                    )
                    citations.append(exact_citation)
            
            # Create result with citations
            result_with_citations = SearchResultWithCitations(
                id=result.id,
                score=result.score,
                document=result.document,
                metadata=result.metadata,
                citations=citations
            )
            results_with_citations.append(result_with_citations)
        
        return results_with_citations
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection"""
        try:
            self.client.delete_collection(name)
            if name == self.collection_name:
                self._collection = None
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
    
    def get_collection_stats(self) -> dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "persist_directory": str(self.persist_directory)
            }
        except:
            return {"name": self.collection_name, "count": 0}
import logging
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Simple result structure"""
    id: str
    score: float
    document: str
    metadata: dict

class VectorStore(Protocol):
    """
    Simple protocol defining vector store interface.
    Using Protocol instead of ABC for lighter weight.
    """
    def add(self, ids: List[str], embeddings: np.ndarray, 
            documents: List[str], metadata: List[dict]) -> None: ...
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 10) -> List[SearchResult]: ...
    
    def delete_collection(self, name: str) -> None: ...
    
    def get_collection_stats(self) -> dict: ...

class ChromaVectorStore:
    """
    ChromaDB implementation - our primary vector store.
    
    Simplified from the abstract implementation:
    - Direct ChromaDB usage
    - No update_documents (just delete + add)
    - Simple error handling
    """
    
    def __init__(self, persist_directory: Path, collection_name: str = "default"):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
    
    @property
    def client(self):
        """Lazy load ChromaDB client"""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                settings = Settings(
                    persist_directory=str(self.persist_directory),
                    anonymized_telemetry=False
                )
                self._client = chromadb.Client(settings)
                
            except ImportError:
                raise ImportError(
                    "ChromaDB not installed. "
                    "Install with: pip install chromadb"
                )
        return self._client
    
    @property
    def collection(self):
        """Get or create collection"""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        return self._collection
    
    def add(self, ids: List[str], embeddings: np.ndarray, 
            documents: List[str], metadata: List[dict]) -> None:
        """Add documents to the collection"""
        if len(ids) == 0:
            return
        
        # Ensure embeddings are the right type
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata
            )
        except Exception as e:
            logging.error(f"Failed to add to ChromaDB: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 10) -> List[SearchResult]:
        """Search for similar documents"""
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to SearchResult objects
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    search_results.append(SearchResult(
                        id=results['ids'][0][i],
                        score=1 - results['distances'][0][i],  # Convert distance to similarity
                        document=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] or {}
                    ))
            
            return search_results
            
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return []
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection"""
        try:
            self.client.delete_collection(name)
            if name == self.collection_name:
                self._collection = None
        except Exception as e:
            logging.error(f"Failed to delete collection: {e}")
    
    def get_collection_stats(self) -> dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "persist_directory": str(self.persist_directory)
            }
        except:
            return {"name": self.collection_name, "count": 0}

class InMemoryVectorStore:
    """
    Simple in-memory vector store for testing or fallback.
    
    Much simpler than ChromaDB but implements the same interface.
    """
    
    def __init__(self):
        self.ids: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.documents: List[str] = []
        self.metadata: List[dict] = []
    
    def add(self, ids: List[str], embeddings: np.ndarray, 
            documents: List[str], metadata: List[dict]) -> None:
        """Add documents to memory"""
        for i, id_val in enumerate(ids):
            if id_val not in self.ids:
                self.ids.append(id_val)
                self.embeddings.append(embeddings[i])
                self.documents.append(documents[i])
                self.metadata.append(metadata[i])
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 10) -> List[SearchResult]:
        """Search using cosine similarity"""
        if not self.embeddings:
            return []
        
        # Compute similarities
        similarities = []
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        for i, emb in enumerate(self.embeddings):
            emb_norm = emb / np.linalg.norm(emb)
            similarity = np.dot(query_norm, emb_norm)
            similarities.append((i, similarity))
        
        # Sort and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Convert to SearchResult
        results = []
        for idx, score in top_results:
            results.append(SearchResult(
                id=self.ids[idx],
                score=score,
                document=self.documents[idx],
                metadata=self.metadata[idx]
            ))
        
        return results
    
    def delete_collection(self, name: str) -> None:
        """Clear all data"""
        self.ids.clear()
        self.embeddings.clear()
        self.documents.clear()
        self.metadata.clear()
    
    def get_collection_stats(self) -> dict:
        """Get stats"""
        return {"count": len(self.ids), "type": "in_memory"}

# Factory function - much simpler than factory class
def create_vector_store(store_type: str, 
                       persist_directory: Optional[Path] = None,
                       collection_name: str = "default") -> VectorStore:
    """
    Simple factory function to create vector stores.
    
    This replaces the complex factory pattern with a simple function.
    """
    if store_type == "chroma":
        if persist_directory is None:
            raise ValueError("persist_directory required for ChromaDB")
        return ChromaVectorStore(persist_directory, collection_name)
    
    elif store_type == "memory":
        return InMemoryVectorStore()
    
    else:
        # Default to in-memory if unknown type
        logging.warning(f"Unknown vector store type: {store_type}. Using in-memory.")
        return InMemoryVectorStore()
```

## Key Design Decisions and Rationale

### 1. Protocol vs Abstract Base Class

**Decision**: Use `typing.Protocol` for VectorStore interface instead of ABC.

**Rationale**:
- Protocols are more Pythonic and lightweight
- No need to inherit from a base class
- Duck typing - if it has the methods, it's a VectorStore
- Easier to test with mock objects
- Follows modern Python practices

### 2. Lazy Loading Pattern

**Decision**: Use `@property` decorators for lazy loading models and clients.

**Rationale**:
- Faster startup time - models only loaded when needed
- Better resource management
- Allows graceful degradation if dependencies missing
- Simplifies initialization code

### 3. Simple Cache Implementation

**Decision**: Implement a basic LRU cache instead of using a separate cache service.

**Rationale**:
- The separate cache service added unnecessary complexity
- Python's OrderedDict makes LRU trivial to implement
- Keeps caching close to where it's used
- Easier to understand and debug

### 4. Configuration via Model Names

**Decision**: Use model name prefixes (e.g., "openai/", "hf/") to determine backend.

**Rationale**:
- Simple and intuitive for users
- No need for separate provider configuration
- Self-documenting in config files
- Follows conventions from other tools

### 5. Direct Error Messages

**Decision**: Provide specific, actionable error messages for missing dependencies.

**Rationale**:
- Users immediately know how to fix issues
- Reduces support burden
- Better developer experience
- Clear upgrade path for optional features

## Testing Strategy

### Unit Tests

```python
# test_embeddings_service.py
import pytest
import numpy as np
from rag_search.services.embeddings_service import EmbeddingsService

class TestEmbeddingsService:
    def test_create_embeddings_basic(self):
        """Test basic embedding creation"""
        service = EmbeddingsService(model_name="sentence-transformers/all-MiniLM-L6-v2")
        texts = ["Hello world", "How are you?"]
        embeddings = service.create_embeddings(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 384  # Model dimension
    
    def test_caching(self):
        """Test that caching works correctly"""
        service = EmbeddingsService(cache_size=10)
        
        # First call - cache miss
        texts = ["Test text"]
        embeddings1 = service.create_embeddings(texts)
        assert service._cache_misses == 1
        assert service._cache_hits == 0
        
        # Second call - cache hit
        embeddings2 = service.create_embeddings(texts)
        assert service._cache_hits == 1
        assert np.array_equal(embeddings1, embeddings2)
    
    def test_different_backends(self):
        """Test loading different backends"""
        # This test would be skipped if backends not available
        models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "openai/text-embedding-3-small",
            "hf/BAAI/bge-small-en"
        ]
        
        for model in models:
            try:
                service = EmbeddingsService(model_name=model)
                # Should not raise an error
                assert service.model_name == model
            except ImportError:
                pytest.skip(f"Backend for {model} not available")
```

### Integration Tests

```python
# test_rag_integration.py
import pytest
from pathlib import Path
import tempfile
from rag_search.services.rag_service import RAGService
from rag_search.services.config import RAGConfig

class TestRAGIntegration:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_full_workflow(self, temp_dir):
        """Test complete RAG workflow"""
        config = RAGConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_store_type="chroma",
            persist_directory=temp_dir,
            chunk_size=100,
            chunk_overlap=20
        )
        
        service = RAGService(config)
        
        # Index a document
        doc_id = "test_doc"
        content = "This is a test document. " * 50  # Make it long enough to chunk
        chunks_created = service.index_document(doc_id, content)
        assert chunks_created > 1
        
        # Search
        results = service.search("test document", top_k=5)
        assert len(results) > 0
        assert results[0].metadata["doc_id"] == doc_id
```

## Performance Considerations

### Memory Usage

The simplified design actually improves memory usage:

1. **No Provider Registry**: Eliminates memory overhead of maintaining provider instances
2. **Direct Model Loading**: Only one model in memory at a time
3. **Simple Caching**: OrderedDict is more memory efficient than complex cache service
4. **Lazy Loading**: Models and clients only loaded when needed

### Speed Improvements

1. **Fewer Abstraction Layers**: Direct calls instead of provider lookups
2. **No Lock Contention**: Removed thread-safe provider switching
3. **Simpler Cache**: Faster cache lookups without service overhead
4. **Direct Numpy**: Avoid conversion overhead from wrapper objects

## Migration Checklist

- [ ] Create new simplified structure
- [ ] Port embeddings functionality
- [ ] Port vector store functionality  
- [ ] Update RAG service coordinator
- [ ] Update UI integration points
- [ ] Update event handlers
- [ ] Migrate configuration handling
- [ ] Add backward compatibility shims
- [ ] Update tests
- [ ] Remove old code
- [ ] Update documentation

## Complete RAG Service Implementation

### Main RAG Service Coordinator

```python
# rag_service.py
import asyncio
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
import logging
from dataclasses import dataclass
import time

from .embeddings_service import EmbeddingsService
from .vector_store import create_vector_store, VectorStore, SearchResult
from .chunking_service import ChunkingService, ChunkingStrategy, Chunk
from ..config import RAGConfig

logger = logging.getLogger(__name__)

@dataclass
class IndexingResult:
    """Result of indexing operation"""
    doc_id: str
    chunks_created: int
    time_taken: float
    success: bool
    error: Optional[str] = None

class RAGService:
    """
    Main RAG service coordinating all components.
    
    This simplified version:
    - Direct composition without factories
    - Clear initialization flow
    - Simple async/sync method variants
    - Built-in error handling
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG service with configuration.
        
        Args:
            config: RAG configuration or None to load from settings
        """
        self.config = config or RAGConfig.from_settings()
        
        # Initialize services directly - no factory needed
        try:
            self.embeddings = EmbeddingsService(
                model_name=self.config.embedding_model,
                cache_size=self.config.embedding_cache_size,
                device=self.config.device,
                batch_size=self.config.batch_size
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            # Fallback to simple model
            self.embeddings = EmbeddingsService(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
        
        # Initialize vector store
        self.vector_store = create_vector_store(
            store_type=self.config.vector_store_type,
            persist_directory=self.config.persist_directory,
            collection_name=self.config.collection_name
        )
        
        # Initialize chunking service
        self.chunking = ChunkingService(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            chunking_strategy=self.config.chunking_strategy,
            min_chunk_size=self.config.min_chunk_size
        )
        
        # Simple metrics
        self._docs_indexed = 0
        self._searches_performed = 0
        self._last_index_time = None
    
    # === Indexing Methods ===
    
    async def index_document(self, 
                           doc_id: str, 
                           content: str,
                           metadata: Optional[Dict[str, Any]] = None,
                           chunking_strategy: Optional[ChunkingStrategy] = None
                           ) -> IndexingResult:
        """
        Index a document asynchronously.
        
        Args:
            doc_id: Unique document identifier
            content: Document content to index
            metadata: Optional metadata to store with chunks
            chunking_strategy: Override default chunking strategy
            
        Returns:
            IndexingResult with status and statistics
        """
        start_time = time.time()
        metadata = metadata or {}
        
        try:
            # Chunk the document
            chunks = await self._chunk_document(content, chunking_strategy)
            if not chunks:
                return IndexingResult(
                    doc_id=doc_id,
                    chunks_created=0,
                    time_taken=time.time() - start_time,
                    success=True
                )
            
            # Create embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await self._create_embeddings_async(chunk_texts)
            
            # Prepare for storage
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_metadata = [
                {
                    **metadata,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "chunk_start": chunk.start,
                    "chunk_end": chunk.end,
                    "chunk_size": len(chunk.text)
                }
                for i, chunk in enumerate(chunks)
            ]
            
            # Store in vector database
            await self._store_chunks(chunk_ids, embeddings, chunk_texts, chunk_metadata)
            
            # Update metrics
            self._docs_indexed += 1
            self._last_index_time = time.time()
            
            return IndexingResult(
                doc_id=doc_id,
                chunks_created=len(chunks),
                time_taken=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to index document {doc_id}: {e}")
            return IndexingResult(
                doc_id=doc_id,
                chunks_created=0,
                time_taken=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def index_document_sync(self, doc_id: str, content: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> IndexingResult:
        """Synchronous version of index_document"""
        return asyncio.run(self.index_document(doc_id, content, metadata))
    
    async def index_batch(self, 
                         documents: List[Dict[str, Any]],
                         show_progress: bool = True) -> List[IndexingResult]:
        """
        Index multiple documents in batch.
        
        Args:
            documents: List of dicts with 'id', 'content', and optional 'metadata'
            show_progress: Whether to log progress
            
        Returns:
            List of IndexingResult for each document
        """
        results = []
        total = len(documents)
        
        for i, doc in enumerate(documents):
            if show_progress and i % 10 == 0:
                logger.info(f"Indexing progress: {i}/{total} documents")
            
            result = await self.index_document(
                doc_id=doc['id'],
                content=doc['content'],
                metadata=doc.get('metadata')
            )
            results.append(result)
        
        if show_progress:
            logger.info(f"Indexed {total} documents")
        
        return results
    
    # === Search Methods ===
    
    async def search(self,
                    query: str,
                    top_k: int = 10,
                    search_type: Literal["semantic", "hybrid", "keyword"] = "semantic",
                    filter_metadata: Optional[Dict[str, Any]] = None,
                    rerank: bool = False) -> List[SearchResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            search_type: Type of search to perform
            filter_metadata: Metadata filters to apply
            rerank: Whether to rerank results
            
        Returns:
            List of search results
        """
        self._searches_performed += 1
        
        if search_type == "semantic":
            return await self._semantic_search(query, top_k, filter_metadata)
        elif search_type == "hybrid":
            return await self._hybrid_search(query, top_k, filter_metadata)
        elif search_type == "keyword":
            return await self._keyword_search(query, top_k, filter_metadata)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    async def _semantic_search(self, 
                              query: str, 
                              top_k: int,
                              filter_metadata: Optional[Dict[str, Any]] = None
                              ) -> List[SearchResult]:
        """Perform semantic similarity search"""
        # Create query embedding
        query_embedding = await self._create_embeddings_async([query])
        query_embedding = query_embedding[0]
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k * 2)
        
        # Apply metadata filters if provided
        if filter_metadata:
            results = [
                r for r in results
                if all(r.metadata.get(k) == v for k, v in filter_metadata.items())
            ]
        
        return results[:top_k]
    
    async def _keyword_search(self,
                             query: str,
                             top_k: int,
                             filter_metadata: Optional[Dict[str, Any]] = None
                             ) -> List[SearchResult]:
        """
        Perform keyword search using FTS5.
        This would integrate with the SQLite FTS5 search.
        """
        # Placeholder - would integrate with actual FTS5 implementation
        logger.warning("Keyword search not yet implemented in simplified version")
        return []
    
    async def _hybrid_search(self,
                            query: str,
                            top_k: int,
                            filter_metadata: Optional[Dict[str, Any]] = None
                            ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword.
        """
        # Get results from both search types
        semantic_results = await self._semantic_search(query, top_k * 2, filter_metadata)
        keyword_results = await self._keyword_search(query, top_k * 2, filter_metadata)
        
        # Simple merging strategy - could be improved with proper scoring
        seen_ids = set()
        merged_results = []
        
        # Interleave results
        for semantic, keyword in zip(semantic_results, keyword_results):
            if semantic and semantic.id not in seen_ids:
                merged_results.append(semantic)
                seen_ids.add(semantic.id)
            
            if keyword and keyword.id not in seen_ids:
                merged_results.append(keyword)
                seen_ids.add(keyword.id)
        
        # Add any remaining
        for result in semantic_results + keyword_results:
            if result.id not in seen_ids and len(merged_results) < top_k:
                merged_results.append(result)
                seen_ids.add(result.id)
        
        return merged_results[:top_k]
    
    # === Helper Methods ===
    
    async def _chunk_document(self, 
                            content: str,
                            strategy: Optional[ChunkingStrategy] = None
                            ) -> List[Chunk]:
        """Chunk document asynchronously"""
        # Chunking is typically CPU-bound, so we run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chunking.chunk_text,
            content,
            strategy
        )
    
    async def _create_embeddings_async(self, texts: List[str]) -> np.ndarray:
        """Create embeddings asynchronously"""
        # Embedding creation might be CPU/GPU bound
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.embeddings.create_embeddings,
            texts
        )
    
    async def _store_chunks(self,
                          ids: List[str],
                          embeddings: np.ndarray,
                          documents: List[str],
                          metadata: List[dict]) -> None:
        """Store chunks in vector database asynchronously"""
        # Vector store operations might involve I/O
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.vector_store.add,
            ids, embeddings, documents, metadata
        )
    
    # === Management Methods ===
    
    def clear_cache(self):
        """Clear all caches"""
        self.embeddings.clear_cache()
        logger.info("Cleared embeddings cache")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            "embeddings_metrics": self.embeddings.get_metrics(),
            "vector_store_stats": self.vector_store.get_collection_stats(),
            "service_metrics": {
                "documents_indexed": self._docs_indexed,
                "searches_performed": self._searches_performed,
                "last_index_time": self._last_index_time
            },
            "config": {
                "embedding_model": self.config.embedding_model,
                "vector_store_type": self.config.vector_store_type,
                "chunk_size": self.config.chunk_size
            }
        }
    
    def delete_collection(self, name: str):
        """Delete a collection from vector store"""
        self.vector_store.delete_collection(name)
        logger.info(f"Deleted collection: {name}")
```

### Configuration Handling

```python
# config.py
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
import os
import toml
import logging

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """
    Simplified RAG configuration.
    
    All settings have sensible defaults and can be overridden via:
    1. Environment variables (RAG_*)
    2. Config file (config.toml)
    3. Direct instantiation
    """
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_size: int = 1000
    device: str = "cpu"  # or "cuda" or "mps"
    batch_size: int = 32
    
    # Vector store settings
    vector_store_type: str = "chroma"
    persist_directory: Optional[Path] = None
    collection_name: str = "default"
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    
    # Search settings
    default_top_k: int = 10
    rerank_enabled: bool = False
    reranker_model: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set defaults"""
        # Set default persist directory if not provided
        if self.persist_directory is None:
            self.persist_directory = Path.home() / ".local" / "share" / "tldw_cli" / "chromadb"
        else:
            self.persist_directory = Path(self.persist_directory)
        
        # Create directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Validate device
        if self.device not in ["cpu", "cuda", "mps"]:
            logger.warning(f"Unknown device: {self.device}. Using CPU.")
            self.device = "cpu"
    
    @classmethod
    def from_settings(cls) -> "RAGConfig":
        """
        Load configuration from environment and config file.
        
        Priority order:
        1. Environment variables (highest)
        2. Config file
        3. Defaults (lowest)
        """
        config = cls()
        
        # Load from config file if exists
        config_path = Path.home() / ".config" / "tldw_cli" / "config.toml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    toml_config = toml.load(f)
                    rag_section = toml_config.get("rag", {})
                    
                    # Update with config file values
                    if "embedding_model" in rag_section:
                        config.embedding_model = rag_section["embedding_model"]
                    if "chunk_size" in rag_section:
                        config.chunk_size = rag_section["chunk_size"]
                    if "chunk_overlap" in rag_section:
                        config.chunk_overlap = rag_section["chunk_overlap"]
                    if "device" in rag_section:
                        config.device = rag_section["device"]
                    if "vector_store_type" in rag_section:
                        config.vector_store_type = rag_section["vector_store_type"]
                        
            except Exception as e:
                logger.error(f"Failed to load RAG config from file: {e}")
        
        # Override with environment variables
        if os.getenv("RAG_EMBEDDING_MODEL"):
            config.embedding_model = os.getenv("RAG_EMBEDDING_MODEL")
        if os.getenv("RAG_DEVICE"):
            config.device = os.getenv("RAG_DEVICE")
        if os.getenv("RAG_CHUNK_SIZE"):
            config.chunk_size = int(os.getenv("RAG_CHUNK_SIZE"))
        if os.getenv("RAG_VECTOR_STORE"):
            config.vector_store_type = os.getenv("RAG_VECTOR_STORE")
        
        return config
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "embedding_model": self.embedding_model,
            "embedding_cache_size": self.embedding_cache_size,
            "device": self.device,
            "batch_size": self.batch_size,
            "vector_store_type": self.vector_store_type,
            "persist_directory": str(self.persist_directory),
            "collection_name": self.collection_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "chunking_strategy": self.chunking_strategy.value,
            "default_top_k": self.default_top_k,
            "rerank_enabled": self.rerank_enabled,
            "reranker_model": self.reranker_model
        }
```

### UI Integration Example

```python
# Updated SearchRAGWindow integration
from tldw_chatbook.RAG_Search.Services.rag_service import RAGService
from tldw_chatbook.RAG_Search.Services.config import RAGConfig

class SearchRAGWindow(Container):
    """Updated to use simplified RAG service"""
    
    def compose(self) -> ComposeResult:
        # ... existing UI code ...
        pass
    
    def on_mount(self) -> None:
        """Initialize RAG service on mount"""
        try:
            # Load config from settings
            config = RAGConfig.from_settings()
            
            # Allow UI overrides
            if self.selected_embedding_model:
                config.embedding_model = self.selected_embedding_model
            
            # Initialize service - much simpler!
            self.rag_service = RAGService(config)
            
            # Get initial metrics
            metrics = self.rag_service.get_metrics()
            self.update_status(f"RAG initialized with {metrics['config']['embedding_model']}")
            
        except Exception as e:
            self.notify(f"Failed to initialize RAG: {e}", severity="error")
            # Fallback to basic service
            self.rag_service = RAGService()
    
    async def perform_search(self, query: str) -> None:
        """Perform RAG search with new simplified API"""
        try:
            # Get search parameters from UI
            search_type = self.search_type_select.value
            top_k = int(self.top_k_input.value)
            
            # Build metadata filter from UI selections
            filter_metadata = {}
            if self.source_filter.value != "all":
                filter_metadata["source"] = self.source_filter.value
            
            # Perform search - clean async API
            results = await self.rag_service.search(
                query=query,
                top_k=top_k,
                search_type=search_type,
                filter_metadata=filter_metadata,
                rerank=self.rerank_enabled.value
            )
            
            # Update UI with results
            await self.display_results(results)
            
        except Exception as e:
            self.notify(f"Search failed: {e}", severity="error")
```

### Backward Compatibility Shim

```python
# compat.py - Temporary compatibility layer during migration
"""
Compatibility shim for old RAG API.

This allows gradual migration of existing code.
"""

from .rag_service import RAGService
from .embeddings_service import EmbeddingsService
import warnings

class LegacyRAGServiceFactory:
    """Compatibility wrapper for old factory pattern"""
    
    def __init__(self):
        warnings.warn(
            "RAGServiceFactory is deprecated. Use RAGService directly.",
            DeprecationWarning,
            stacklevel=2
        )
    
    def create_complete_rag_services(self, config=None):
        """Mimics old factory method"""
        service = RAGService(config)
        return {
            'embeddings': service.embeddings,
            'vector_store': service.vector_store,
            'chunking': service.chunking,
            'rag': service
        }

# Alias for backward compatibility
RAGServiceFactory = LegacyRAGServiceFactory

class LegacyEmbeddingsService(EmbeddingsService):
    """Compatibility wrapper for old embeddings API"""
    
    def add_provider(self, provider_id, provider):
        """No-op for compatibility"""
        warnings.warn(
            "add_provider is deprecated. Providers are now selected via model_name.",
            DeprecationWarning
        )
    
    def set_provider(self, provider_id):
        """No-op for compatibility"""
        warnings.warn(
            "set_provider is deprecated. Use a new EmbeddingsService instance.",
            DeprecationWarning
        )

# Usage in existing code would still work:
# factory = RAGServiceFactory()  # Shows deprecation warning
# services = factory.create_complete_rag_services()
# embeddings = services['embeddings']
```

### Error Handling Patterns

```python
# error_handling.py
from typing import Optional, TypeVar, Callable, Any
from functools import wraps
import logging

T = TypeVar('T')

class RAGError(Exception):
    """Base exception for RAG operations"""
    pass

class EmbeddingError(RAGError):
    """Error creating embeddings"""
    pass

class IndexingError(RAGError):
    """Error indexing documents"""
    pass

class SearchError(RAGError):
    """Error during search"""
    pass

def with_fallback(fallback_value: T, log_errors: bool = True):
    """
    Decorator to provide fallback values on error.
    
    Usage:
        @with_fallback(fallback_value=[])
        def search(...) -> List[Result]:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"{func.__name__} failed: {e}")
                return fallback_value
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"{func.__name__} failed: {e}")
                return fallback_value
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Example usage in RAG service
class RAGService:
    @with_fallback(fallback_value=[], log_errors=True)
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Search with automatic fallback to empty results on error"""
        # ... implementation
```

### Performance Optimizations

```python
# performance.py
import functools
import time
from typing import Callable, Any
import psutil
import logging

logger = logging.getLogger(__name__)

def measure_performance(func: Callable) -> Callable:
    """Decorator to measure function performance"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = await func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.debug(
                f"{func.__name__} - "
                f"Time: {end_time - start_time:.2f}s, "
                f"Memory: {end_memory - start_memory:+.1f}MB"
            )
            
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed after {time.time() - start_time:.2f}s")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Similar implementation for sync functions
        pass
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# Optimized batch processing
class OptimizedRAGService(RAGService):
    """Performance-optimized RAG service"""
    
    @measure_performance
    async def index_batch_optimized(self, documents: List[dict]) -> List[IndexingResult]:
        """
        Optimized batch indexing with parallel processing.
        """
        # Group documents by size for better batching
        small_docs = [d for d in documents if len(d['content']) < 1000]
        large_docs = [d for d in documents if len(d['content']) >= 1000]
        
        # Process small documents in larger batches
        small_results = []
        for i in range(0, len(small_docs), 10):
            batch = small_docs[i:i+10]
            # Create embeddings for entire batch at once
            contents = [d['content'] for d in batch]
            all_chunks = []
            
            for content in contents:
                chunks = await self._chunk_document(content)
                all_chunks.extend([c.text for c in chunks])
            
            # Single embedding call for all chunks
            if all_chunks:
                embeddings = await self._create_embeddings_async(all_chunks)
                # ... store results
        
        # Process large documents individually
        large_results = []
        for doc in large_docs:
            result = await self.index_document(doc['id'], doc['content'])
            large_results.append(result)
        
        return small_results + large_results
```

## Implementation Timeline

### Week 1: Core Structure
- Create new simplified directory structure
- Implement basic embeddings service
- Implement vector store module
- Write initial tests

### Week 2: Integration
- Implement main RAG service
- Update configuration handling
- Create backward compatibility layer
- Integration testing

### Week 3: Migration
- Update UI components
- Migrate event handlers
- Update existing tests
- Performance testing

### Week 4: Cleanup
- Remove old code
- Update documentation
- Final testing
- Release preparation

## Risk Mitigation

### Potential Risks and Mitigations

1. **Breaking Changes**
   - Risk: Existing code depends on old API
   - Mitigation: Compatibility shim layer, deprecation warnings

2. **Performance Regression**
   - Risk: Simplified code might be slower
   - Mitigation: Performance benchmarks, optimization pass

3. **Missing Features**
   - Risk: Some edge case features might be lost
   - Mitigation: Comprehensive testing, user feedback period

4. **Migration Complexity**
   - Risk: Complex migration for existing users
   - Mitigation: Automated migration scripts, clear documentation

## Success Metrics

### Quantitative Metrics
- **Code Reduction**: Target 70% fewer lines of code
- **Test Coverage**: Maintain >80% coverage
- **Performance**: No regression in indexing/search speed
- **Memory Usage**: 20% reduction in memory footprint

### Qualitative Metrics
- **Developer Experience**: Easier to understand and modify
- **User Experience**: No change in functionality
- **Maintainability**: Reduced time to fix bugs
- **Documentation**: Clear and comprehensive

## Final Notes

This simplification maintains the core strengths of the RAG system while removing unnecessary complexity. The modular design is preserved, making it easy to extend and modify. The simplified codebase will be easier to maintain and debug, while providing the same functionality to end users.

The key insight is that **simplicity doesn't mean less capable** - it means focusing on what's actually used and needed, rather than what might theoretically be useful someday.

---

*Document created: 2025-06-29*  
*Last updated: 2025-06-29*