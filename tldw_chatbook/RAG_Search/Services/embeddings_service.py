# embeddings_service.py
# Description: Service for managing embeddings and vector storage with multi-provider support
#
# Imports
from typing import List, Dict, Any, Optional, Tuple, Union, Protocol, Callable
from abc import ABC, abstractmethod
import json
from pathlib import Path
from loguru import logger
import hashlib
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
from enum import Enum
#
# Local Imports
from ...Utils.optional_deps import DEPENDENCIES_AVAILABLE, get_safe_import
from .cache_service import get_cache_service

logger = logger.bind(module="embeddings_service")

# Check dependencies
EMBEDDINGS_AVAILABLE = DEPENDENCIES_AVAILABLE.get('embeddings_rag', False)
CHROMADB_AVAILABLE = DEPENDENCIES_AVAILABLE.get('chromadb', False)

# Safe imports for optional dependencies
numpy = get_safe_import('numpy')
torch = get_safe_import('torch')
transformers = get_safe_import('transformers')
sentence_transformers = get_safe_import('sentence_transformers')

if CHROMADB_AVAILABLE:
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        CHROMADB_AVAILABLE = False
        logger.warning("ChromaDB import failed despite being marked as available")

# Type aliases for optional dependencies
if numpy is not None:
    np = numpy
else:
    np = None

if torch is not None:
    Tensor = torch.Tensor
    normalize = torch.nn.functional.normalize
else:
    Tensor = Any
    normalize = None


class EmbeddingProviderType(Enum):
    """Supported embedding provider types"""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    CUSTOM = "custom"


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources used by the provider"""
        pass


class VectorStore(ABC):
    """Abstract base class for vector storage backends"""
    
    @abstractmethod
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """Add documents with embeddings to a collection"""
        pass
    
    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collections"""
        pass
    
    def update_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """Update existing documents (optional - default is delete + add)"""
        # Default implementation: delete and re-add
        self.delete_documents(collection_name, ids)
        return self.add_documents(collection_name, documents, embeddings, metadatas, ids)
    
    def delete_documents(self, collection_name: str, ids: List[str]) -> bool:
        """Delete specific documents (optional)"""
        # Default implementation: not supported
        logger.warning(f"{self.__class__.__name__} does not support document deletion")
        return False


# Pooling function type for HuggingFace models
PoolingFn = Callable[[Tensor, Tensor], Tensor]


def _masked_mean(last_hidden: Tensor, attn: Tensor) -> Tensor:
    """Default pooling: mean of vectors where attention_mask is 1."""
    if normalize is None:
        raise ImportError("PyTorch not available for masked mean pooling")
    mask = attn.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-9)
    avg = summed / lengths
    return normalize(avg, p=2, dim=1)


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        if sentence_transformers is None:
            raise ImportError("sentence-transformers not available. Install with: pip install tldw_chatbook[embeddings_rag]")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self._model = None
        self._dimension = None
        self._lock = threading.RLock()
        
    @property
    def model(self):
        """Lazy load the model"""
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                # Get dimension from a test embedding
                test_embedding = self._model.encode(["test"], show_progress_bar=False)
                self._dimension = len(test_embedding[0])
            return self._model
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using sentence-transformers"""
        with self._lock:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            # Force model loading to get dimension
            _ = self.model
        return self._dimension
    
    def cleanup(self) -> None:
        """Cleanup model resources"""
        with self._lock:
            self._model = None
            self._dimension = None


class HuggingFaceProvider(EmbeddingProvider):
    """HuggingFace transformers embedding provider with advanced pooling"""
    
    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        max_length: int = 512,
        device: Optional[str] = None,
        batch_size: int = 32,
        pooling: Optional[PoolingFn] = None,
        dimension: Optional[int] = None
    ):
        if transformers is None:
            raise ImportError("transformers not available. Install with: pip install tldw_chatbook[embeddings_rag]")
        
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.max_length = max_length
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.pooling = pooling or _masked_mean
        self._dimension = dimension
        self._model = None
        self._tokenizer = None
        self._lock = threading.RLock()
    
    @property
    def model(self):
        """Lazy load the model"""
        with self._lock:
            if self._model is None:
                from transformers import AutoModel, AutoTokenizer
                logger.info(f"Loading HuggingFace model: {self.model_name}")
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=self.trust_remote_code
                )
                
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=dtype
                ).to(self.device).eval()
                
                # Infer dimension if not provided
                if self._dimension is None:
                    with torch.no_grad():
                        test_input = self._tokenizer(
                            ["test"], 
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.max_length
                        ).to(self.device)
                        test_output = self._model(**test_input)
                        test_embedding = self.pooling(
                            test_output.last_hidden_state,
                            test_input.attention_mask
                        )
                        self._dimension = test_embedding.shape[1]
                        
            return self._model, self._tokenizer
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using HuggingFace transformers"""
        with self._lock:
            model, tokenizer = self.model
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = self.pooling(
                        outputs.last_hidden_state,
                        inputs.attention_mask
                    )
                    all_embeddings.extend(embeddings.cpu().numpy().tolist())
            
            return all_embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            # Force model loading to get dimension
            _ = self.model
        return self._dimension
    
    def cleanup(self) -> None:
        """Cleanup model resources"""
        with self._lock:
            self._model = None
            self._tokenizer = None


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embeddings provider"""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.dimension = dimension
        self._lock = threading.RLock()
        
        if not self.api_key and not self.base_url:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY env var or pass api_key)")
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenAI API with retry logic"""
        import requests
        from time import sleep
        
        with self._lock:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            url = self.base_url or "https://api.openai.com/v1/embeddings"
            if self.base_url and not self.base_url.endswith("/embeddings"):
                url = f"{self.base_url}/embeddings"
            
            data = {
                "model": self.model_name,
                "input": texts
            }
            
            if self.dimension:
                data["dimensions"] = self.dimension
            
            # Retry logic with exponential backoff
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, headers=headers, json=data, timeout=30)
                    response.raise_for_status()
                    
                    result = response.json()
                    embeddings = [item["embedding"] for item in result["data"]]
                    
                    # Update dimension if not set
                    if not self.dimension and embeddings:
                        self.dimension = len(embeddings[0])
                    
                    return embeddings
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"OpenAI API timeout (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise
                        
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit
                        logger.warning(f"OpenAI API rate limit (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            # Extract retry-after header if available
                            retry_after = e.response.headers.get('Retry-After', retry_delay)
                            sleep(float(retry_after))
                            retry_delay *= 2
                        else:
                            raise
                    elif e.response.status_code >= 500:  # Server error
                        logger.warning(f"OpenAI API server error {e.response.status_code} (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            raise
                    else:
                        # Don't retry client errors
                        raise
                        
                except Exception as e:
                    logger.error(f"Unexpected error in OpenAI embeddings: {e}")
                    raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.dimension is None:
            # Create a test embedding to get dimension
            test_embeddings = self.create_embeddings(["test"])
            self.dimension = len(test_embeddings[0])
        return self.dimension
    
    def cleanup(self) -> None:
        """No cleanup needed for API provider"""
        pass


class ChromaDBStore(VectorStore):
    """ChromaDB implementation of vector store"""
    
    def __init__(self, client, memory_limit_bytes: Optional[int] = None):
        self.client = client
        self.memory_limit_bytes = memory_limit_bytes
        self._lock = threading.RLock()
        
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """Add documents to ChromaDB collection"""
        with self._lock:
            try:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={}
                )
                collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                return True
            except Exception as e:
                logger.error(f"Error adding documents to ChromaDB: {e}")
                return False
    
    def search(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Search ChromaDB collection"""
        with self._lock:
            try:
                collection = self.client.get_collection(name=collection_name)
                results = collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=where
                )
                return results
            except Exception as e:
                logger.error(f"Error searching ChromaDB: {e}")
                return None
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete ChromaDB collection"""
        with self._lock:
            try:
                self.client.delete_collection(name=collection_name)
                return True
            except Exception as e:
                logger.error(f"Error deleting collection: {e}")
                return False
    
    def list_collections(self) -> List[str]:
        """List all ChromaDB collections"""
        with self._lock:
            try:
                collections = self.client.list_collections()
                return [col.name for col in collections]
            except Exception as e:
                logger.error(f"Error listing collections: {e}")
                return []
    
    def update_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """Update documents in ChromaDB"""
        with self._lock:
            try:
                collection = self.client.get_collection(name=collection_name)
                collection.update(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                return True
            except Exception as e:
                logger.error(f"Error updating documents: {e}")
                return False
    
    def delete_documents(self, collection_name: str, ids: List[str]) -> bool:
        """Delete documents from ChromaDB"""
        with self._lock:
            try:
                collection = self.client.get_collection(name=collection_name)
                collection.delete(ids=ids)
                return True
            except Exception as e:
                logger.error(f"Error deleting documents: {e}")
                return False


class InMemoryStore(VectorStore):
    """In-memory implementation of vector store for testing and lightweight usage"""
    
    def __init__(self):
        self.collections: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """Add documents to in-memory collection"""
        with self._lock:
            if collection_name not in self.collections:
                self.collections[collection_name] = {
                    "documents": [],
                    "embeddings": [],
                    "metadatas": [],
                    "ids": []
                }
            
            collection = self.collections[collection_name]
            collection["documents"].extend(documents)
            collection["embeddings"].extend(embeddings)
            collection["metadatas"].extend(metadatas)
            collection["ids"].extend(ids)
            return True
    
    def search(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Search in-memory collection using cosine similarity"""
        with self._lock:
            if collection_name not in self.collections:
                return None
            
            collection = self.collections[collection_name]
            if not collection["embeddings"]:
                return {"ids": [], "documents": [], "metadatas": [], "distances": []}
            
            # Simple cosine similarity search
            results = {"ids": [], "documents": [], "metadatas": [], "distances": []}
            
            for query_embedding in query_embeddings:
                scores = []
                for i, doc_embedding in enumerate(collection["embeddings"]):
                    # Cosine similarity
                    if np is not None:
                        query_vec = np.array(query_embedding)
                        doc_vec = np.array(doc_embedding)
                        score = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
                    else:
                        # Fallback without numpy
                        dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                        query_norm = sum(a * a for a in query_embedding) ** 0.5
                        doc_norm = sum(b * b for b in doc_embedding) ** 0.5
                        score = dot_product / (query_norm * doc_norm)
                    
                    scores.append((score, i))
                
                # Sort by score and take top n_results
                scores.sort(reverse=True, key=lambda x: x[0])
                top_results = scores[:n_results]
                
                batch_ids = []
                batch_docs = []
                batch_meta = []
                batch_dist = []
                
                for score, idx in top_results:
                    batch_ids.append(collection["ids"][idx])
                    batch_docs.append(collection["documents"][idx])
                    batch_meta.append(collection["metadatas"][idx])
                    batch_dist.append(1 - score)  # Convert similarity to distance
                
                results["ids"].append(batch_ids)
                results["documents"].append(batch_docs)
                results["metadatas"].append(batch_meta)
                results["distances"].append(batch_dist)
            
            return results
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete in-memory collection"""
        with self._lock:
            if collection_name in self.collections:
                del self.collections[collection_name]
            return True
    
    def list_collections(self) -> List[str]:
        """List all in-memory collections"""
        with self._lock:
            return list(self.collections.keys())


class EmbeddingsService:
    """Service for managing embeddings with multi-provider support and pluggable vector storage"""
    
    def __init__(
        self, 
        persist_directory: Optional[Path] = None,
        memory_limit_bytes: Optional[int] = None,
        vector_store: Optional[VectorStore] = None,
        embedding_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the embeddings service with multi-provider support
        
        Args:
            persist_directory: Directory to store vector data (if using ChromaDB)
            memory_limit_bytes: Optional memory limit for vector store
            vector_store: Optional custom vector store implementation
            embedding_config: Optional embedding configuration dict
        """
        self.persist_directory = persist_directory
        if persist_directory:
            persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._provider_lock = threading.RLock()
        self._executor_lock = threading.RLock()
        
        # Initialize vector store
        self.vector_store = vector_store
        if not self.vector_store and persist_directory and CHROMADB_AVAILABLE:
            try:
                settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
                
                if memory_limit_bytes:
                    settings.chroma_segment_cache_policy = "LRU"
                    settings.chroma_memory_limit_bytes = memory_limit_bytes
                    logger.info(f"ChromaDB LRU cache configured with {memory_limit_bytes} bytes limit")
                
                client = chromadb.PersistentClient(
                    path=str(persist_directory),
                    settings=settings
                )
                self.vector_store = ChromaDBStore(client, memory_limit_bytes)
                logger.info(f"ChromaDB initialized at {persist_directory}")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                logger.info("Falling back to in-memory vector store")
                self.vector_store = InMemoryStore()
        elif not self.vector_store:
            logger.info("Using in-memory vector store")
            self.vector_store = InMemoryStore()
        
        # Embedding providers
        self.providers: Dict[str, EmbeddingProvider] = {}
        self.current_provider_id: Optional[str] = None
        self.embedding_config = embedding_config or {}
        
        # Try to use cache service, but don't fail if unavailable
        try:
            self.cache_service = get_cache_service()
        except Exception as e:
            logger.warning(f"Cache service unavailable: {e}. Continuing without cache.")
            self.cache_service = None
        
        # For backward compatibility
        self.embedding_model = None  
        self.memory_manager = None
        
        # Performance settings
        self.max_workers = 4
        self.batch_size = 32
        self.enable_parallel_processing = True
        self._executor = None
        
        # ChromaDB client for backward compatibility
        self.client = getattr(self.vector_store, 'client', None) if isinstance(self.vector_store, ChromaDBStore) else None
        
    def initialize_from_config(self, config: Dict[str, Any]) -> bool:
        """
        Initialize providers from legacy embedding_config format
        
        Args:
            config: Configuration dict with embedding_config section
            
        Returns:
            True if successful
        """
        with self._provider_lock:
            try:
                # Handle legacy config format
                embed_config = config.get("embedding_config", {})
                if not embed_config:
                    # Try nested locations
                    embed_config = config.get("COMPREHENSIVE_CONFIG_RAW", {}).get("embedding_config", {})
                
                if not embed_config:
                    logger.warning("No embedding_config found in provided configuration")
                    return False
                
                models = embed_config.get("models", {})
                default_model = embed_config.get("default_model_id")
                
                # Load each model configuration
                for model_id, model_cfg in models.items():
                    provider_type = model_cfg.get("provider", "").lower()
                    
                    if provider_type == "huggingface":
                        provider = HuggingFaceProvider(
                            model_name=model_cfg.get("model_name_or_path"),
                            trust_remote_code=model_cfg.get("trust_remote_code", False),
                            max_length=model_cfg.get("max_length", 512),
                            device=model_cfg.get("device"),
                            batch_size=model_cfg.get("batch_size", 32),
                            dimension=model_cfg.get("dimension")
                        )
                        self.providers[model_id] = provider
                    elif provider_type == "openai":
                        provider = OpenAIProvider(
                            model_name=model_cfg.get("model_name_or_path", "text-embedding-3-small"),
                            api_key=model_cfg.get("api_key"),
                            base_url=model_cfg.get("base_url"),
                            dimension=model_cfg.get("dimension")
                        )
                        self.providers[model_id] = provider
                    elif provider_type == "sentence_transformers":
                        # Handle sentence-transformers models
                        provider = SentenceTransformerProvider(
                            model_name=model_cfg.get("model_name_or_path"),
                            device=model_cfg.get("device")
                        )
                        self.providers[model_id] = provider
                    else:
                        logger.warning(f"Unknown provider type: {provider_type} for model {model_id}")
                
                # Set default provider
                if default_model and default_model in self.providers:
                    self.current_provider_id = default_model
                    logger.info(f"Set default embedding provider: {default_model}")
                elif self.providers:
                    # Use first available provider as default
                    self.current_provider_id = next(iter(self.providers))
                    logger.info(f"Set first available provider as default: {self.current_provider_id}")
                
                return len(self.providers) > 0
                
            except Exception as e:
                logger.error(f"Error initializing from config: {e}")
                return False
    
    def add_provider(self, provider_id: str, provider: EmbeddingProvider) -> None:
        """Add a new embedding provider"""
        with self._provider_lock:
            self.providers[provider_id] = provider
            if not self.current_provider_id:
                self.current_provider_id = provider_id
                logger.info(f"Set {provider_id} as current provider")
    
    def set_provider(self, provider_id: str) -> bool:
        """Switch to a different embedding provider"""
        with self._provider_lock:
            if provider_id not in self.providers:
                logger.error(f"Provider {provider_id} not found")
                return False
            self.current_provider_id = provider_id
            logger.info(f"Switched to provider: {provider_id}")
            return True
    
    def get_current_provider(self) -> Optional[EmbeddingProvider]:
        """Get the current embedding provider"""
        with self._provider_lock:
            if not self.current_provider_id:
                return None
            return self.providers.get(self.current_provider_id)
    
    def initialize_embedding_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model (backward compatibility)
        This method now creates a SentenceTransformer provider
        """
        with self._provider_lock:
            try:
                # Check if it's a sentence-transformers model
                if "sentence-transformers" in model_name or "/" not in model_name:
                    provider = SentenceTransformerProvider(model_name)
                    provider_id = f"compat_{model_name.replace('/', '_')}"
                else:
                    # Assume HuggingFace model
                    provider = HuggingFaceProvider(model_name)
                    provider_id = f"compat_hf_{model_name.replace('/', '_')}"
                
                self.add_provider(provider_id, provider)
                self.current_provider_id = provider_id
                
                # Set backward compatibility attribute
                self.embedding_model = provider
                
                logger.info(f"Initialized embedding model: {model_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                return False
    
    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create the thread pool executor."""
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="embeddings"
                )
            return self._executor
    
    def _close_executor(self):
        """Close the thread pool executor with timeout."""
        with self._executor_lock:
            if self._executor:
                try:
                    # Try to shutdown gracefully
                    self._executor.shutdown(wait=True)
                except Exception as e:
                    logger.warning(f"Error during executor shutdown: {e}")
                    # Force shutdown if graceful shutdown fails
                    try:
                        self._executor.shutdown(wait=False)
                    except:
                        pass
                finally:
                    self._executor = None
    
    def _create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a batch of texts (thread-safe)."""
        provider = self.get_current_provider()
        if not provider:
            raise ValueError("No embedding provider available")
        return provider.create_embeddings(texts)
    
    def create_embeddings(self, texts: List[str], provider_id: Optional[str] = None) -> Optional[List[List[float]]]:
        """
        Create embeddings for a list of texts with parallel processing
        
        Args:
            texts: List of text strings to embed
            provider_id: Optional specific provider to use
            
        Returns:
            List of embedding vectors, or None if failed
        """
        with self._provider_lock:
            # Get provider
            if provider_id:
                provider = self.providers.get(provider_id)
                if not provider:
                    logger.error(f"Provider {provider_id} not found")
                    return None
            else:
                provider = self.get_current_provider()
                if not provider:
                    # Try to initialize default provider for backward compatibility
                    if not self.initialize_embedding_model():
                        logger.error("No embedding provider available")
                        return None
                    provider = self.get_current_provider()
        
        try:
            # Try to use cache service if available
            if self.cache_service:
                try:
                    cached_embeddings, uncached_texts = self.cache_service.get_embeddings_batch(texts)
                    
                    # Create mapping of text to indices (handle duplicates)
                    text_to_indices = {}
                    for i, text in enumerate(texts):
                        if text not in text_to_indices:
                            text_to_indices[text] = []
                        text_to_indices[text].append(i)
                    
                    embeddings = [None] * len(texts)
                    
                    # Fill in cached embeddings
                    for text, embedding in cached_embeddings.items():
                        if text in text_to_indices:
                            for idx in text_to_indices[text]:
                                embeddings[idx] = embedding
                    
                    # Generate embeddings for uncached texts
                    if uncached_texts:
                        if self.enable_parallel_processing and len(uncached_texts) > self.batch_size:
                            new_embeddings = self._create_embeddings_parallel(uncached_texts)
                        else:
                            new_embeddings = provider.create_embeddings(uncached_texts)
                        
                        # Cache the new embeddings
                        text_embedding_pairs = list(zip(uncached_texts, new_embeddings))
                        self.cache_service.cache_embeddings_batch(text_embedding_pairs)
                        
                        # Fill in the results
                        for text, embedding in text_embedding_pairs:
                            if text in text_to_indices:
                                for idx in text_to_indices[text]:
                                    embeddings[idx] = embedding
                    
                    return embeddings
                except Exception as e:
                    logger.warning(f"Cache service error: {e}. Continuing without cache.")
            
            # No cache or cache failed - generate all embeddings
            if self.enable_parallel_processing and len(texts) > self.batch_size:
                return self._create_embeddings_parallel(texts)
            else:
                return provider.create_embeddings(texts)
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return None
    
    def _create_embeddings_parallel(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings in parallel using thread pool.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        provider = self.get_current_provider()
        if not provider:
            raise ValueError("No embedding provider available")
            
        if len(texts) <= self.batch_size:
            return provider.create_embeddings(texts)
        
        # Split into batches
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        
        # Process batches in parallel
        executor = self._get_executor()
        future_to_batch = {}
        
        for i, batch in enumerate(batches):
            future = executor.submit(self._create_embeddings_batch, batch)
            future_to_batch[future] = i
        
        # Collect results in order
        batch_results = [None] * len(batches)
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results[batch_idx] = future.result()
            except Exception as e:
                logger.error(f"Error in parallel embedding batch {batch_idx}: {e}")
                # Fallback to sequential processing for this batch
                try:
                    batch_results[batch_idx] = provider.create_embeddings(batches[batch_idx])
                except Exception as fallback_e:
                    logger.error(f"Fallback embedding also failed: {fallback_e}")
                    batch_results[batch_idx] = []
        
        # Flatten results
        all_embeddings = []
        for batch_embeddings in batch_results:
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def get_or_create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Get or create a collection (backward compatibility for ChromaDB)"""
        if isinstance(self.vector_store, ChromaDBStore) and self.client:
            try:
                # Try to get existing collection
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata=metadata or {}
                )
                logger.debug(f"Got collection: {collection_name}")
                return collection
            except Exception as e:
                logger.error(f"Error getting/creating collection {collection_name}: {e}")
                return None
        else:
            # For non-ChromaDB stores, return a dummy object
            logger.debug(f"Non-ChromaDB store, returning dummy collection for: {collection_name}")
            return {"name": collection_name, "metadata": metadata}
    
    def add_documents_to_collection(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        batch_size: Optional[int] = None
    ) -> bool:
        """
        Add documents with embeddings to a collection with batch processing
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            ids: List of unique IDs
            batch_size: Batch size for insertion (None = use default)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vector_store:
            logger.error("No vector store available")
            return False
            
        try:
            if batch_size is None:
                batch_size = self.batch_size
                
            # Process in batches for better performance
            if len(documents) > batch_size:
                return self._add_documents_batch(
                    collection_name, documents, embeddings, metadatas, ids, batch_size
                )
            else:
                # Single batch
                return self.vector_store.add_documents(
                    collection_name, documents, embeddings, metadatas, ids
                )
                
        except Exception as e:
            logger.error(f"Error adding documents to collection: {e}")
            return False
    
    def _add_documents_batch(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        batch_size: int
    ) -> bool:
        """
        Add documents to collection in batches.
        
        Args:
            collection_name: Name of the collection
            documents: Document texts
            embeddings: Embedding vectors
            metadatas: Metadata dicts
            ids: Document IDs
            batch_size: Size of each batch
            
        Returns:
            True if all batches successful
        """
        total_docs = len(documents)
        success_count = 0
        
        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            
            try:
                success = self.vector_store.add_documents(
                    collection_name,
                    documents[i:end_idx],
                    embeddings[i:end_idx],
                    metadatas[i:end_idx],
                    ids[i:end_idx]
                )
                if success:
                    success_count += (end_idx - i)
                
                # Small delay to avoid overwhelming the store
                if end_idx < total_docs:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error adding batch {i}-{end_idx} to collection: {e}")
                # Continue with next batch
                continue
        
        logger.debug(f"Successfully added {success_count}/{total_docs} documents in batches")
        return success_count == total_docs
    
    def search_collection(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Search a collection using query embeddings
        
        Args:
            collection_name: Name of the collection to search
            query_embeddings: Query embedding vectors
            n_results: Number of results to return
            where: Optional filter conditions
            
        Returns:
            Search results dict or None if failed
        """
        if not self.vector_store:
            logger.error("No vector store available")
            return None
            
        try:
            # Update access time for memory management
            if self.memory_manager:
                try:
                    self.memory_manager.update_collection_access_time(collection_name)
                except Exception as e:
                    logger.warning(f"Failed to update collection access time: {e}")
                    # Continue with search even if memory management fails
                
            results = self.vector_store.search(
                collection_name,
                query_embeddings,
                n_results,
                where
            )
            return results
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {e}")
            return None
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        if not self.vector_store:
            return False
            
        try:
            result = self.vector_store.delete_collection(collection_name)
            if result:
                logger.info(f"Deleted collection: {collection_name}")
            return result
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        if not self.vector_store:
            return []
            
        try:
            return self.vector_store.list_collections()
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection"""
        if isinstance(self.vector_store, ChromaDBStore) and self.client:
            collection = self.get_or_create_collection(collection_name)
            if not collection:
                return None
                
            try:
                return {
                    "name": collection.name,
                    "count": collection.count(),
                    "metadata": collection.metadata
                }
            except Exception as e:
                logger.error(f"Error getting collection info: {e}")
                return None
        else:
            # For non-ChromaDB stores, return basic info
            collections = self.vector_store.list_collections()
            if collection_name in collections:
                return {
                    "name": collection_name,
                    "count": -1,  # Unknown
                    "metadata": {}
                }
            return None
    
    def set_memory_manager(self, memory_manager):
        """Set the memory manager for this service."""
        self.memory_manager = memory_manager
    
    def get_memory_usage_summary(self) -> Optional[Dict[str, Any]]:
        """Get memory usage summary through memory manager."""
        if self.memory_manager:
            return self.memory_manager.get_memory_usage_summary()
        return None
    
    async def run_memory_cleanup(self) -> Dict[str, int]:
        """Run memory cleanup through memory manager."""
        if self.memory_manager:
            return await self.memory_manager.run_automatic_cleanup()
        return {}
    
    def get_cleanup_recommendations(self) -> List[Dict[str, Any]]:
        """Get cleanup recommendations from memory manager."""
        if self.memory_manager:
            return self.memory_manager.get_cleanup_recommendations()
        return []
    
    def configure_performance(
        self,
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        enable_parallel: Optional[bool] = None
    ):
        """
        Configure performance settings.
        
        Args:
            max_workers: Number of worker threads for parallel processing
            batch_size: Batch size for operations
            enable_parallel: Enable/disable parallel processing
        """
        if max_workers is not None:
            self.max_workers = max_workers
            # Close existing executor to recreate with new worker count
            self._close_executor()
            
        if batch_size is not None:
            self.batch_size = batch_size
            
        if enable_parallel is not None:
            self.enable_parallel_processing = enable_parallel
            
        logger.info(f"Performance configured: workers={self.max_workers}, batch_size={self.batch_size}, parallel={self.enable_parallel_processing}")
    
    def clear_collection(self, collection_name: str) -> bool:
        """
        Clear a collection by deleting and recreating it.
        
        Args:
            collection_name: Name of the collection to clear
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vector_store:
            logger.error("No vector store available")
            return False
            
        try:
            # Delete the collection
            self.vector_store.delete_collection(collection_name)
            logger.info(f"Cleared collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection {collection_name}: {e}")
            return False
    
    def update_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """
        Update existing documents in a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            ids: List of document IDs to update
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vector_store:
            logger.error("No vector store available")
            return False
            
        try:
            result = self.vector_store.update_documents(
                collection_name, documents, embeddings, metadatas, ids
            )
            if result:
                logger.info(f"Updated {len(documents)} documents in collection: {collection_name}")
            return result
        except Exception as e:
            logger.error(f"Error updating documents in collection {collection_name}: {e}")
            return False
    
    def delete_documents(self, collection_name: str, ids: List[str]) -> bool:
        """
        Delete specific documents from a collection.
        
        Args:
            collection_name: Name of the collection
            ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vector_store:
            logger.error("No vector store available")
            return False
            
        try:
            result = self.vector_store.delete_documents(collection_name, ids)
            if result:
                logger.info(f"Deleted {len(ids)} documents from collection: {collection_name}")
            return result
        except Exception as e:
            logger.error(f"Error deleting documents from collection {collection_name}: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self._close_executor()
        self._cleanup_providers()
        return False
    
    def _cleanup_providers(self):
        """Cleanup all embedding providers"""
        with self._provider_lock:
            for provider_id, provider in self.providers.items():
                try:
                    provider.cleanup()
                    logger.debug(f"Cleaned up provider: {provider_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up provider {provider_id}: {e}")
            self.providers.clear()
            self.current_provider_id = None
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, '_executor_lock'):
            self._close_executor()
        if hasattr(self, '_provider_lock'):
            self._cleanup_providers()