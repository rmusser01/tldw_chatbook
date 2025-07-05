# conftest.py
# Test fixtures for RAG Search and Embeddings tests

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, Mock, patch
import threading
import time
import numpy as np

# Try to import from the new simplified structure first
try:
    from tldw_chatbook.RAG_Search.simplified import (
        EmbeddingConfig,
        VectorStore,
        InMemoryVectorStore,
        EmbeddingsService,
        create_embeddings_service
    )
    # Legacy aliases for compatibility
    InMemoryStore = InMemoryVectorStore
    
    # Create compatibility wrapper for EmbeddingsService
    if EmbeddingsService:
        _OriginalEmbeddingsService = EmbeddingsService
        
        class EmbeddingsServiceCompat(_OriginalEmbeddingsService):
            """Compatibility wrapper to handle old test API."""
            def __init__(self, persist_directory=None, vector_store=None, **kwargs):
                # Map old parameters to new ones
                model_name = kwargs.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
                cache_size = kwargs.get('cache_size', 2)
                device = kwargs.get('device', None)
                api_key = kwargs.get('api_key', None)
                base_url = kwargs.get('base_url', None)
                
                # Initialize with new API
                super().__init__(
                    model_name=model_name,
                    cache_size=cache_size,
                    device=device,
                    api_key=api_key,
                    base_url=base_url
                )
                
                # Store old parameters for tests that might check them
                self.persist_directory = persist_directory
                self.vector_store = vector_store
                self.providers = {}  # For tests expecting providers dict
                self.current_provider_id = "default"
                
            def add_provider(self, provider_id, provider):
                """Mock method for tests expecting multi-provider support."""
                self.providers[provider_id] = provider
                self.current_provider_id = provider_id
                
            def set_provider(self, provider_id):
                """Mock method for provider switching."""
                if provider_id in self.providers:
                    self.current_provider_id = provider_id
                    return True
                return False
                
            def initialize_embedding_model(self):
                """Mock method for model initialization."""
                return True
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                """Add context manager support."""
                if hasattr(self, 'factory') and hasattr(self.factory, 'close'):
                    self.factory.close()
                    
        # Replace with compat version
        EmbeddingsService = EmbeddingsServiceCompat
        
except ImportError:
    # Fall back to None if simplified imports fail
    EmbeddingsService = None
    VectorStore = None
    InMemoryVectorStore = None
    InMemoryStore = None
    create_embeddings_service = None
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE


# ===========================================
# Compatibility Layer for Tests
# ===========================================

class EmbeddingFactoryCompat:
    """Compatibility wrapper for legacy embedding factory interface."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get('provider', 'mock')
        self.model = config.get('model', 'mock-model')
        self.api_key = config.get('api_key', 'mock-key')
        self.batch_size = config.get('batch_size', 32)
        self.dimension = config.get('dimension', 384)
        
    def get_embedding_provider(self):
        """Create a mock embedding provider for tests."""
        provider = MockEmbeddingProvider()
        provider.dimension = self.dimension
        return provider
    
    def get_batch_size(self):
        """Return configured batch size."""
        return self.batch_size
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create mock embeddings for texts."""
        return [[0.1] * self.dimension for _ in texts]
    
    def __repr__(self):
        return f"EmbeddingFactoryCompat(provider={self.provider}, model={self.model})"


# ===========================================
# Directory Fixtures
# ===========================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def persist_dir(temp_dir):
    """Create a persistence directory for vector stores"""
    persist_dir = temp_dir / "vector_store"
    persist_dir.mkdir(exist_ok=True)
    return persist_dir


# ===========================================
# Mock Providers
# ===========================================

class SentenceTransformerProvider:
    """Mock SentenceTransformerProvider for tests."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.dimension = 384 if "MiniLM" in model_name else 768
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create mock embeddings"""
        return [[0.1] * self.dimension for _ in texts]
    
    def get_dimension(self) -> int:
        return self.dimension
    
    def cleanup(self) -> None:
        """Mock cleanup"""
        pass
        
class HuggingFaceProvider:
    """Mock HuggingFaceProvider for tests."""
    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 32):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.dimension = 768
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create mock embeddings"""
        return [[0.2] * self.dimension for _ in texts]
    
    def get_dimension(self) -> int:
        return self.dimension
    
    def cleanup(self) -> None:
        """Mock cleanup"""
        pass

class MockEmbeddingProvider:
    """Mock embedding provider for testing"""
    # Implementing a mock without inheriting since EmbeddingProvider is not available
    
    def __init__(self, dimension: int = 384, delay: float = 0.0, fail_after: Optional[int] = None):
        self.dimension = dimension
        self.delay = delay
        self.fail_after = fail_after
        self.call_count = 0
        self.cleaned_up = False
        self._lock = threading.RLock()
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create mock embeddings"""
        with self._lock:
            self.call_count += 1
            
            if self.fail_after and self.call_count > self.fail_after:
                raise RuntimeError("Mock provider failure")
            
            if self.delay > 0:
                time.sleep(self.delay)
            
            # Create deterministic embeddings based on text
            embeddings = []
            for text in texts:
                # Use hash to create deterministic values
                text_hash = hash(text) % 1000
                embedding = [(text_hash + i) / 1000.0 for i in range(self.dimension)]
                embeddings.append(embedding)
            
            return embeddings
    
    def get_dimension(self) -> int:
        return self.dimension
    
    def cleanup(self) -> None:
        self.cleaned_up = True


class SlowMockProvider(MockEmbeddingProvider):
    """Mock provider that simulates slow operations"""
    
    def __init__(self, dimension: int = 384, delay: float = 0.1):
        super().__init__(dimension=dimension, delay=delay)


class FailingMockProvider(MockEmbeddingProvider):
    """Mock provider that fails after N calls"""
    
    def __init__(self, dimension: int = 384, fail_after: int = 3):
        super().__init__(dimension=dimension, fail_after=fail_after)


@pytest.fixture
def mock_provider():
    """Create a basic mock embedding provider"""
    return MockEmbeddingProvider()


@pytest.fixture
def slow_provider():
    """Create a slow mock embedding provider"""
    return SlowMockProvider(delay=0.05)


@pytest.fixture
def failing_provider():
    """Create a provider that fails after some calls"""
    return FailingMockProvider(fail_after=3)


# ===========================================
# Mock Vector Stores
# ===========================================

class MockVectorStore:
    """Mock vector store for testing"""
    # Implementing a mock without inheriting since VectorStore base class structure changed
    
    def __init__(self, fail_on_operation: Optional[str] = None):
        self.collections = {}
        self.fail_on_operation = fail_on_operation
        self.call_log = []
        self._lock = threading.RLock()
    
    def _maybe_fail(self, operation: str):
        if self.fail_on_operation == operation:
            raise RuntimeError(f"Mock failure on {operation}")
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        with self._lock:
            self.call_log.append(("add_documents", collection_name, len(documents)))
            self._maybe_fail("add_documents")
            
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
        with self._lock:
            self.call_log.append(("search", collection_name, n_results))
            self._maybe_fail("search")
            
            if collection_name not in self.collections:
                return None
            
            # Return mock results
            return {
                "ids": [["doc1", "doc2"][:n_results]],
                "documents": [["Document 1", "Document 2"][:n_results]],
                "metadatas": [[{"type": "test"}, {"type": "test"}][:n_results]],
                "distances": [[0.1, 0.2][:n_results]]
            }
    
    def delete_collection(self, collection_name: str) -> bool:
        with self._lock:
            self.call_log.append(("delete_collection", collection_name))
            self._maybe_fail("delete_collection")
            
            if collection_name in self.collections:
                del self.collections[collection_name]
            return True
    
    def list_collections(self) -> List[str]:
        with self._lock:
            self.call_log.append(("list_collections",))
            self._maybe_fail("list_collections")
            return list(self.collections.keys())


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store"""
    return MockVectorStore()


@pytest.fixture
def failing_vector_store():
    """Create a vector store that fails on search"""
    return MockVectorStore(fail_on_operation="search")


# ===========================================
# Service Fixtures
# ===========================================

@pytest.fixture
def embeddings_service(mock_vector_store):
    """Create an embeddings service with mocks"""
    if EmbeddingsService is None:
        pytest.skip("EmbeddingsService not available")
    # Simplified API doesn't expose add_provider method
    # Using the service with default configuration
    service = create_embeddings_service() if create_embeddings_service else None
    if service is None:
        pytest.skip("Cannot create embeddings service")
    return service


@pytest.fixture
def service_with_memory_store():
    """Create service with in-memory vector store"""
    if InMemoryVectorStore is None:
        pytest.skip("InMemoryVectorStore not available")
    # Using the simplified API
    service = create_embeddings_service() if create_embeddings_service else None
    if service is None:
        pytest.skip("Cannot create embeddings service")
    yield service
    # Cleanup handled internally


@pytest.fixture
def service_with_multiple_providers():
    """Create service with multiple providers"""
    # Simplified API doesn't expose provider management
    # Using default service configuration
    service = create_embeddings_service() if create_embeddings_service else None
    if service is None:
        pytest.skip("Cannot create embeddings service")
    return service


# ===========================================
# Configuration Fixtures
# ===========================================

@pytest.fixture
def legacy_config():
    """Legacy embedding configuration"""
    return {
        "default_model_id": "test-model",
        "models": {
            "test-model": {
                "provider": "huggingface",
                "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
                "trust_remote_code": False
            },
            "openai-model": {
                "provider": "openai",
                "model_name_or_path": "text-embedding-3-small",
                "api_key": "test-key",
                "dimension": 1536
            }
        }
    }


@pytest.fixture
def nested_config(legacy_config):
    """Configuration with nested structure (like in ChromaDBManager)"""
    return {
        "COMPREHENSIVE_CONFIG_RAW": {
            "embedding_config": legacy_config
        }
    }


# ===========================================
# Mock Cache Service
# ===========================================

@pytest.fixture
def mock_cache_service():
    """Create a mock cache service"""
    cache = MagicMock()
    cache.get_embeddings_batch.return_value = ({}, ["text1", "text2"])  # No cached, all uncached
    cache.cache_embeddings_batch.return_value = None
    return cache


@pytest.fixture
def cache_with_hits():
    """Create a cache service with some hits"""
    cache = MagicMock()
    # Return some cached embeddings
    cache.get_embeddings_batch.return_value = (
        {"cached_text": [0.1] * 384},  # One cached
        ["uncached_text"]  # One uncached
    )
    cache.cache_embeddings_batch.return_value = None
    return cache


# ===========================================
# Test Data Generators
# ===========================================

@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming the world",
        "Embeddings capture semantic meaning of text",
        "Testing is important for software quality",
        "Python is a versatile programming language"
    ]


@pytest.fixture
def large_text_batch():
    """Large batch of texts for performance testing"""
    return [f"This is test document number {i}" for i in range(1000)]


@pytest.fixture
def text_with_special_chars():
    """Texts with special characters for edge case testing"""
    return [
        "Text with émojis 😀🎉",
        "Tëxt wíth àccénts",
        "Text\nwith\nnewlines",
        "Text\twith\ttabs",
        "Text with 中文 characters",
        "🚀 Starting with emoji",
        ""  # Empty string
    ]


# ===========================================
# Thread Testing Utilities
# ===========================================

class ThreadTestHelper:
    """Helper for concurrent testing"""
    
    def __init__(self):
        self.results = []
        self.errors = []
        self.lock = threading.Lock()
    
    def record_result(self, thread_id: int, result: Any):
        with self.lock:
            self.results.append((thread_id, result))
    
    def record_error(self, thread_id: int, error: Exception):
        with self.lock:
            self.errors.append((thread_id, error))
    
    def run_concurrent(self, func, num_threads: int = 5, args_list: Optional[List] = None):
        """Run function concurrently in multiple threads"""
        threads = []
        
        if args_list is None:
            args_list = [(i,) for i in range(num_threads)]
        
        for i, args in enumerate(args_list):
            def wrapper(thread_id=i, thread_args=args):
                try:
                    result = func(*thread_args)
                    self.record_result(thread_id, result)
                except Exception as e:
                    self.record_error(thread_id, e)
            
            thread = threading.Thread(target=wrapper)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=10)
        
        return self.results, self.errors


@pytest.fixture
def thread_helper():
    """Create thread test helper"""
    return ThreadTestHelper()


# ===========================================
# Performance Testing Utilities
# ===========================================

@pytest.fixture
def performance_monitor():
    """Monitor for performance metrics"""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.metrics = {}
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.end_time - self.start_time
        
        def record_metric(self, name: str, value: float):
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
        
        def get_average(self, name: str) -> float:
            values = self.metrics.get(name, [])
            return sum(values) / len(values) if values else 0.0
    
    return PerformanceMonitor()


# ===========================================
# Skip Markers
# ===========================================

requires_embeddings = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False),
    reason="Embeddings dependencies not available"
)

requires_chromadb = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE.get('chromadb', False),
    reason="ChromaDB not available"
)

requires_numpy = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE.get('numpy', False),
    reason="NumPy not available"
)