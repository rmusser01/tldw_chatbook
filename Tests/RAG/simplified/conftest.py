"""
conftest.py for simplified RAG tests.
Provides comprehensive fixtures and utilities for testing the simplified RAG implementation.
"""

import pytest
import tempfile
import shutil
import asyncio
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import MagicMock, Mock
from dataclasses import dataclass
import numpy as np

# Import optional dependency checker
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE, check_embeddings_rag_deps

# Force check embeddings dependencies before tests run
# This is a workaround for the dependency check bug where it returns cached False
check_embeddings_rag_deps()


# === Directory Fixtures ===

@pytest.fixture
def temp_dir():
    """Create a temporary directory that's cleaned up after the test."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def chroma_persist_dir(temp_dir):
    """Create a temporary directory for ChromaDB persistence."""
    persist_dir = temp_dir / "chromadb"
    persist_dir.mkdir(exist_ok=True)
    yield persist_dir


# === Sample Data Fixtures ===

@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        {
            "id": "doc1",
            "title": "Introduction to Python",
            "content": "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "metadata": {
                "source": "media",
                "author": "John Doe",
                "date": "2024-01-15",
                "tags": ["python", "programming", "tutorial"]
            }
        },
        {
            "id": "doc2", 
            "title": "Machine Learning Basics",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Common algorithms include decision trees, neural networks, and support vector machines.",
            "metadata": {
                "source": "notes",
                "author": "Jane Smith",
                "date": "2024-02-20",
                "tags": ["ML", "AI", "algorithms"]
            }
        },
        {
            "id": "doc3",
            "title": "Web Development with Django",
            "content": "Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. It follows the model-template-view architectural pattern.",
            "metadata": {
                "source": "conversations",
                "participants": ["user", "assistant"],
                "date": "2024-03-10"
            }
        }
    ]


@pytest.fixture
def sample_queries():
    """Provide sample search queries."""
    return [
        "What is Python programming?",
        "machine learning algorithms",
        "web development frameworks",
        "artificial intelligence",
        "programming languages"
    ]


# === Mock Embeddings Fixtures ===

@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings service with deterministic outputs."""
    mock = MagicMock()
    
    def create_embeddings(texts):
        # Return deterministic embeddings based on text
        embeddings = []
        for text in texts:
            # Create a simple embedding based on character frequencies
            text_lower = text.lower()
            embedding = []
            for i in range(384):  # Standard embedding dimension
                char_idx = i % 26
                char = chr(ord('a') + char_idx)
                freq = text_lower.count(char) / max(len(text_lower), 1)
                # Add some hash-based variation
                hash_val = hash(text + str(i)) % 1000
                embedding.append(freq + hash_val / 10000)
            embeddings.append(embedding)
        return embeddings
    
    mock.create_embeddings.side_effect = create_embeddings
    mock.get_dimension.return_value = 384
    mock.model_name = "mock-embedding-model"
    return mock


@pytest.fixture
def lightweight_embeddings():
    """Create a real lightweight embeddings service for integration tests."""
    try:
        from sentence_transformers import SentenceTransformer
        # Use a very small model for testing
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        class LightweightEmbeddings:
            def __init__(self):
                self.model = model
                
            def create_embeddings(self, texts: List[str]) -> List[List[float]]:
                embeddings = self.model.encode(texts)
                return embeddings.tolist()
                
            def get_dimension(self) -> int:
                return 384  # all-MiniLM-L6-v2 dimension
                
            @property
            def model_name(self) -> str:
                return "all-MiniLM-L6-v2"
        
        return LightweightEmbeddings()
    except ImportError:
        # Fall back to mock if sentence-transformers not available
        return mock_embeddings()


# === Vector Store Fixtures ===

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store with basic functionality."""
    mock = MagicMock()
    mock.documents = {}
    mock.embeddings = {}
    
    def add_documents(doc_ids, contents, metadatas, embeddings):
        for i, doc_id in enumerate(doc_ids):
            mock.documents[doc_id] = {
                "content": contents[i],
                "metadata": metadatas[i],
                "embedding": embeddings[i]
            }
        return True
    
    def search(query_embedding, top_k=5, filter_dict=None):
        # Simple distance-based search
        results = []
        for doc_id, doc in mock.documents.items():
            if filter_dict:
                # Check metadata filters
                match = all(
                    doc["metadata"].get(k) == v 
                    for k, v in filter_dict.items()
                )
                if not match:
                    continue
            
            # Calculate simple distance
            distance = np.linalg.norm(
                np.array(query_embedding) - np.array(doc["embedding"])
            )
            results.append((doc_id, doc, distance))
        
        # Sort by distance
        results.sort(key=lambda x: x[2])
        results = results[:top_k]
        
        return {
            'ids': [[r[0] for r in results]],
            'documents': [[r[1]["content"] for r in results]],
            'metadatas': [[r[1]["metadata"] for r in results]],
            'distances': [[r[2] for r in results]]
        }
    
    mock.add_documents.side_effect = add_documents
    mock.search.side_effect = search
    mock.delete_collection.return_value = True
    mock.get_stats.return_value = {
        'total_documents': len(mock.documents),
        'total_chunks': len(mock.documents)
    }
    return mock


# === Chunking Fixtures ===

@pytest.fixture
def simple_chunking_service():
    """Create a simple chunking service for testing."""
    class SimpleChunkingService:
        def chunk_text(self, text: str, chunk_size: int = 100, 
                      chunk_overlap: int = 20) -> List[Dict[str, Any]]:
            if not text:
                return []
            
            words = text.split()
            chunks = []
            chunk_words = max(1, chunk_size // 5)  # Assume 5 chars per word
            step = max(1, chunk_words - chunk_overlap // 5)
            
            for i in range(0, len(words), step):
                chunk_text = " ".join(words[i:i + chunk_words])
                if chunk_text:  # Don't add empty chunks
                    chunks.append({
                        "text": chunk_text,
                        "chunk_index": len(chunks),
                        "start_word": i,
                        "end_word": min(i + chunk_words, len(words))
                    })
            
            return chunks
    
    return SimpleChunkingService()


# === Configuration Fixtures ===

@pytest.fixture
def test_rag_config(chroma_persist_dir, request):
    """Create a test RAG configuration."""
    import uuid
    from tldw_chatbook.RAG_Search.simplified.config import (
        RAGConfig, EmbeddingConfig, VectorStoreConfig, 
        ChunkingConfig, SearchConfig
    )
    
    # Create unique collection name for each test
    test_name = request.node.name
    unique_collection = f"test_{test_name}_{uuid.uuid4().hex[:8]}"
    
    return RAGConfig(
        embedding=EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",  # Use real model
            device="cpu",
            cache_size=100,
            batch_size=10
        ),
        vector_store=VectorStoreConfig(
            type="memory",  # Use memory to avoid ChromaDB conflicts
            persist_directory=None,
            collection_name=unique_collection,
            distance_metric="cosine"
        ),
        chunking=ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            chunking_method="words"
        ),
        search=SearchConfig(
            default_top_k=5,
            score_threshold=0.5,
            enable_cache=True,
            cache_size=100,
            cache_ttl=300
        )
    )


@pytest.fixture
def memory_rag_config(request):
    """Create an in-memory RAG configuration."""
    import uuid
    from tldw_chatbook.RAG_Search.simplified.config import (
        RAGConfig, EmbeddingConfig, VectorStoreConfig,
        ChunkingConfig, SearchConfig
    )
    
    # Create unique collection name for each test
    test_name = request.node.name if hasattr(request, 'node') else 'unknown'
    unique_collection = f"test_memory_{test_name}_{uuid.uuid4().hex[:8]}"
    
    return RAGConfig(
        embedding=EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",  # Use real model
            device="cpu"
        ),
        vector_store=VectorStoreConfig(
            type="memory",
            collection_name=unique_collection,
            distance_metric="cosine"
        ),
        chunking=ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            chunking_method="words"
        ),
        search=SearchConfig(
            default_top_k=10,
            enable_cache=False
        )
    )


# === RAG Service Fixtures ===

@pytest.fixture
def mock_rag_service(test_rag_config, mock_embeddings, mock_vector_store):
    """Create a mock RAG service."""
    from tldw_chatbook.RAG_Search.simplified import create_rag_service_from_config
    
    service = create_rag_service_from_config(test_rag_config)
    service.embeddings = mock_embeddings
    service.vector_store = mock_vector_store
    return service


# === Async Fixtures ===

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# === Performance Fixtures ===

@pytest.fixture
def performance_timer():
    """Simple timer for performance measurements."""
    class Timer:
        def __init__(self):
            self.times = {}
            
        def start(self, name: str):
            self.times[name] = time.time()
            
        def stop(self, name: str) -> float:
            if name not in self.times:
                return 0
            elapsed = time.time() - self.times[name]
            del self.times[name]
            return elapsed
            
        def measure(self, name: str):
            """Context manager for timing."""
            class TimerContext:
                def __init__(self, timer, name):
                    self.timer = timer
                    self.name = name
                    self.elapsed = 0
                    
                def __enter__(self):
                    self.timer.start(self.name)
                    return self
                    
                def __exit__(self, *args):
                    self.elapsed = self.timer.stop(self.name)
                    
            return TimerContext(self, name)
    
    return Timer()


# === Test Data Generators ===

@pytest.fixture
def document_generator():
    """Generate test documents with various characteristics."""
    def generate(count: int, doc_type: str = "mixed") -> List[Dict[str, Any]]:
        documents = []
        for i in range(count):
            if doc_type == "short":
                content = f"Short document {i}."
            elif doc_type == "long":
                content = f"This is a long document number {i}. " * 50
            elif doc_type == "code":
                content = f"""
def function_{i}():
    '''Function that does something.'''
    result = compute_value({i})
    return result * 2
"""
            else:  # mixed
                content = f"Document {i} contains various information about topic {i}. " * 10
            
            documents.append({
                "id": f"doc_{i}",
                "title": f"Document {i}",
                "content": content,
                "metadata": {
                    "source": ["media", "notes", "conversations"][i % 3],
                    "index": i,
                    "type": doc_type
                }
            })
        return documents
    
    return generate


# === Test Markers ===

pytest.mark.slow = pytest.mark.slow
pytest.mark.integration = pytest.mark.integration
pytest.mark.unit = pytest.mark.unit
pytest.mark.property = pytest.mark.property

# Conditional markers based on dependencies
try:
    import chromadb
    import sentence_transformers
    import torch
    has_rag_deps = True
except ImportError:
    has_rag_deps = False

pytest.mark.requires_rag_deps = pytest.mark.skipif(
    not has_rag_deps,
    reason="RAG dependencies not available"
)

pytest.mark.requires_chromadb = pytest.mark.skipif(
    not has_rag_deps,
    reason="ChromaDB not available"
)


# === Utility Functions ===

def assert_valid_embedding(embedding: List[float], expected_dim: int = 384):
    """Assert that an embedding is valid."""
    assert isinstance(embedding, list), "Embedding must be a list"
    assert len(embedding) == expected_dim, f"Expected dimension {expected_dim}, got {len(embedding)}"
    assert all(isinstance(x, (int, float)) for x in embedding), "All values must be numeric"
    assert any(x != 0 for x in embedding), "Embedding should not be all zeros"


def assert_valid_search_result(result: Dict[str, Any]):
    """Assert that a search result has the expected structure."""
    assert "id" in result
    assert "content" in result
    assert "score" in result
    assert 0 <= result["score"] <= 1, "Score should be normalized"
    assert "metadata" in result