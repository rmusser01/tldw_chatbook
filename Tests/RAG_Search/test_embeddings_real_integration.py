# test_embeddings_real_integration.py
# Integration tests for embeddings service using real components

import pytest
import tempfile
import shutil
import time
import threading
import os
from pathlib import Path
from typing import List, Dict, Any

from tldw_chatbook.RAG_Search.Services.embeddings_service import (
    EmbeddingsService,
    SentenceTransformerProvider,
    HuggingFaceProvider,
    OpenAIProvider,
    ChromaDBStore,
    InMemoryStore
)
from tldw_chatbook.RAG_Search.Services.cache_service import CacheService
from tldw_chatbook.RAG_Search.Services.memory_management_service import MemoryManagementService
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Test marker for integration tests
pytestmark = pytest.mark.integration

# Skip tests if required dependencies are not available
requires_sentence_transformers = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE.get('sentence_transformers', False),
    reason="sentence-transformers not installed"
)
requires_chromadb = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE.get('chromadb', False),
    reason="chromadb not installed"
)
requires_torch = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE.get('torch', False),
    reason="torch not installed"
)

#######################################################################################################################
#
# Fixtures

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


@pytest.fixture
def cache_dir(temp_dir):
    """Create a cache directory"""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


@pytest.fixture
def real_embedding_service(persist_dir):
    """Create a real embedding service with default provider"""
    service = EmbeddingsService(persist_directory=persist_dir)
    # Initialize with a small model for testing
    if DEPENDENCIES_AVAILABLE.get('sentence_transformers', False):
        service.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    else:
        # Fallback to in-memory store if no embedding providers available
        service.vector_store = InMemoryStore()
    return service


@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing deals with text data.",
        "Deep learning uses neural networks with multiple layers.",
        "Data science combines statistics and programming."
    ]


@pytest.fixture
def real_cache_service(cache_dir):
    """Create a real cache service"""
    return CacheService(cache_dir=cache_dir)


@pytest.fixture
def real_memory_service():
    """Create a real memory management service"""
    return MemoryManagementService()


#######################################################################################################################
#
# Test Classes

class TestRealEmbeddingsWorkflow:
    """Test complete embedding workflow with real providers"""
    
    @requires_sentence_transformers
    def test_full_rag_workflow_with_real_model(self, persist_dir, sample_texts):
        """Test complete RAG workflow with real sentence transformer model"""
        # Create service with real ChromaDB if available
        if DEPENDENCIES_AVAILABLE.get('chromadb', False):
            from tldw_chatbook.RAG_Search.Services.embeddings_service import ChromaDBStore
            import chromadb
            from chromadb.config import Settings
            settings = Settings(anonymized_telemetry=False, allow_reset=True)
            client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)
            vector_store = ChromaDBStore(client)
        else:
            vector_store = InMemoryStore()
        
        service = EmbeddingsService(
            persist_directory=persist_dir,
            vector_store=vector_store
        )
        
        # Initialize a real embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        assert service.initialize_embedding_model(model_name)
        
        # Create embeddings with real model
        embeddings = service.create_embeddings(sample_texts)
        assert embeddings is not None
        assert len(embeddings) == len(sample_texts)
        assert all(len(emb) == 384 for emb in embeddings)  # MiniLM-L6-v2 dimension
        
        # Add to collection
        collection_name = "real_documents"
        doc_ids = [f"doc_{i}" for i in range(len(sample_texts))]
        metadatas = [{"source": "test", "index": i} for i in range(len(sample_texts))]
        
        success = service.add_documents_to_collection(
            collection_name,
            sample_texts,
            embeddings,
            metadatas,
            doc_ids
        )
        assert success
        
        # Search for similar documents
        query = "What programming languages are used in data science?"
        query_embeddings = service.create_embeddings([query])
        
        results = service.search_collection(
            collection_name,
            query_embeddings,
            n_results=3
        )
        
        assert results is not None
        assert "ids" in results
        assert len(results["ids"][0]) <= 3
        
        # Should find Python and data science related documents
        returned_docs = results["documents"][0]
        assert any("Python" in doc for doc in returned_docs)
        
        # Cleanup
        service.delete_collection(collection_name)
    
    @requires_sentence_transformers
    @requires_torch
    def test_multiple_real_providers(self, persist_dir):
        """Test using multiple real embedding providers"""
        service = EmbeddingsService(persist_directory=persist_dir)
        
        # Add multiple real providers
        st_provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        service.add_provider("minilm", st_provider)
        
        # Try to add HuggingFace provider if available
        if DEPENDENCIES_AVAILABLE.get('transformers', False):
            hf_provider = HuggingFaceProvider("sentence-transformers/all-MiniLM-L6-v2")
            service.add_provider("huggingface", hf_provider)
        
        # Create embeddings with different providers
        test_text = ["Machine learning with Python is powerful"]
        
        # Use sentence transformers provider
        service.set_provider("minilm")
        st_embeddings = service.create_embeddings(test_text)
        assert st_embeddings is not None
        assert len(st_embeddings[0]) == 384
        
        # If HuggingFace is available, compare embeddings
        if "huggingface" in service.providers:
            service.set_provider("huggingface")
            hf_embeddings = service.create_embeddings(test_text)
            assert hf_embeddings is not None
            # Embeddings should be similar but not identical due to implementation differences
            assert len(hf_embeddings[0]) == len(st_embeddings[0])
    
    @requires_sentence_transformers
    def test_real_concurrent_operations(self, persist_dir, sample_texts):
        """Test concurrent operations with real embeddings"""
        service = EmbeddingsService(persist_directory=persist_dir)
        service.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        
        results = []
        errors = []
        threads = []
        
        def process_documents(thread_id):
            try:
                # Each thread processes different documents
                thread_texts = [f"{text} (Thread {thread_id})" for text in sample_texts[:2]]
                
                # Create embeddings
                embeddings = service.create_embeddings(thread_texts)
                if embeddings is None:
                    raise RuntimeError(f"Thread {thread_id} failed to create embeddings")
                
                # Add to collection
                collection_name = f"thread_{thread_id}_collection"
                doc_ids = [f"thread_{thread_id}_doc_{i}" for i in range(len(thread_texts))]
                
                success = service.add_documents_to_collection(
                    collection_name,
                    thread_texts,
                    embeddings,
                    [{"thread": thread_id} for _ in thread_texts],
                    doc_ids
                )
                
                if not success:
                    raise RuntimeError(f"Thread {thread_id} failed to add documents")
                
                # Search in own collection
                query_embeddings = service.create_embeddings([thread_texts[0]])
                search_results = service.search_collection(
                    collection_name, 
                    query_embeddings, 
                    n_results=1
                )
                
                results.append((thread_id, search_results is not None))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run concurrent operations
        for i in range(3):  # Use 3 threads for faster testing
            thread = threading.Thread(target=process_documents, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=30)
        
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert all(result[1] for result in results)  # All searches successful
        
        # Verify all collections were created
        collections = service.list_collections()
        thread_collections = [c for c in collections if c.startswith("thread_")]
        assert len(thread_collections) == 3


class TestRealChromaDBIntegration:
    """Test integration with real ChromaDB"""
    
    @requires_chromadb
    @requires_sentence_transformers
    def test_chromadb_real_persistence(self, persist_dir):
        """Test that ChromaDB really persists data across service instances"""
        collection_name = "persistent_real_collection"
        test_docs = [
            "ChromaDB is a vector database",
            "It provides persistence for embeddings",
            "Vector search is efficient"
        ]
        
        # First service instance
        service1 = EmbeddingsService(persist_directory=str(persist_dir))
        service1.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        
        # Add documents
        embeddings = service1.create_embeddings(test_docs)
        doc_ids = [f"persist_{i}" for i in range(len(test_docs))]
        service1.add_documents_to_collection(
            collection_name,
            test_docs,
            embeddings,
            [{"persistent": True, "index": i} for i in range(len(test_docs))],
            doc_ids
        )
        
        # Close first service
        del service1
        
        # Create new service instance
        service2 = EmbeddingsService(persist_directory=str(persist_dir))
        service2.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        
        # Collection should still exist
        collections = service2.list_collections()
        assert collection_name in collections
        
        # Search should find the documents
        query = "vector database persistence"
        query_embeddings = service2.create_embeddings([query])
        results = service2.search_collection(collection_name, query_embeddings, n_results=2)
        
        assert results is not None
        assert len(results["ids"][0]) > 0
        
        # Should find ChromaDB related documents
        returned_docs = results["documents"][0]
        assert any("ChromaDB" in doc or "persistence" in doc for doc in returned_docs)
    
    @requires_chromadb
    @requires_sentence_transformers
    def test_chromadb_metadata_filtering(self, persist_dir):
        """Test ChromaDB metadata filtering with real data"""
        service = EmbeddingsService(persist_directory=persist_dir)
        service.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        
        # Add documents with different metadata
        docs = [
            "Python programming tutorial",
            "Java programming guide",
            "Python data science handbook",
            "JavaScript web development"
        ]
        
        embeddings = service.create_embeddings(docs)
        metadatas = [
            {"language": "python", "topic": "programming"},
            {"language": "java", "topic": "programming"},
            {"language": "python", "topic": "data_science"},
            {"language": "javascript", "topic": "web"}
        ]
        doc_ids = [f"doc_{i}" for i in range(len(docs))]
        
        service.add_documents_to_collection(
            "filtered_collection",
            docs,
            embeddings,
            metadatas,
            doc_ids
        )
        
        # Search with metadata filter
        query_embedding = service.create_embeddings(["programming languages"])
        
        # Note: Actual metadata filtering depends on ChromaDB capabilities
        results = service.search_collection(
            "filtered_collection",
            query_embedding,
            n_results=10
        )
        
        assert results is not None
        assert len(results["ids"][0]) > 0
        
        # Manual filtering of results by metadata
        python_results = [
            (doc, meta) for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            if meta.get("language") == "python"
        ]
        assert len(python_results) == 2


class TestRealCacheIntegration:
    """Test real cache service integration"""
    
    @requires_sentence_transformers
    def test_embeddings_with_real_cache(self, persist_dir, cache_dir, sample_texts):
        """Test embedding service with real cache"""
        # Create cache service
        cache_service = CacheService(cache_dir=cache_dir)
        
        # Create embedding service
        service = EmbeddingsService(
            persist_directory=persist_dir
        )
        # Cache service is created internally by EmbeddingsService
        service.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        
        # First call - should create embeddings
        start_time = time.time()
        embeddings1 = service.create_embeddings(sample_texts)
        first_call_time = time.time() - start_time
        
        assert embeddings1 is not None
        
        # Second call - should use cache
        start_time = time.time()
        embeddings2 = service.create_embeddings(sample_texts)
        second_call_time = time.time() - start_time
        
        assert embeddings2 is not None
        
        # Cache should make second call faster
        assert second_call_time < first_call_time * 0.5
        
        # Embeddings should be identical
        for e1, e2 in zip(embeddings1, embeddings2):
            assert e1 == e2
        
        # Check cache stats
        stats = cache_service.get_stats()
        assert stats["hits"] > 0
        assert stats["misses"] > 0
        assert stats["hit_rate"] > 0


class TestRealMemoryManagement:
    """Test real memory management integration"""
    
    @requires_sentence_transformers
    def test_memory_cleanup_with_real_embeddings(self, persist_dir):
        """Test memory cleanup with real embedding operations"""
        # Create memory service with low threshold for testing
        memory_service = MemoryManagementService()
        
        # Create embedding service
        service = EmbeddingsService(
            persist_directory=persist_dir
        )
        # Set memory manager after creation
        service.set_memory_manager(memory_service)
        service.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        
        # Generate many embeddings to trigger cleanup
        large_texts = []
        for i in range(100):
            large_texts.append(f"This is document number {i} with some content to embed.")
        
        # Process in batches
        batch_size = 10
        for i in range(0, len(large_texts), batch_size):
            batch = large_texts[i:i+batch_size]
            embeddings = service.create_embeddings(batch)
            
            # Add to collection to use memory
            service.add_documents_to_collection(
                f"collection_{i//batch_size}",
                batch,
                embeddings,
                [{"batch": i//batch_size} for _ in batch],
                [f"doc_{j}" for j in range(i, i+len(batch))]
            )
        
        # Memory cleanup should have been triggered
        # Check that service is still functional
        test_embedding = service.create_embeddings(["Test after cleanup"])
        assert test_embedding is not None


class TestRealErrorHandling:
    """Test error handling with real components"""
    
    @requires_sentence_transformers
    def test_invalid_model_name(self, persist_dir):
        """Test handling of invalid model names"""
        service = EmbeddingsService(persist_directory=persist_dir)
        
        # Try to initialize with invalid model
        success = service.initialize_embedding_model("invalid/model/name/that/doesnt/exist")
        assert not success
        
        # Service should still be usable with a valid model
        success = service.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        assert success
    
    @requires_sentence_transformers
    def test_corrupted_embeddings_handling(self, persist_dir):
        """Test handling of corrupted data"""
        service = EmbeddingsService(persist_directory=persist_dir)
        service.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        
        # Create valid embeddings
        texts = ["Valid text"]
        embeddings = service.create_embeddings(texts)
        
        # Corrupt the embeddings
        corrupted_embeddings = [[float('nan')] * len(embeddings[0])]
        
        # Try to add corrupted embeddings
        success = service.add_documents_to_collection(
            "corrupted_collection",
            texts,
            corrupted_embeddings,
            [{"corrupted": True}],
            ["corrupt_1"]
        )
        
        # Should handle gracefully (either reject or sanitize)
        # Service should still be functional
        new_embeddings = service.create_embeddings(["New text after corruption"])
        assert new_embeddings is not None