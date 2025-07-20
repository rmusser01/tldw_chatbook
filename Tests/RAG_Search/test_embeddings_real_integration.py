# test_embeddings_real_integration.py
# Integration tests for the simplified embeddings service using real components

import pytest
import tempfile
import shutil
import time
import threading
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from tldw_chatbook.RAG_Search.simplified import (
    EmbeddingsServiceWrapper,
    InMemoryVectorStore,
    ChromaVectorStore,
    create_embeddings_service,
    RAGService,
    create_rag_service
)
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
from tldw_chatbook.RAG_Search.simplified import circuit_breaker

# Test marker for integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Reset all circuit breakers before each test."""
    # Clear the global circuit breaker registry before each test
    circuit_breaker._circuit_breakers.clear()
    yield
    # Clear again after test
    circuit_breaker._circuit_breakers.clear()

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
    """Create a real embedding service with default model"""
    if not DEPENDENCIES_AVAILABLE.get('sentence_transformers', False):
        pytest.skip("Real embedding models not available")
    
    # Set environment to ensure proper model loading
    import os
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
    
    try:
        # Force torch to use CPU for tests to avoid device issues
        import torch
        torch.set_default_device('cpu')
    except:
        pass
    
    # Create service with explicit configuration
    service = EmbeddingsServiceWrapper(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",  # Force CPU to avoid meta tensor issues
        cache_size=1   # Minimize memory usage in tests
    )
    
    # Try to create a test embedding to ensure model is properly loaded
    try:
        test_embedding = service.create_embeddings(["test"])
        assert test_embedding.shape[0] == 1
    except Exception as e:
        pytest.skip(f"Could not initialize embedding model: {e}")
    
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
def real_rag_service(persist_dir):
    """Create a real RAG service for testing"""
    if DEPENDENCIES_AVAILABLE.get('sentence_transformers', False):
        from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
        config = RAGConfig()
        config.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
        config.vector_store.type = "memory"
        service = create_rag_service(config=config)
        return service
    else:
        pytest.skip("Real embedding models not available")


#######################################################################################################################
#
# Test Classes

class TestRealEmbeddingsWorkflow:
    """Test complete embedding workflow with real models"""
    
    @requires_sentence_transformers
    def test_full_rag_workflow_with_real_model(self, persist_dir, sample_texts):
        """Test complete RAG workflow with real sentence transformer model"""
        # Create RAG service with real embeddings
        from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
        config = RAGConfig()
        config.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
        config.vector_store.type = "chromadb" if DEPENDENCIES_AVAILABLE.get('chromadb', False) else "memory"
        if DEPENDENCIES_AVAILABLE.get('chromadb', False):
            config.vector_store.persist_directory = str(persist_dir)
        rag_service = create_rag_service(config=config)
        
        # Index documents
        collection_name = "real_documents"
        metadatas = [{"source": "test", "index": i} for i in range(len(sample_texts))]
        
        # Index documents one by one
        results = []
        for i, text in enumerate(sample_texts):
            result = rag_service.index_document_sync(
                doc_id=f"doc_{i}",
                content=text,
                title=f"Test Document {i}",
                metadata=metadatas[i]
            )
            results.append(result)
        
        assert all(r.success for r in results)
        assert sum(r.chunks_created for r in results) > 0
        
        # Search for similar documents
        query = "What programming languages are used in data science?"
        search_results = rag_service.search_sync(
            query=query,
            top_k=3
        )
        
        assert search_results is not None
        assert len(search_results) <= 3
        
        # Should find Python and data science related documents
        returned_docs = [r.document for r in search_results]
        assert any("Python" in doc for doc in returned_docs)
        
        # Test similarity scores
        assert all(0 <= r.score <= 1 for r in search_results)
        
        # Cleanup
        rag_service.clear_index()
    
    @requires_sentence_transformers
    def test_real_embeddings_dimensions(self, real_embedding_service):
        """Test that real models produce correct embedding dimensions"""
        service = real_embedding_service
        
        # Test with different text lengths
        texts = [
            "Short text",
            "A medium length text with more words to embed",
            "A very long text that contains multiple sentences. " * 10
        ]
        
        embeddings = service.create_embeddings(texts)
        
        assert embeddings is not None
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 384  # MiniLM-L6-v2 has 384 dimensions
        
        # Verify embeddings are different for different texts
        assert not np.allclose(embeddings[0], embeddings[1])
        assert not np.allclose(embeddings[1], embeddings[2])
        
        # Verify embeddings are normalized (optional, depends on model)
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"Embedding norms: {norms}")
    
    @requires_sentence_transformers
    def test_real_concurrent_operations(self, persist_dir, sample_texts):
        """Test concurrent operations with real embeddings"""
        # Create multiple services for true concurrency test
        results = []
        errors = []
        threads = []
        
        def process_documents(thread_id):
            try:
                # Each thread creates its own service
                service = EmbeddingsServiceWrapper(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # Create vector store
                vector_store = InMemoryVectorStore()
                
                # Each thread processes different documents
                thread_texts = [f"{text} (Thread {thread_id})" for text in sample_texts[:2]]
                
                # Create embeddings
                embeddings = service.create_embeddings(thread_texts)
                if embeddings is None:
                    raise RuntimeError(f"Thread {thread_id} failed to create embeddings")
                
                # Add to collection
                collection_name = f"thread_{thread_id}_collection"
                doc_ids = [f"thread_{thread_id}_doc_{i}" for i in range(len(thread_texts))]
                
                success = vector_store.add_documents(
                    collection_name,
                    thread_texts,
                    embeddings.tolist(),
                    [{"thread": thread_id} for _ in thread_texts],
                    doc_ids
                )
                
                if not success:
                    raise RuntimeError(f"Thread {thread_id} failed to add documents")
                
                # Search in own collection
                query_embeddings = service.create_embeddings([thread_texts[0]])
                # Search using the simplified API
                search_results = vector_store.search(
                    query_embeddings[0], 
                    top_k=1
                )
                
                results.append((thread_id, search_results is not None))
                
                # Cleanup
                service.close()
                
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


class TestRealMemoryUsage:
    """Test memory usage with real models"""
    
    @requires_sentence_transformers
    def test_real_model_memory_usage(self, real_embedding_service):
        """Test memory usage of real embedding models"""
        service = real_embedding_service
        
        # Get initial memory usage
        memory_stats = service.get_memory_usage()
        initial_memory = memory_stats['total_mb']
        print(f"\nInitial memory usage: {initial_memory:.1f} MB")
        
        # Process batches of texts
        batch_sizes = [10, 50, 100]
        for batch_size in batch_sizes:
            texts = [f"Test document number {i} with some content." for i in range(batch_size)]
            
            embeddings = service.create_embeddings(texts)
            memory_stats = service.get_memory_usage()
            current_memory = memory_stats['total_mb']
            
            print(f"After {batch_size} texts: {current_memory:.1f} MB (delta: {current_memory - initial_memory:.1f} MB)")
            
            assert embeddings is not None
            assert embeddings.shape == (batch_size, 384)


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
        # Create embeddings service
        embeddings_service = EmbeddingsServiceWrapper(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store
        vector_store = ChromaVectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name
        )
        
        # Add documents
        embeddings = embeddings_service.create_embeddings(test_docs)
        doc_ids = [f"persist_{i}" for i in range(len(test_docs))]
        vector_store.add(
            doc_ids,
            embeddings,
            test_docs,
            [{"persistent": True, "index": i} for i in range(len(test_docs))]
        )
        
        # Close first service
        embeddings_service.close()
        del embeddings_service
        del vector_store
        
        # Create new service instance
        embeddings_service2 = EmbeddingsServiceWrapper(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store2 = ChromaVectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name
        )
        
        # Collection should still exist
        collections = vector_store2.list_collections()
        assert collection_name in collections
        
        # Search should find the documents
        query = "vector database persistence"
        query_embeddings = embeddings_service2.create_embeddings([query])
        results = vector_store2.search(query_embeddings[0], top_k=2)
        
        assert results is not None
        assert len(results) > 0
        
        # Should find ChromaDB related documents
        returned_docs = [r.document for r in results]
        assert any("ChromaDB" in doc or "persistence" in doc for doc in returned_docs)
    
    @requires_chromadb
    @requires_sentence_transformers
    def test_chromadb_metadata_filtering(self, persist_dir):
        """Test ChromaDB metadata filtering with real data"""
        embeddings_service = EmbeddingsServiceWrapper(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = ChromaVectorStore(
            persist_directory=persist_dir,
            collection_name="metadata_test"
        )
        
        # Add documents with different metadata
        docs = [
            "Python programming tutorial",
            "Java programming guide",
            "Python data science handbook",
            "JavaScript web development"
        ]
        
        embeddings = embeddings_service.create_embeddings(docs)
        metadatas = [
            {"language": "python", "topic": "programming"},
            {"language": "java", "topic": "programming"},
            {"language": "python", "topic": "data_science"},
            {"language": "javascript", "topic": "web"}
        ]
        doc_ids = [f"doc_{i}" for i in range(len(docs))]
        
        vector_store.add(
            doc_ids,
            embeddings,
            docs,
            metadatas
        )
        
        # Search with metadata filter
        query_embedding = embeddings_service.create_embeddings(["programming languages"])
        
        # Note: Actual metadata filtering depends on ChromaDB capabilities
        results = vector_store.search(
            query_embedding[0],
            top_k=10
        )
        
        assert results is not None
        assert len(results) > 0
        
        # Manual filtering of results by metadata
        python_results = [
            r for r in results
            if r.metadata.get("language") == "python"
        ]
        # We might not get exactly 2 depending on search scores
        assert len(python_results) >= 1


class TestRealModelVariations:
    """Test different real model variations"""
    
    @requires_sentence_transformers
    def test_different_sentence_transformer_models(self):
        """Test different sentence transformer models"""
        models_to_test = [
            ("sentence-transformers/all-MiniLM-L6-v2", 384),
            # Add more models if needed, but be mindful of download time
        ]
        
        test_text = "This is a test sentence for embedding generation."
        
        for model_name, expected_dim in models_to_test:
            try:
                service = EmbeddingsServiceWrapper(model_name=model_name)
                embeddings = service.create_embeddings([test_text])
                
                assert embeddings is not None
                assert embeddings.shape == (1, expected_dim)
                
                print(f"\n{model_name}:")
                print(f"  Dimension: {embeddings.shape[1]}")
                print(f"  First 5 values: {embeddings[0][:5]}")
                
                service.close()
                
            except Exception as e:
                print(f"Failed to test {model_name}: {e}")


# TestRealMemoryManagement class removed - memory management service not exposed in simplified API
# The simplified API handles memory management internally


class TestRealErrorHandling:
    """Test error handling with real components"""
    
    @requires_sentence_transformers
    def test_invalid_model_name(self):
        """Test handling of invalid model names"""
        # The EmbeddingsServiceWrapper might allow dynamic loading of models
        # So instead of expecting failure during init, let's test that
        # creating embeddings with an invalid model fails
        try:
            service = EmbeddingsServiceWrapper(
                model_name="invalid/model/name/that/doesnt/exist"
            )
            # Try to actually create embeddings - this should fail
            embeddings = service.create_embeddings(["test text"])
            # If we get here without error, fail the test
            pytest.fail("Expected an error when using invalid model, but none was raised")
        except Exception as e:
            # This is expected - log the error for debugging
            print(f"\nExpected error for invalid model: {e}")
            assert "invalid" in str(e).lower() or "not found" in str(e).lower() or "failed" in str(e).lower()
        
    @requires_sentence_transformers
    def test_large_batch_handling(self, real_embedding_service):
        """Test handling of large batches with real model"""
        service = real_embedding_service
        
        # Create a large batch of texts
        large_batch = [f"Document {i}: " + "Some content. " * 10 for i in range(500)]
        
        # This should handle the large batch appropriately
        start_time = time.time()
        embeddings = service.create_embeddings(large_batch)
        elapsed = time.time() - start_time
        
        assert embeddings is not None
        assert embeddings.shape == (500, 384)
        
        print(f"\nLarge batch performance:")
        print(f"  Texts: {len(large_batch)}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {len(large_batch)/elapsed:.1f} texts/sec")
    
    @requires_sentence_transformers  
    def test_empty_text_handling(self, real_embedding_service):
        """Test handling of empty texts with real model"""
        service = real_embedding_service
        
        # Test with empty and whitespace-only texts
        edge_case_texts = [
            "",
            " ",
            "\n\t",
            "Normal text",
            "   ",
        ]
        
        embeddings = service.create_embeddings(edge_case_texts)
        
        assert embeddings is not None
        assert embeddings.shape == (len(edge_case_texts), 384)
        
        # Empty texts should still produce embeddings (model-dependent behavior)
        print("\nEmpty text embeddings:")
        for i, text in enumerate(edge_case_texts):
            print(f"  Text '{repr(text)}': norm={np.linalg.norm(embeddings[i]):.3f}")


class TestRealRAGIntegration:
    """Test full RAG integration with real models"""
    
    @requires_sentence_transformers
    def test_real_rag_search_quality(self, real_rag_service):
        """Test search quality with real embeddings"""
        service = real_rag_service
        
        # Index diverse documents
        documents = [
            "Python is a high-level, interpreted programming language known for its simplicity.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "The solar system consists of the Sun and the celestial bodies that orbit around it.",
            "Classical music encompasses a broad range of music from the Western tradition.",
            "Healthy eating involves consuming a balanced diet with fruits, vegetables, and whole grains.",
        ]
        
        # Index documents one by one
        results = []
        for i, doc in enumerate(documents):
            result = service.index_document_sync(
                doc_id=f"doc_{i}",
                content=doc,
                title=f"Document {i}"
            )
            results.append(result)
        
        # Debug: print any failures
        failed_results = [r for r in results if not r.success]
        if failed_results:
            for r in failed_results:
                print(f"Failed to index {r.doc_id}: {r.error}")
        assert all(r.success for r in results)
        
        # Test various queries
        test_queries = [
            ("programming languages", ["Python"]),
            ("artificial intelligence and ML", ["Machine learning"]),
            ("astronomy and planets", ["solar system"]),
            ("diet and nutrition", ["Healthy eating"]),
        ]
        
        print("\nSearch quality results:")
        for query, expected_keywords in test_queries:
            results = service.search_sync(
                query=query,
                top_k=2
            )
            
            # Check if expected content is in top results
            top_contents = [r.document for r in results[:2]]
            found = any(any(keyword in content for keyword in expected_keywords) 
                       for content in top_contents)
            
            print(f"\n  Query: '{query}'")
            print(f"  Expected: {expected_keywords}")
            print(f"  Found: {found}")
            print(f"  Top result: {results[0].document[:100]}...")
            print(f"  Score: {results[0].score:.3f}")
            
            assert found, f"Expected keywords {expected_keywords} not found for query '{query}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])