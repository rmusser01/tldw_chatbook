# test_embeddings_properties.py
# Property-based tests for the simplified embeddings service using Hypothesis

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition, invariant, Bundle
import threading
import time
import numpy as np
from typing import List, Dict, Any, Set
import hashlib
from unittest.mock import patch, Mock, MagicMock

from tldw_chatbook.RAG_Search.simplified import (
    EmbeddingsServiceWrapper,
    InMemoryVectorStore,
    create_embeddings_service
)

# Import test utilities from conftest
from .conftest import requires_numpy


# Hypothesis settings for embeddings tests
settings.register_profile(
    "embeddings",
    deadline=5000,  # 5 seconds for complex operations
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile("embeddings")


# Custom strategies for embeddings domain
@st.composite
def valid_text(draw):
    """Generate valid text for embeddings"""
    # Mix of different text types
    text_type = draw(st.sampled_from(["normal", "empty", "special", "long", "unicode"]))
    
    if text_type == "empty":
        return ""
    elif text_type == "special":
        return draw(st.text(alphabet="!@#$%^&*()_+-=[]{}|;:,.<>?", min_size=1, max_size=100))
    elif text_type == "long":
        return draw(st.text(min_size=500, max_size=1000))
    elif text_type == "unicode":
        return draw(st.text(alphabet=st.characters(min_codepoint=0x100, max_codepoint=0x1000), min_size=1, max_size=50))
    else:
        return draw(st.text(min_size=1, max_size=200))


@st.composite
def text_batch(draw, min_size=1, max_size=10):
    """Generate a batch of texts"""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return draw(st.lists(valid_text(), min_size=size, max_size=size))


@st.composite
def embedding_dimension(draw):
    """Generate valid embedding dimensions"""
    return draw(st.sampled_from([128, 256, 384, 512, 768, 1024, 1536]))


@st.composite
def provider_config(draw):
    """Generate provider configuration"""
    return {
        "dimension": draw(embedding_dimension()),
        "delay": draw(st.floats(min_value=0, max_value=0.01)),  # Small delays for testing
        "fail_after": draw(st.one_of(st.none(), st.integers(min_value=10, max_value=100)))
    }


class TestEmbeddingProperties:
    """Property-based tests for embedding invariants"""
    
    @given(texts=text_batch())
    def test_embedding_determinism(self, texts):
        """Same text should always produce same embedding"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Create deterministic mock embeddings
            def deterministic_embed(input_texts, as_list=True):
                # Use hash of text to generate deterministic embeddings
                return np.array([[hash(text) % 100 / 100.0 + i * 0.01 for i in range(384)] for text in input_texts])
            
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = deterministic_embed
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            
            # Create embeddings twice
            embeddings1 = service.create_embeddings(texts)
            embeddings2 = service.create_embeddings(texts)
            
            # Should be identical
            assert embeddings1 is not None
            assert embeddings2 is not None
            assert embeddings1.shape == embeddings2.shape
            assert np.allclose(embeddings1, embeddings2)
            
            service.close()
    
    @given(
        texts=text_batch(min_size=1, max_size=5),
        dimension=embedding_dimension()
    )
    def test_embedding_dimension_consistency(self, texts, dimension):
        """All embeddings should have consistent dimension"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Mock embeddings with specified dimension
            mock_instance = MagicMock()
            mock_instance.embed.return_value = np.array([[0.1] * dimension for _ in texts])
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            embeddings = service.create_embeddings(texts)
            
            assert embeddings is not None
            assert embeddings.shape == (len(texts), dimension)
            assert embeddings.shape[1] == dimension  # All have same dimension
            
            service.close()
    
    @given(texts=text_batch())
    def test_embedding_count_matches_input(self, texts):
        """Number of embeddings should match number of input texts"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            mock_instance = MagicMock()
            mock_instance.embed.return_value = np.array([[0.1] * 384 for _ in texts])
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            embeddings = service.create_embeddings(texts)
            
            assert embeddings is not None
            assert embeddings.shape[0] == len(texts)
            
            service.close()
    
    @given(
        texts=st.lists(valid_text(), min_size=10, max_size=50),
        batch_size=st.integers(min_value=1, max_value=10)
    )
    def test_batch_processing_consistency(self, texts, batch_size):
        """Batch processing should produce same results regardless of batch size"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Create consistent embeddings regardless of batch size
            def consistent_embed(input_texts, as_list=True):
                return np.array([[hash(text) % 100 / 100.0 + i * 0.01 for i in range(384)] for text in input_texts])
            
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = consistent_embed
            mock_factory.return_value = mock_instance
            
            # Process all at once
            service1 = EmbeddingsServiceWrapper()
            embeddings_single = service1.create_embeddings(texts)
            service1.close()
            
            # Process in smaller batches (simulated by multiple calls)
            service2 = EmbeddingsServiceWrapper()
            embeddings_batched = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = service2.create_embeddings(batch)
                embeddings_batched.append(batch_embeddings)
            
            # Concatenate batched results
            embeddings_batch = np.vstack(embeddings_batched) if embeddings_batched else np.array([])
            service2.close()
            
            # Results should be identical
            assert embeddings_single.shape == embeddings_batch.shape
            assert np.allclose(embeddings_single, embeddings_batch)
    
    @given(
        texts=text_batch(),
        num_models=st.integers(min_value=2, max_value=5)
    )
    def test_model_isolation(self, texts, num_models):
        """Different models should produce different embeddings"""
        # Test that different model configurations produce different results
        dimensions = [384, 512, 768, 1024, 1536]
        model_names = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "bert-base-uncased",
            "openai/text-embedding-3-small",
            "intfloat/e5-small-v2"
        ]
        
        results = {}
        
        for i in range(num_models):
            with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
                dimension = dimensions[i % len(dimensions)]
                
                # Each model produces different embeddings
                mock_instance = MagicMock()
                mock_instance.embed.return_value = np.array([
                    [float(i) + j * 0.01 for j in range(dimension)] for _ in texts
                ])
                mock_factory.return_value = mock_instance
                
                service = EmbeddingsServiceWrapper(model_name=model_names[i % len(model_names)])
                embeddings = service.create_embeddings(texts)
                results[f"model_{i}"] = embeddings
                service.close()
        
        # Each model should produce different dimensional embeddings
        for i in range(num_models):
            expected_dim = dimensions[i % len(dimensions)]
            assert results[f"model_{i}"].shape[1] == expected_dim


class TestVectorStoreProperties:
    """Property-based tests for vector store operations"""
    
    @given(
        collection_names=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=["L", "N"]), min_size=1, max_size=50),
            min_size=1,
            max_size=5,
            unique=True
        )
    )
    def test_collection_lifecycle(self, collection_names):
        """Collections should be created, listed, and deleted correctly"""
        store = InMemoryVectorStore()
        
        # Create collections
        for name in collection_names:
            success = store.add_documents(
                name,
                ["test doc"],
                [[0.1, 0.2]],
                [{"test": True}],
                ["doc1"]
            )
            assert success
        
        # All collections should be listed
        listed = store.list_collections()
        assert set(listed) == set(collection_names)
        
        # Delete collections
        for name in collection_names:
            assert store.delete_collection(name)
        
        # No collections should remain
        assert len(store.list_collections()) == 0
    
    @given(
        texts=text_batch(min_size=1, max_size=20),
        n_results=st.integers(min_value=1, max_value=10)
    )
    def test_search_result_bounds(self, texts, n_results):
        """Search should never return more than n_results"""
        # Import MockEmbeddingProvider from conftest
        from .conftest import MockEmbeddingProvider
        
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Use MockEmbeddingProvider for consistent behavior
            provider = MockEmbeddingProvider()
            
            def mock_embed(input_texts, as_list=True):
                embeddings = provider.create_embeddings(input_texts)
                return np.array(embeddings)
            
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = mock_embed
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            vector_store = InMemoryVectorStore()
        
            # Add documents
            embeddings = service.create_embeddings(texts)
            doc_ids = [f"doc_{i}" for i in range(len(texts))]
            
            vector_store.add_documents(
                "test_collection",
                texts,
                embeddings,
                [{"id": i} for i in range(len(texts))],
                doc_ids
            )
            
            # Search
            query_embeddings = service.create_embeddings(["query"])
            results = vector_store.search(
                query_embeddings[0],
                top_k=n_results
            )
            
            assert results is not None
            assert len(results) <= min(n_results, len(texts))
            
            service.close()
    
    @given(
        doc_ids=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=5,
            max_size=10,
            unique=True
        )
    )
    def test_document_id_uniqueness(self, doc_ids):
        """Document IDs should be unique within a collection"""
        store = InMemoryVectorStore()
        
        # Add documents with unique IDs
        texts = [f"Document {id}" for id in doc_ids]
        embeddings = [[float(i), float(i+1)] for i in range(len(doc_ids))]
        metadatas = [{"id": id} for id in doc_ids]
        
        success = store.add_documents(
            "test_collection",
            texts,
            embeddings,
            metadatas,
            doc_ids
        )
        assert success
        
        # Verify all documents are stored
        collection = store._collections["test_collection"]
        assert len(collection["ids"]) == len(doc_ids)
        assert len(set(collection["ids"])) == len(doc_ids)  # All unique


class TestThreadSafetyProperties:
    """Property-based tests for thread safety"""
    
    @given(
        num_threads=st.integers(min_value=2, max_value=10),
        texts_per_thread=st.integers(min_value=1, max_value=5)
    )
    def test_concurrent_embedding_creation(self, num_threads, texts_per_thread):
        """Concurrent embedding creation should be thread-safe"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Thread-safe mock
            lock = threading.Lock()
            call_count = 0
            
            def thread_safe_embed(texts, as_list=True):
                nonlocal call_count
                with lock:
                    call_count += 1
                    # Small delay to increase chance of race conditions
                    time.sleep(0.001)
                return np.array([[float(call_count) + i * 0.01 for i in range(384)] for _ in texts])
            
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = thread_safe_embed
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            
            results = []
            errors = []
            threads = []
            
            def create_embeddings(thread_id):
                try:
                    texts = [f"Thread {thread_id} text {i}" for i in range(texts_per_thread)]
                    embeddings = service.create_embeddings(texts)
                    results.append((thread_id, embeddings))
                except Exception as e:
                    errors.append((thread_id, e))
            
            # Start threads
            for i in range(num_threads):
                thread = threading.Thread(target=create_embeddings, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=10)
            
            # All threads should succeed
            assert len(errors) == 0
            assert len(results) == num_threads
            
            # Each thread should get correct number of embeddings
            for thread_id, embeddings in results:
                assert embeddings.shape == (texts_per_thread, 384)
            
            service.close()
    
    @given(
        num_operations=st.integers(min_value=5, max_value=20)
    )
    def test_concurrent_service_creation(self, num_operations):
        """Concurrent service creation should be thread-safe"""
        results = []
        errors = []
        threads = []
        
        def create_service_and_embed(op_id):
            try:
                with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
                    mock_instance = MagicMock()
                    mock_instance.embed.return_value = np.array([[0.1] * 384])
                    mock_factory.return_value = mock_instance
                    
                    # Create a new service instance
                    service = EmbeddingsServiceWrapper(
                        model_name=f"model_{op_id % 3}"  # Vary models
                    )
                    
                    # Try to create embeddings
                    embeddings = service.create_embeddings([f"test {op_id}"])
                    results.append((op_id, embeddings is not None))
                    
                    service.close()
            except Exception as e:
                errors.append((op_id, e))
        
        # Execute operations concurrently
        for i in range(num_operations):
            thread = threading.Thread(target=create_service_and_embed, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # All operations should succeed
        assert len(errors) == 0
        assert len(results) == num_operations
        assert all(success for _, success in results)


class EmbeddingServiceStateMachine(RuleBasedStateMachine):
    """Stateful testing for EmbeddingsServiceWrapper with vector stores"""
    
    def __init__(self):
        super().__init__()
        self.vector_store = InMemoryVectorStore()
        self.service = None
        self.collections = set()
        self.documents = {}  # collection -> list of doc ids
        self._init_service()
    
    def _init_service(self):
        """Initialize service with mock factory"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            mock_instance = MagicMock()
            # Return consistent embeddings based on text hash
            def mock_embed(texts, as_list=True):
                return [[hash(text) % 100 / 100.0 + i * 0.01 for i in range(384)] for text in texts]
            mock_instance.embed.side_effect = mock_embed
            mock_factory.return_value = mock_instance
            
            self.service = EmbeddingsServiceWrapper()
            # Keep the mock alive
            self._mock_factory = mock_factory
            self._mock_instance = mock_instance
    
    collections_bundle = Bundle('collections')
    
    @rule(
        texts=text_batch(),
        collection_name=st.text(min_size=1, max_size=20)
    )
    def create_and_store_embeddings(self, texts, collection_name):
        """Create embeddings and store in collection"""
        if not self.service:
            self._init_service()
        
        embeddings = self.service.create_embeddings(texts)
        assert embeddings is not None
        assert embeddings.shape[0] == len(texts)
        
        # Store in vector store
        doc_ids = [f"{collection_name}_doc_{i}_{hash(texts[i])}" for i in range(len(texts))]
        success = self.vector_store.add_documents(
            collection_name,
            texts,
            embeddings.tolist(),  # Convert numpy to list
            [{"text": t} for t in texts],
            doc_ids
        )
        assert success
        
        self.collections.add(collection_name)
        if collection_name not in self.documents:
            self.documents[collection_name] = []
        self.documents[collection_name].extend(doc_ids)
        
        return collection_name
    
    @rule(collection_name=collections_bundle)
    def search_collection(self, collection_name):
        """Search in an existing collection"""
        query_embeddings = self.service.create_embeddings(["search query"])
        results = self.vector_store.search(
            collection_name,
            query_embeddings.tolist(),
            n_results=5
        )
        
        assert results is not None
        assert "ids" in results
    
    @rule(collection_name=collections_bundle)
    def delete_collection(self, collection_name):
        """Delete a collection"""
        success = self.vector_store.delete_collection(collection_name)
        assert success
        
        self.collections.remove(collection_name)
        if collection_name in self.documents:
            del self.documents[collection_name]
    
    @invariant()
    def collections_consistency(self):
        """Listed collections should match tracked collections"""
        listed = set(self.vector_store.list_collections())
        # Collections should match exactly
        assert self.collections == listed
    
    def teardown(self):
        """Clean up after test"""
        if self.service:
            self.service.close()


class TestEmbeddingServiceWrapper:
    """Additional property tests for EmbeddingsServiceWrapper"""
    
    @given(texts=st.lists(valid_text(), min_size=0, max_size=10))
    def test_empty_and_edge_cases(self, texts):
        """Test handling of empty lists and edge cases"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            mock_instance = MagicMock()
            mock_instance.embed.return_value = np.array([[0.1] * 384 for _ in texts]) if texts else np.array([])
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            embeddings = service.create_embeddings(texts)
            
            assert embeddings is not None
            if len(texts) == 0:
                assert embeddings.shape == (0, 384)  # Empty but with correct dimension
            else:
                assert embeddings.shape == (len(texts), 384)
            
            service.close()
    
    @given(
        model_name=st.sampled_from([
            "sentence-transformers/all-MiniLM-L6-v2",
            "openai/text-embedding-3-small",
            "bert-base-uncased"
        ])
    )
    def test_model_name_handling(self, model_name):
        """Test that different model names are handled correctly"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            mock_instance = MagicMock()
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper(model_name=model_name)
            
            assert service.model_name == model_name
            
            # Verify factory was called with appropriate config
            mock_factory.assert_called_once()
            
            service.close()


# Run stateful tests
TestEmbeddingServiceStateful = EmbeddingServiceStateMachine.TestCase


if __name__ == "__main__":
    pytest.main([__file__, "-v"])