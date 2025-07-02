# test_embeddings_properties.py
# Property-based tests for embeddings service using Hypothesis

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition, invariant, Bundle
import threading
import time
from typing import List, Dict, Any, Set
import hashlib

from tldw_chatbook.RAG_Search.simplified import (
    EmbeddingsService,
    InMemoryVectorStore,
    create_embeddings_service
)
# Note: EmbeddingFactoryCompat is no longer available in simplified API

# Import test utilities from conftest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import MockEmbeddingProvider, requires_numpy


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
        provider = MockEmbeddingProvider(dimension=384)
        service = EmbeddingsService(vector_store=InMemoryVectorStore())
        service.add_provider("test", provider)
        
        # Create embeddings twice
        embeddings1 = service.create_embeddings(texts)
        embeddings2 = service.create_embeddings(texts)
        
        # Should be identical
        assert embeddings1 is not None
        assert embeddings2 is not None
        assert len(embeddings1) == len(embeddings2)
        
        for e1, e2 in zip(embeddings1, embeddings2):
            assert e1 == e2
    
    @given(
        texts=text_batch(min_size=1, max_size=5),
        dimension=embedding_dimension()
    )
    def test_embedding_dimension_consistency(self, texts, dimension):
        """All embeddings should have consistent dimension"""
        provider = MockEmbeddingProvider(dimension=dimension)
        service = EmbeddingsService(vector_store=InMemoryVectorStore())
        service.add_provider("test", provider)
        
        embeddings = service.create_embeddings(texts)
        
        assert embeddings is not None
        assert all(len(emb) == dimension for emb in embeddings)
        assert provider.get_dimension() == dimension
    
    @given(texts=text_batch())
    def test_embedding_count_matches_input(self, texts):
        """Number of embeddings should match number of input texts"""
        provider = MockEmbeddingProvider()
        service = EmbeddingsService(vector_store=InMemoryVectorStore())
        service.add_provider("test", provider)
        
        embeddings = service.create_embeddings(texts)
        
        assert embeddings is not None
        assert len(embeddings) == len(texts)
    
    @given(
        texts=st.lists(valid_text(), min_size=10, max_size=50),
        batch_size=st.integers(min_value=1, max_value=10)
    )
    def test_batch_processing_consistency(self, texts, batch_size):
        """Batch processing should produce same results as single processing"""
        provider = MockEmbeddingProvider()
        service = EmbeddingsService(vector_store=InMemoryVectorStore())
        service.add_provider("test", provider)
        
        # Process without batching
        service.configure_performance(enable_parallel=False)
        embeddings_single = service.create_embeddings(texts)
        
        # Process with batching
        service.configure_performance(
            enable_parallel=True,
            batch_size=batch_size
        )
        embeddings_batch = service.create_embeddings(texts)
        
        # Results should be identical
        assert embeddings_single == embeddings_batch
    
    @given(
        texts=text_batch(),
        num_providers=st.integers(min_value=2, max_value=5)
    )
    def test_provider_isolation(self, texts, num_providers):
        """Different providers should not interfere with each other"""
        service = EmbeddingsService(vector_store=InMemoryVectorStore())
        
        # Add multiple providers with different dimensions
        dimensions = [384, 512, 768, 1024, 1536]
        for i in range(num_providers):
            provider = MockEmbeddingProvider(dimension=dimensions[i % len(dimensions)])
            service.add_provider(f"provider_{i}", provider)
        
        # Create embeddings with each provider
        results = {}
        for i in range(num_providers):
            service.set_provider(f"provider_{i}")
            embeddings = service.create_embeddings(texts)
            results[f"provider_{i}"] = embeddings
        
        # Each provider should produce different dimensional embeddings
        for i in range(num_providers):
            assert all(len(emb) == dimensions[i % len(dimensions)] 
                      for emb in results[f"provider_{i}"])


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
        provider = MockEmbeddingProvider()
        service = EmbeddingsService(vector_store=InMemoryVectorStore())
        service.add_provider("test", provider)
        
        # Add documents
        embeddings = service.create_embeddings(texts)
        doc_ids = [f"doc_{i}" for i in range(len(texts))]
        
        service.add_documents_to_collection(
            "test_collection",
            texts,
            embeddings,
            [{"id": i} for i in range(len(texts))],
            doc_ids
        )
        
        # Search
        query_embeddings = service.create_embeddings(["query"])
        results = service.search_collection(
            "test_collection",
            query_embeddings,
            n_results=n_results
        )
        
        assert results is not None
        assert len(results["ids"][0]) <= min(n_results, len(texts))
    
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
        collection = store.collections["test_collection"]
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
        provider = MockEmbeddingProvider()
        service = EmbeddingsService(vector_store=InMemoryVectorStore())
        service.add_provider("test", provider)
        
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
            assert len(embeddings) == texts_per_thread
    
    @given(
        num_operations=st.integers(min_value=5, max_value=20),
        operation_types=st.lists(
            st.sampled_from(["add", "remove", "switch"]),
            min_size=5,
            max_size=20
        )
    )
    def test_concurrent_provider_management(self, num_operations, operation_types):
        """Concurrent provider management should be thread-safe"""
        service = EmbeddingsService(vector_store=InMemoryVectorStore())
        
        # Add initial providers
        for i in range(3):
            provider = MockEmbeddingProvider()
            service.add_provider(f"provider_{i}", provider)
        
        results = []
        errors = []
        threads = []
        
        def perform_operation(op_id, op_type):
            try:
                if op_type == "add":
                    provider = MockEmbeddingProvider()
                    service.add_provider(f"dynamic_{op_id}", provider)
                elif op_type == "switch" and service.providers:
                    import random
                    provider_id = random.choice(list(service.providers.keys()))
                    service.set_provider(provider_id)
                elif op_type == "remove" and len(service.providers) > 1:
                    # Note: The service doesn't have remove_provider, so we skip
                    pass
                
                # Try to create embeddings
                embeddings = service.create_embeddings(["test"])
                results.append((op_id, embeddings is not None))
            except Exception as e:
                errors.append((op_id, e))
        
        # Execute operations concurrently
        for i, op_type in enumerate(operation_types[:num_operations]):
            thread = threading.Thread(target=perform_operation, args=(i, op_type))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Operations should not cause crashes
        assert len(errors) == 0


class EmbeddingServiceStateMachine(RuleBasedStateMachine):
    """Stateful testing for EmbeddingsService"""
    
    def __init__(self):
        super().__init__()
        self.service = EmbeddingsService(vector_store=InMemoryVectorStore())
        self.providers = {}
        self.collections = set()
        self.documents = {}  # collection -> list of doc ids
    
    providers_bundle = Bundle('providers')
    collections_bundle = Bundle('collections')
    
    @rule(
        provider_id=st.text(min_size=1, max_size=20),
        dimension=embedding_dimension()
    )
    def add_provider(self, provider_id, dimension):
        """Add a new provider"""
        if provider_id not in self.providers:
            provider = MockEmbeddingProvider(dimension=dimension)
            self.service.add_provider(provider_id, provider)
            self.providers[provider_id] = provider
            return provider_id
    
    @rule(provider_id=providers_bundle)
    def switch_provider(self, provider_id):
        """Switch to a different provider"""
        success = self.service.set_provider(provider_id)
        assert success
        assert self.service.current_provider_id == provider_id
    
    @rule(
        texts=text_batch(),
        collection_name=st.text(min_size=1, max_size=20)
    )
    def create_and_store_embeddings(self, texts, collection_name):
        """Create embeddings and store in collection"""
        if not self.service.current_provider_id:
            # Need at least one provider
            self.add_provider("default", 384)
            self.service.set_provider("default")
        
        embeddings = self.service.create_embeddings(texts)
        assert embeddings is not None
        assert len(embeddings) == len(texts)
        
        # Store in collection
        doc_ids = [f"{collection_name}_doc_{i}_{hash(texts[i])}" for i in range(len(texts))]
        success = self.service.add_documents_to_collection(
            collection_name,
            texts,
            embeddings,
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
        if not self.service.current_provider_id:
            return  # Skip if no provider
        
        query_embeddings = self.service.create_embeddings(["search query"])
        results = self.service.search_collection(
            collection_name,
            query_embeddings,
            n_results=5
        )
        
        assert results is not None
        assert "ids" in results
    
    @rule(collection_name=collections_bundle)
    def delete_collection(self, collection_name):
        """Delete a collection"""
        success = self.service.delete_collection(collection_name)
        assert success
        
        self.collections.remove(collection_name)
        if collection_name in self.documents:
            del self.documents[collection_name]
    
    @invariant()
    def collections_consistency(self):
        """Listed collections should match tracked collections"""
        listed = set(self.service.list_collections())
        # Some collections might exist from other tests
        assert self.collections.issubset(listed)
    
    @invariant()
    def provider_consistency(self):
        """Current provider should be in providers list"""
        if self.service.current_provider_id:
            assert self.service.current_provider_id in self.service.providers


# TestCompatibilityProperties class removed as EmbeddingFactoryCompat is no longer available in simplified API
# The simplified API doesn't expose legacy compatibility features


# Run stateful tests
TestEmbeddingServiceStateful = EmbeddingServiceStateMachine.TestCase


if __name__ == "__main__":
    pytest.main([__file__, "-v"])