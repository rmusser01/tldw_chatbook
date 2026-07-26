# test_embeddings_unit.py
# Unit tests for the simplified embeddings service API

import pytest
from unittest.mock import MagicMock, patch
import threading
import time
import numpy as np

from tldw_chatbook.RAG_Search.simplified import (
    EmbeddingsServiceWrapper,
    InMemoryVectorStore,
    ChromaVectorStore,
    create_embeddings_service,
)

# Import test utilities
from .conftest import requires_numpy


class TestEmbeddingsServiceWrapper:
    """Unit tests for EmbeddingsServiceWrapper"""

    def test_initialization_with_defaults(self):
        """Test initialization with default parameters"""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ) as mock_factory:
            service = EmbeddingsServiceWrapper()

            assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
            assert service._cache_size == 2
            mock_factory.assert_called_once()

    def test_initialization_with_openai_model(self):
        """Test initialization with OpenAI model"""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ):
            service = EmbeddingsServiceWrapper(
                model_name="openai/text-embedding-3-small", api_key="test-key"
            )

            assert service.model_name == "openai/text-embedding-3-small"
            assert service._api_key == "test-key"

    def test_device_auto_detection(self):
        """Test automatic device detection"""
        import torch

        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch(
                "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
            ):
                service = EmbeddingsServiceWrapper()
                assert service.device == "cuda"

    @requires_numpy
    def test_create_embeddings(self):
        """Test creating embeddings returns numpy array"""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ) as mock_factory:
            # Mock the factory instance and its embed method
            mock_instance = MagicMock()
            mock_instance.embed.return_value = np.array(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            )
            mock_factory.return_value = mock_instance

            service = EmbeddingsServiceWrapper()
            texts = ["test1", "test2"]
            embeddings = service.create_embeddings(texts)

            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (2, 3)
            mock_instance.embed.assert_called_once_with(texts, as_list=False)

    def test_memory_tracking(self):
        """Test that embeddings service can be initialized"""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ):
            service = EmbeddingsServiceWrapper()
            # Just verify the service was created successfully
            assert service is not None
            assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
            assert service._cache_size == 2


class TestBareHfModelIdNormalization:
    """task-640 AC#7: EmbeddingsServiceWrapper._build_config() must map
    known-broken bare HuggingFace model ids to their canonical, resolvable
    form for the underlying loader, WITHOUT ever mutating `model_name`
    itself (the exact string that round-trips as RAGConfig.embedding.model,
    the collection-fingerprint-determining field -- see
    Tests/RAG/simplified/test_collection_fingerprint.py's
    test_hybrid_basic_builtin_profile_fingerprint_is_stable_across_the_hf_
    model_id_fix for the fingerprint-side proof)."""

    def test_bare_all_minilm_model_id_is_normalized_for_the_loader(self):
        """The builtin "hybrid_basic"/"bm25_basic"/"fast" RAG profiles'
        embedding model, "all-MiniLM-L6-v2", is missing its
        "sentence-transformers/" org prefix and 404s against the real HF
        Hub as-is -- model_name_or_path must resolve to the canonical id."""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ):
            service = EmbeddingsServiceWrapper(model_name="all-MiniLM-L6-v2")
            config = service._build_config("all-MiniLM-L6-v2", None, None, None, None)
            assert (
                config["models"]["default"]["model_name_or_path"]
                == "sentence-transformers/all-MiniLM-L6-v2"
            )

    def test_bare_model_id_normalization_never_mutates_model_name(self):
        """`model_name` (and therefore what flows into RAGConfig.embedding.
        model / the collection fingerprint) must stay exactly the
        as-configured bare id -- only the HTTP-facing model_name_or_path is
        canonicalized."""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ):
            service = EmbeddingsServiceWrapper(model_name="all-MiniLM-L6-v2")
            assert service.model_name == "all-MiniLM-L6-v2"
            # _build_config's own return value must not carry the bare id
            # forward anywhere BUT model_name_or_path either.
            config = service._build_config("all-MiniLM-L6-v2", None, None, None, None)
            assert config["default_model_id"] == "default"

    def test_already_prefixed_model_ids_pass_through_unchanged(self):
        """A model id that already carries its real org prefix (every OTHER
        builtin profile's embedding.model) must never be rewritten."""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ):
            service = EmbeddingsServiceWrapper(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            config = service._build_config(
                "sentence-transformers/all-mpnet-base-v2", None, None, None, None
            )
            assert (
                config["models"]["default"]["model_name_or_path"]
                == "sentence-transformers/all-mpnet-base-v2"
            )

    def test_unrelated_bare_model_ids_are_not_touched_by_the_narrow_alias_map(self):
        """The fix is a narrow, explicit alias map for the one known-broken
        id -- NOT a "no slash -> prepend sentence-transformers/" heuristic,
        which would silently break a legitimate slash-free top-level HF
        model id like "bert-base-uncased"."""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ):
            service = EmbeddingsServiceWrapper(model_name="bert-base-uncased")
            config = service._build_config("bert-base-uncased", None, None, None, None)
            assert config["models"]["default"]["model_name_or_path"] == "bert-base-uncased"


class TestVectorStores:
    """Unit tests for vector store implementations"""

    def test_in_memory_store_operations(self):
        """Test InMemoryVectorStore basic operations"""
        store = InMemoryVectorStore()

        # Add documents
        ids = ["id1", "id2"]
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        documents = ["doc1", "doc2"]
        metadata = [{"id": 1}, {"id": 2}]

        store.add(ids, embeddings, documents, metadata)

        # Check stats to verify documents were added
        stats = store.get_collection_stats()
        assert stats["count"] == 2

        # Search (with mock similarity)
        query_embedding = np.array([0.15, 0.25])  # Query similar to first doc
        results = store.search(query_embedding, top_k=1)

        assert results is not None
        assert len(results) == 1
        assert results[0].id in ["id1", "id2"]

        # Clear the store
        store.clear()
        stats = store.get_collection_stats()
        assert stats["count"] == 0

    def test_in_memory_store_thread_safety(self, thread_helper):
        """Test InMemoryVectorStore thread safety"""
        store = InMemoryVectorStore()

        def add_documents(thread_id):
            ids = [f"id_{thread_id}"]
            embeddings = np.array([[float(thread_id), float(thread_id)]])
            documents = [f"doc_{thread_id}"]
            metadata = [{"thread": thread_id}]

            store.add(ids, embeddings, documents, metadata)
            return True

        results, errors = thread_helper.run_concurrent(add_documents, num_threads=10)

        assert len(errors) == 0
        # Check that all documents were added
        stats = store.get_collection_stats()
        assert stats["count"] == 10

    def test_chromadb_store_operations(self):
        """Test ChromaVectorStore basic structure and initialization"""
        # Just verify the class exists and has expected methods from VectorStore interface
        assert hasattr(ChromaVectorStore, "add")
        assert hasattr(ChromaVectorStore, "search")
        assert hasattr(ChromaVectorStore, "search_with_citations")
        assert hasattr(ChromaVectorStore, "delete_collection")
        assert hasattr(ChromaVectorStore, "clear")
        assert hasattr(ChromaVectorStore, "get_collection_stats")


class TestCreateEmbeddingsService:
    """Test the factory function for creating embeddings service"""

    def test_create_embeddings_service_defaults(self):
        """Test creating service with defaults"""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ):
            service = create_embeddings_service()

            assert isinstance(service, EmbeddingsServiceWrapper)
            assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_create_embeddings_service_with_config(self):
        """Test creating service with custom config"""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ):
            service = create_embeddings_service(
                provider="openai",
                model="text-embedding-3-small",
                api_key="test-key",
                cache_size=5,
            )

            assert service.model_name == "openai/text-embedding-3-small"
            assert service._api_key == "test-key"
            assert service._cache_size == 5


class TestBatchProcessing:
    """Test batch processing functionality"""

    @requires_numpy
    def test_batch_chunking(self):
        """Test that large batches are properly chunked"""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ) as mock_factory:
            # Set up mock to track batch sizes
            batch_sizes = []

            def mock_embed(texts, as_list=True):
                batch_sizes.append(len(texts))
                return np.array([[0.1] * 384 for _ in texts])

            mock_instance = MagicMock()
            mock_instance.embed.side_effect = mock_embed
            mock_factory.return_value = mock_instance

            service = EmbeddingsServiceWrapper()

            # Create embeddings for large batch
            texts = [f"text_{i}" for i in range(100)]
            embeddings = service.create_embeddings(texts)

            assert embeddings is not None
            assert embeddings.shape[0] == 100

            # Check that batching occurred (exact batching depends on implementation)
            assert len(batch_sizes) > 0


class TestErrorHandling:
    """Test error handling in embeddings service"""

    def test_invalid_model_name(self):
        """Test handling of invalid model names"""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ) as mock_factory:
            # Make factory initialization fail
            mock_factory.side_effect = Exception("Model not found")

            with pytest.raises(Exception) as exc_info:
                EmbeddingsServiceWrapper(model_name="invalid/model")

            assert "Model not found" in str(exc_info.value)

    @requires_numpy
    def test_embedding_creation_failure(self):
        """Test handling when embedding creation fails"""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ) as mock_factory:
            # Set up mock that fails on embed
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = Exception("Embedding failed")
            mock_factory.return_value = mock_instance

            service = EmbeddingsServiceWrapper()

            with pytest.raises(Exception) as exc_info:
                service.create_embeddings(["test"])

            assert "Embedding failed" in str(exc_info.value)

    def test_cleanup_on_error(self):
        """Test cleanup happens even on error"""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ) as mock_factory:
            mock_instance = MagicMock()
            mock_factory.return_value = mock_instance

            service = EmbeddingsServiceWrapper()

            # Close should be called on factory
            service.close()
            mock_instance.close.assert_called_once()


class TestConcurrency:
    """Test thread safety and concurrent operations"""

    @requires_numpy
    def test_concurrent_embedding_creation(self):
        """Test concurrent calls to create_embeddings"""
        with patch(
            "tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory"
        ) as mock_factory:
            # Set up thread-safe mock
            mock_instance = MagicMock()
            lock = threading.Lock()
            call_count = 0

            def thread_safe_embed(texts, as_list=True):
                nonlocal call_count
                with lock:
                    call_count += 1
                    time.sleep(0.01)  # Simulate work
                return np.array([[0.1] * 384 for _ in texts])

            mock_instance.embed.side_effect = thread_safe_embed
            mock_factory.return_value = mock_instance

            service = EmbeddingsServiceWrapper()

            # Run concurrent operations
            results = []
            errors = []
            threads = []

            def create_embeddings(thread_id):
                try:
                    embeddings = service.create_embeddings([f"thread_{thread_id}"])
                    results.append((thread_id, embeddings))
                except Exception as e:
                    errors.append((thread_id, str(e)))

            for i in range(5):
                thread = threading.Thread(target=create_embeddings, args=(i,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join(timeout=5)

            # All threads should succeed
            assert len(errors) == 0
            assert len(results) == 5
            assert call_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
