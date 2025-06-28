# conftest.py for RAG tests
# Provides common fixtures and markers for RAG-related tests

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# Reset and re-initialize dependency checks before importing anything else
from tldw_chatbook.Utils.optional_deps import reset_dependency_checks, initialize_dependency_checks, DEPENDENCIES_AVAILABLE
reset_dependency_checks()
initialize_dependency_checks()

# Now check the availability
EMBEDDINGS_AVAILABLE = DEPENDENCIES_AVAILABLE.get('embeddings_rag', False)
CHROMADB_AVAILABLE = DEPENDENCIES_AVAILABLE.get('chromadb', False)

# If still not detected but we know they're installed, force enable
if not EMBEDDINGS_AVAILABLE:
    try:
        import torch
        import transformers
        import numpy
        import chromadb
        import sentence_transformers
        EMBEDDINGS_AVAILABLE = True
        CHROMADB_AVAILABLE = True
        # Update the global state
        DEPENDENCIES_AVAILABLE['embeddings_rag'] = True
        DEPENDENCIES_AVAILABLE['chromadb'] = True
        DEPENDENCIES_AVAILABLE['torch'] = True
        DEPENDENCIES_AVAILABLE['transformers'] = True
        DEPENDENCIES_AVAILABLE['numpy'] = True
        DEPENDENCIES_AVAILABLE['sentence_transformers'] = True
    except ImportError:
        pass

# Define custom markers
pytest.mark.requires_embeddings = pytest.mark.skipif(
    not EMBEDDINGS_AVAILABLE,
    reason="Embeddings dependencies (sentence-transformers) not installed. Install with: pip install -e '.[embeddings_rag]'"
)

pytest.mark.requires_chromadb = pytest.mark.skipif(
    not CHROMADB_AVAILABLE,
    reason="ChromaDB not installed. Install with: pip install -e '.[embeddings_rag]'"
)

pytest.mark.requires_rag_deps = pytest.mark.skipif(
    not (EMBEDDINGS_AVAILABLE and CHROMADB_AVAILABLE),
    reason="RAG dependencies not installed. Install with: pip install -e '.[embeddings_rag]'"
)

# Register markers with pytest
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "requires_embeddings: mark test as requiring embeddings dependencies"
    )
    config.addinivalue_line(
        "markers", "requires_chromadb: mark test as requiring ChromaDB"
    )
    config.addinivalue_line(
        "markers", "requires_rag_deps: mark test as requiring all RAG dependencies"
    )


# Mock embedding provider for tests
class MockEmbeddingProvider:
    """Mock embedding provider that returns consistent 2D embeddings"""
    def __init__(self, dimension=2, model_name="mock-model"):
        self.dimension = dimension
        self.model_name = model_name
        self._call_count = 0
        
    def create_embeddings(self, texts):
        """Create mock embeddings"""
        self._call_count += 1
        embeddings = []
        for i, text in enumerate(texts):
            # Generate deterministic embeddings based on text
            hash_val = hash(text) % 1000 / 1000.0
            if self.dimension == 2:
                embeddings.append([hash_val, hash_val * 0.5])
            else:
                # For higher dimensions, create a pattern
                emb = [hash_val + (j * 0.001) for j in range(self.dimension)]
                embeddings.append(emb)
        return embeddings
    
    def get_dimension(self):
        return self.dimension
        
    @property
    def call_count(self):
        return self._call_count
    
    def cleanup(self):
        """Cleanup method for compatibility"""
        pass


# Global mock for sentence transformers
@pytest.fixture(autouse=True, scope="session")
def mock_sentence_transformers():
    """Auto-mock sentence transformers for all tests"""
    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        mock_model = MagicMock()
        
        # Mock the encode method to return 2D embeddings
        def mock_encode(texts, *args, **kwargs):
            embeddings = []
            for text in texts:
                hash_val = hash(text) % 1000 / 1000.0
                embeddings.append([hash_val, hash_val * 0.5])
            return np.array(embeddings)
        
        mock_model.encode = MagicMock(side_effect=mock_encode)
        mock_model.get_sentence_embedding_dimension.return_value = 2
        mock_model.max_seq_length = 512
        
        mock_st.return_value = mock_model
        yield mock_st


# Prevent auto-initialization of providers
@pytest.fixture(autouse=True)
def prevent_provider_init():
    """Prevent automatic provider initialization in tests"""
    from tldw_chatbook.RAG_Search.Services import embeddings_service
    
    # Save original methods
    orig_init_from_config = embeddings_service.EmbeddingsService.initialize_from_config
    orig_init_model = embeddings_service.EmbeddingsService.initialize_embedding_model
    
    # Mock methods to prevent initialization
    def mock_init_from_config(self, config):
        return True  # Pretend it worked
    
    def mock_init_model(self, model_name=None):
        # Create a mock provider instead of real one
        mock_provider = MockEmbeddingProvider(dimension=2)
        provider_id = "test_provider"
        self.providers[provider_id] = mock_provider
        self.current_provider_id = provider_id
        self.embedding_model = mock_provider
        
        # Store original create_embeddings for use in wrapper
        if not hasattr(self, '_original_create_embeddings'):
            self._original_create_embeddings = self.create_embeddings
        
        # Wrap create_embeddings to use our mock provider
        def mock_create_embeddings(texts, provider_id=None):
            # If we have a cache service, try to use it
            if hasattr(self, 'cache_service') and self.cache_service:
                try:
                    cached, uncached = self.cache_service.get_embeddings_batch(texts)
                    
                    # Generate embeddings for uncached texts
                    if uncached:
                        new_embeddings = mock_provider.create_embeddings(uncached)
                        # Cache the new embeddings
                        cache_dict = {text: emb for text, emb in zip(uncached, new_embeddings)}
                        self.cache_service.cache_embeddings_batch(cache_dict)
                    
                    # Combine cached and new embeddings in original order
                    result = []
                    for text in texts:
                        if text in cached:
                            result.append(cached[text])
                        else:
                            idx = uncached.index(text)
                            result.append(new_embeddings[idx])
                    return result
                except:
                    # If cache fails, just generate all
                    pass
            
            return mock_provider.create_embeddings(texts)
        
        self.create_embeddings = mock_create_embeddings
        
        return True
    
    # Patch the methods
    embeddings_service.EmbeddingsService.initialize_from_config = mock_init_from_config
    embeddings_service.EmbeddingsService.initialize_embedding_model = mock_init_model
    
    yield
    
    # Restore original methods
    embeddings_service.EmbeddingsService.initialize_from_config = orig_init_from_config
    embeddings_service.EmbeddingsService.initialize_embedding_model = orig_init_model


@pytest.fixture
def mock_chromadb_client():
    """Mock ChromaDB client for tests"""
    with patch('chromadb.PersistentClient') as mock_client_class:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        # Mock collection methods
        mock_collection.add.return_value = None
        mock_collection.query.return_value = {
            'ids': [['id1', 'id2']],
            'embeddings': None,
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'source': 'test'}, {'source': 'test'}]],
            'distances': [[0.1, 0.2]]
        }
        mock_collection.get.return_value = {
            'ids': ['id1'],
            'embeddings': None,
            'documents': ['doc1'],
            'metadatas': [{'source': 'test'}]
        }
        mock_collection.delete.return_value = None
        mock_collection.update.return_value = None
        
        # Mock client methods
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.return_value = mock_collection
        mock_client.list_collections.return_value = []
        mock_client.delete_collection.return_value = None
        
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_embeddings_service_factory():
    """Factory to create properly mocked embeddings services"""
    created_services = []
    
    def _create_service(persist_directory=None, dimension=2, embedding_config=None):
        # Import here to avoid circular imports
        from tldw_chatbook.RAG_Search.Services.embeddings_service import EmbeddingsService
        
        # Patch the provider initialization
        with patch.object(EmbeddingsService, 'initialize_embedding_model') as mock_init:
            with patch.object(EmbeddingsService, 'add_provider') as mock_add:
                service = EmbeddingsService(
                    persist_directory=persist_directory,
                    embedding_config=embedding_config or {}
                )
                
                # Create a mock provider
                mock_provider = MockEmbeddingProvider(dimension=dimension)
                
                # Manually set up the service with our mock
                service.providers = {'mock': mock_provider}
                service.current_provider_id = 'mock'
                service.embedding_model = mock_provider
                
                # Override create_embeddings to use our mock
                original_create = service.create_embeddings
                def mock_create_embeddings(texts, provider_id=None):
                    provider = service.providers.get(provider_id or service.current_provider_id)
                    if provider:
                        return provider.create_embeddings(texts)
                    return None
                
                service.create_embeddings = mock_create_embeddings
                
                created_services.append(service)
                return service
    
    yield _create_service
    
    # Cleanup
    for service in created_services:
        if hasattr(service, 'close'):
            service.close()