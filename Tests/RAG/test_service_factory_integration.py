# test_service_factory_integration.py
# Integration tests for RAG service factory using real services

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, Any

from tldw_chatbook.RAG_Search.Services.service_factory import RAGServiceFactory
from tldw_chatbook.RAG_Search.Services.embeddings_service import EmbeddingsService
from tldw_chatbook.RAG_Search.Services.chunking_service import ChunkingService
from tldw_chatbook.RAG_Search.Services.indexing_service import IndexingService
from tldw_chatbook.RAG_Search.Services.cache_service import CacheService
from tldw_chatbook.RAG_Search.Services.memory_management_service import MemoryManagementService
from tldw_chatbook.RAG_Search.Services.rag_service import RAGService
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

# Test marker for integration tests
pytestmark = pytest.mark.integration

# Skip tests if required dependencies are not available
requires_embeddings = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False),
    reason="embeddings dependencies not installed"
)

#######################################################################################################################
#
# Fixtures

@pytest.fixture
def temp_dirs():
    """Create temporary directories for all services"""
    base_dir = tempfile.mkdtemp()
    dirs = {
        'embeddings': Path(base_dir) / 'embeddings',
        'cache': Path(base_dir) / 'cache',
        'db': Path(base_dir) / 'db',
        'data': Path(base_dir) / 'data'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    yield dirs
    
    shutil.rmtree(base_dir)


@pytest.fixture
def real_media_db(temp_dirs):
    """Create a real MediaDatabase instance"""
    db_path = temp_dirs['db'] / 'media.db'
    db = MediaDatabase(str(db_path))
    yield db
    db.close()


@pytest.fixture
def real_chachanotes_db(temp_dirs):
    """Create a real CharactersRAGDB instance"""
    db_path = temp_dirs['db'] / 'chachanotes.db'
    db = CharactersRAGDB(str(db_path), 'test_client')
    yield db
    db.close()


@pytest.fixture
def real_config(temp_dirs):
    """Create a real configuration for testing"""
    # Create a minimal config file
    config_content = f"""
[rag]
batch_size = 16
num_workers = 2
use_gpu = false

[rag.retriever]
top_k = 5
similarity_threshold = 0.7

[rag.processor]
chunk_size = 512
chunk_overlap = 50

[rag.generator]
max_length = 1024
temperature = 0.7

[rag.chroma]
persist_directory = "{temp_dirs['embeddings']}"

[rag.cache]
cache_dir = "{temp_dirs['cache']}"
max_size_mb = 100

[memory]
max_total_size_mb = 500.0
enable_automatic_cleanup = true
cleanup_threshold = 0.8
"""
    
    config_path = temp_dirs['data'] / 'config.toml'
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Set environment variable to use this config
    old_config = os.environ.get('TLDW_CONFIG_PATH')
    os.environ['TLDW_CONFIG_PATH'] = str(config_path)
    
    yield config_path
    
    # Restore old config
    if old_config:
        os.environ['TLDW_CONFIG_PATH'] = old_config
    else:
        os.environ.pop('TLDW_CONFIG_PATH', None)


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            'text': "Artificial intelligence is transforming technology.",
            'metadata': {'source': 'tech_article', 'topic': 'AI'}
        },
        {
            'text': "Machine learning algorithms can learn from data.",
            'metadata': {'source': 'ml_guide', 'topic': 'ML'}
        },
        {
            'text': "Natural language processing enables computers to understand text.",
            'metadata': {'source': 'nlp_book', 'topic': 'NLP'}
        }
    ]


#######################################################################################################################
#
# Test Classes

class TestRealServiceFactory:
    """Test RAGServiceFactory with real service instances"""
    
    def test_create_real_embeddings_service(self, temp_dirs, real_config):
        """Test creating a real embeddings service"""
        factory = RAGServiceFactory()
        
        service = factory.create_embeddings_service(temp_dirs['embeddings'])
        
        assert service is not None
        assert isinstance(service, EmbeddingsService)
        
        # Verify service is functional
        if DEPENDENCIES_AVAILABLE.get('sentence_transformers', False):
            # Initialize with a small model
            success = service.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
            assert success
            
            # Test creating embeddings
            embeddings = service.create_embeddings(["Test text"])
            assert embeddings is not None
            assert len(embeddings) == 1
    
    def test_create_real_chunking_service(self, real_config):
        """Test creating a real chunking service"""
        factory = RAGServiceFactory()
        
        service = factory.create_chunking_service()
        
        assert service is not None
        assert isinstance(service, ChunkingService)
        
        # Test chunking functionality
        long_text = " ".join(["This is a test sentence."] * 100)
        chunks = service.chunk_text(long_text)
        
        assert len(chunks) > 1  # Should create multiple chunks
        assert all(len(chunk) <= service.chunk_size for chunk in chunks)
    
    @requires_embeddings
    def test_create_real_indexing_service(self, temp_dirs, real_config, real_media_db, real_chachanotes_db):
        """Test creating a real indexing service with dependencies"""
        factory = RAGServiceFactory()
        
        # Create dependencies first
        embeddings_service = factory.create_embeddings_service(temp_dirs['embeddings'])
        chunking_service = factory.create_chunking_service()
        
        # Create indexing service
        service = factory.create_indexing_service(
            embeddings_service=embeddings_service,
            chunking_service=chunking_service,
            media_db=real_media_db,
            chachanotes_db=real_chachanotes_db
        )
        
        assert service is not None
        assert isinstance(service, IndexingService)
        assert service.embeddings_service is embeddings_service
        assert service.chunking_service is chunking_service
        assert service.media_db is real_media_db
        assert service.chachanotes_db is real_chachanotes_db
    
    def test_create_real_cache_service(self, temp_dirs, real_config):
        """Test creating a real cache service"""
        factory = RAGServiceFactory()
        
        service = factory.create_cache_service()
        
        assert service is not None
        assert isinstance(service, CacheService)
        
        # Test cache functionality
        key = "test_key"
        value = {"data": "test_value"}
        
        service.set(key, value)
        retrieved = service.get(key)
        
        assert retrieved == value
    
    def test_create_real_memory_management_service(self, temp_dirs, real_config):
        """Test creating a real memory management service"""
        factory = RAGServiceFactory()
        
        # Create embeddings service first
        embeddings_service = factory.create_embeddings_service(temp_dirs['embeddings'])
        
        service = factory.create_memory_management_service(embeddings_service)
        
        assert service is not None
        assert isinstance(service, MemoryManagementService)
        assert service.embeddings_service is embeddings_service
        
        # Test memory tracking
        initial_usage = service.get_memory_usage()
        assert isinstance(initial_usage, dict)
        assert 'total_mb' in initial_usage
    
    @requires_embeddings
    def test_create_complete_real_rag_services(self, temp_dirs, real_config, real_media_db, real_chachanotes_db):
        """Test creating a complete set of real RAG services"""
        factory = RAGServiceFactory()
        
        services = factory.create_complete_rag_services(
            embeddings_dir=temp_dirs['embeddings'],
            media_db=real_media_db,
            chachanotes_db=real_chachanotes_db
        )
        
        # Verify all services were created
        assert 'embeddings' in services
        assert 'chunking' in services
        assert 'indexing' in services
        assert 'cache' in services
        assert 'rag' in services
        assert 'memory_manager' in services
        
        # Verify all services are real instances
        assert isinstance(services['embeddings'], EmbeddingsService)
        assert isinstance(services['chunking'], ChunkingService)
        assert isinstance(services['indexing'], IndexingService)
        assert isinstance(services['cache'], CacheService)
        assert isinstance(services['rag'], RAGService)
        assert isinstance(services['memory_manager'], MemoryManagementService)
        
        # Verify services are properly connected
        assert services['indexing'].embeddings_service is services['embeddings']
        assert services['indexing'].chunking_service is services['chunking']
        assert services['memory_manager'].embeddings_service is services['embeddings']
    
    @requires_embeddings
    def test_end_to_end_rag_workflow(self, temp_dirs, real_config, real_media_db, real_chachanotes_db, sample_documents):
        """Test complete RAG workflow with real services"""
        factory = RAGServiceFactory()
        
        # Create all services
        services = factory.create_complete_rag_services(
            embeddings_dir=temp_dirs['embeddings'],
            media_db=real_media_db,
            chachanotes_db=real_chachanotes_db
        )
        
        # Initialize embedding model
        services['embeddings'].initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        
        # Index documents
        indexing_service = services['indexing']
        for doc in sample_documents:
            # Add to media database first
            media_id = real_media_db.insert_media_item(
                title=f"Document from {doc['metadata']['source']}",
                content=doc['text'],
                media_type='document',
                author='test',
                metadata=doc['metadata']
            )
            
            # Index the document
            success = indexing_service.index_media_item(media_id)
            assert success
        
        # Search using RAG service
        rag_service = services['rag']
        query = "What is machine learning?"
        
        results = rag_service.search(
            query=query,
            n_results=3,
            collection_name='media_items'
        )
        
        assert results is not None
        assert len(results) > 0
        
        # Should find ML-related document
        found_ml = any('machine learning' in str(r).lower() for r in results)
        assert found_ml
    
    def test_service_factory_without_embeddings(self, temp_dirs, real_config, real_media_db, real_chachanotes_db):
        """Test service factory when embeddings dependencies are not available"""
        factory = RAGServiceFactory()
        
        # Temporarily mock dependencies as unavailable
        from unittest.mock import patch
        with patch.dict('tldw_chatbook.Utils.optional_deps.DEPENDENCIES_AVAILABLE', {'embeddings_rag': False}):
            services = factory.create_complete_rag_services(
                embeddings_dir=temp_dirs['embeddings'],
                media_db=real_media_db,
                chachanotes_db=real_chachanotes_db
            )
        
        # Some services should still be created
        assert 'chunking' in services
        assert 'cache' in services
        assert isinstance(services['chunking'], ChunkingService)
        assert isinstance(services['cache'], CacheService)
        
        # But embeddings-dependent services should be None
        assert services.get('embeddings') is None
        assert services.get('indexing') is None
        assert services.get('memory_manager') is None
    
    @requires_embeddings
    def test_service_reconfiguration(self, temp_dirs, real_config):
        """Test reconfiguring services after creation"""
        factory = RAGServiceFactory()
        
        # Create embeddings service
        service = factory.create_embeddings_service(temp_dirs['embeddings'])
        
        # Initialize with one model
        service.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        embeddings1 = service.create_embeddings(["Test text"])
        dim1 = len(embeddings1[0])
        
        # Reconfigure with different model (if available)
        try:
            service.initialize_embedding_model("sentence-transformers/all-mpnet-base-v2")
            embeddings2 = service.create_embeddings(["Test text"])
            dim2 = len(embeddings2[0])
            
            # Different models should have different dimensions
            assert dim1 != dim2  # MiniLM: 384, MPNet: 768
        except Exception:
            # If second model not available, just verify first worked
            assert dim1 == 384  # MiniLM dimension
    
    @requires_embeddings
    def test_service_error_recovery(self, temp_dirs, real_config, real_media_db, real_chachanotes_db):
        """Test that services can recover from errors"""
        factory = RAGServiceFactory()
        
        services = factory.create_complete_rag_services(
            embeddings_dir=temp_dirs['embeddings'],
            media_db=real_media_db,
            chachanotes_db=real_chachanotes_db
        )
        
        # Try to index without initializing embedding model
        indexing_service = services['indexing']
        
        # This should fail gracefully
        success = indexing_service.index_media_item(999999)  # Non-existent ID
        assert not success
        
        # Service should still be functional after error
        services['embeddings'].initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        
        # Add a real media item
        media_id = real_media_db.insert_media_item(
            title="Recovery Test",
            content="Testing error recovery",
            media_type='document'
        )
        
        # This should now work
        success = indexing_service.index_media_item(media_id)
        assert success