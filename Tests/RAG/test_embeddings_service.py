# test_embeddings_service.py
# Unit tests for the RAG embeddings service

import pytest
import tempfile
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock, call

from tldw_chatbook.RAG_Search.Services.embeddings_service import EmbeddingsService


@pytest.mark.requires_rag_deps
class TestEmbeddingsService:
    """Test the embeddings service"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for ChromaDB"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_cache_service(self):
        """Create a mock cache service"""
        mock = MagicMock()
        mock.get_embeddings_batch.return_value = ({}, ["text1", "text2", "text3"])
        return mock
    
    @pytest.fixture
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.CHROMADB_AVAILABLE', True)
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.EMBEDDINGS_AVAILABLE', True)
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.chromadb')
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.get_cache_service')
    def embeddings_service(self, mock_get_cache, mock_chromadb, mock_cache_service, temp_dir):
        """Create an embeddings service instance with mocked dependencies"""
        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Mock cache service
        mock_get_cache.return_value = mock_cache_service
        
        service = EmbeddingsService(temp_dir)
        service.client = mock_client
        return service
    
    def test_initialization(self, embeddings_service, temp_dir):
        """Test service initialization"""
        assert embeddings_service.persist_directory == temp_dir
        assert embeddings_service.client is not None
        assert embeddings_service.cache_service is not None
        assert embeddings_service.embedding_model is None
    
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.CHROMADB_AVAILABLE', False)
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.get_cache_service')
    def test_initialization_without_chromadb(self, mock_get_cache, mock_cache_service, temp_dir):
        """Test initialization when ChromaDB is not available"""
        mock_get_cache.return_value = mock_cache_service
        
        service = EmbeddingsService(temp_dir)
        assert service.client is None
        assert service.cache_service is not None
    
    def test_initialize_embedding_model(self, embeddings_service):
        """Test embedding model initialization"""
        mock_model = MagicMock()
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            result = embeddings_service.initialize_embedding_model()
        
            assert result is True
            assert embeddings_service.embedding_model == mock_model
    
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.EMBEDDINGS_AVAILABLE', False)
    def test_initialize_embedding_model_no_deps(self, embeddings_service):
        """Test embedding model initialization without dependencies"""
        result = embeddings_service.initialize_embedding_model()
        assert result is False
        assert embeddings_service.embedding_model is None
    
    def test_create_embeddings(self, embeddings_service, mock_cache_service):
        """Test creating embeddings for texts"""
        # Setup mock model
        mock_model = MagicMock()
        # Create a mock numpy array with tolist() method
        mock_array = MagicMock()
        mock_array.tolist.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_model.encode.return_value = mock_array
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            # Initialize model
            embeddings_service.initialize_embedding_model()
        
            # Create embeddings
            texts = ["text1", "text2", "text3"]
            embeddings = embeddings_service.create_embeddings(texts)
        
            assert embeddings is not None
            assert len(embeddings) == 3
            assert embeddings[0] == [0.1, 0.2]
            assert embeddings[1] == [0.3, 0.4]
            assert embeddings[2] == [0.5, 0.6]
        
            # Check that cache was used
            mock_cache_service.get_embeddings_batch.assert_called_once_with(texts)
            mock_cache_service.cache_embeddings_batch.assert_called_once()
    
    def test_create_embeddings_with_cache(self, embeddings_service, mock_cache_service):
        """Test creating embeddings with some cached"""
        # Setup cache to return some cached embeddings
        mock_cache_service.get_embeddings_batch.return_value = (
            {"text1": [0.1, 0.2], "text3": [0.5, 0.6]},
            ["text2"]
        )
        
        # Setup mock model
        mock_model = MagicMock()
        # Create a mock numpy array with tolist() method
        mock_array = MagicMock()
        mock_array.tolist.return_value = [[0.3, 0.4]]  # Only for text2
        mock_model.encode.return_value = mock_array
        embeddings_service.embedding_model = mock_model
        
        # Create embeddings
        texts = ["text1", "text2", "text3"]
        embeddings = embeddings_service.create_embeddings(texts)
        
        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2]  # From cache
        assert embeddings[1] == [0.3, 0.4]  # Generated
        assert embeddings[2] == [0.5, 0.6]  # From cache
        
        # Model should only be called for uncached text
        mock_model.encode.assert_called_once_with(["text2"])
    
    def test_create_embeddings_no_model(self, embeddings_service):
        """Test creating embeddings without initialized model"""
        embeddings_service.embedding_model = None
        
        # Mock initialization to fail
        with patch.object(embeddings_service, 'initialize_embedding_model', return_value=False):
            embeddings = embeddings_service.create_embeddings(["text"])
            assert embeddings is None
    
    def test_get_or_create_collection(self, embeddings_service):
        """Test getting or creating a collection"""
        mock_collection = MagicMock()
        embeddings_service.client.get_or_create_collection.return_value = mock_collection
        
        collection = embeddings_service.get_or_create_collection("test_collection")
        
        assert collection == mock_collection
        embeddings_service.client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            metadata={}
        )
    
    def test_get_or_create_collection_with_metadata(self, embeddings_service):
        """Test getting or creating a collection with metadata"""
        mock_collection = MagicMock()
        embeddings_service.client.get_or_create_collection.return_value = mock_collection
        
        metadata = {"description": "Test collection"}
        collection = embeddings_service.get_or_create_collection("test_collection", metadata)
        
        embeddings_service.client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            metadata=metadata
        )
    
    def test_get_or_create_collection_no_client(self, embeddings_service):
        """Test getting collection without ChromaDB client"""
        embeddings_service.client = None
        
        collection = embeddings_service.get_or_create_collection("test_collection")
        assert collection is None
    
    def test_add_documents_to_collection(self, embeddings_service):
        """Test adding documents to a collection"""
        mock_collection = MagicMock()
        
        with patch.object(embeddings_service, 'get_or_create_collection', return_value=mock_collection):
            documents = ["doc1", "doc2"]
            embeddings_list = [[0.1, 0.2], [0.3, 0.4]]
            metadatas = [{"id": 1}, {"id": 2}]
            ids = ["id1", "id2"]
            
            result = embeddings_service.add_documents_to_collection(
                "test_collection",
                documents,
                embeddings_list,
                metadatas,
                ids
            )
            
            assert result is True
            mock_collection.add.assert_called_once_with(
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=ids
            )
    
    def test_add_documents_to_collection_no_collection(self, embeddings_service):
        """Test adding documents when collection creation fails"""
        with patch.object(embeddings_service, 'get_or_create_collection', return_value=None):
            result = embeddings_service.add_documents_to_collection(
                "test_collection", [], [], [], []
            )
            assert result is False
    
    def test_search_collection(self, embeddings_service):
        """Test searching a collection"""
        mock_collection = MagicMock()
        mock_results = {
            'documents': [["doc1", "doc2"]],
            'metadatas': [[{"id": 1}, {"id": 2}]],
            'distances': [[0.1, 0.2]]
        }
        mock_collection.query.return_value = mock_results
        
        with patch.object(embeddings_service, 'get_or_create_collection', return_value=mock_collection):
            query_embeddings = [[0.5, 0.6]]
            results = embeddings_service.search_collection(
                "test_collection",
                query_embeddings,
                n_results=2
            )
            
            assert results == mock_results
            mock_collection.query.assert_called_once_with(
                query_embeddings=query_embeddings,
                n_results=2,
                where=None
            )
    
    def test_search_collection_with_filter(self, embeddings_service):
        """Test searching a collection with filters"""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {'documents': [[]]}
        
        with patch.object(embeddings_service, 'get_or_create_collection', return_value=mock_collection):
            where_filter = {"metadata_field": "value"}
            embeddings_service.search_collection(
                "test_collection",
                [[0.5, 0.6]],
                n_results=5,
                where=where_filter
            )
            
            mock_collection.query.assert_called_once_with(
                query_embeddings=[[0.5, 0.6]],
                n_results=5,
                where=where_filter
            )
    
    def test_delete_collection(self, embeddings_service):
        """Test deleting a collection"""
        embeddings_service.client.delete_collection.return_value = None
        
        result = embeddings_service.delete_collection("test_collection")
        
        assert result is True
        embeddings_service.client.delete_collection.assert_called_once_with(name="test_collection")
    
    def test_delete_collection_no_client(self, embeddings_service):
        """Test deleting collection without client"""
        embeddings_service.client = None
        
        result = embeddings_service.delete_collection("test_collection")
        assert result is False
    
    def test_list_collections(self, embeddings_service):
        """Test listing collections"""
        mock_collection1 = MagicMock()
        mock_collection1.name = "collection1"
        mock_collection2 = MagicMock()
        mock_collection2.name = "collection2"
        
        embeddings_service.client.list_collections.return_value = [mock_collection1, mock_collection2]
        
        collections = embeddings_service.list_collections()
        
        assert collections == ["collection1", "collection2"]
    
    def test_list_collections_no_client(self, embeddings_service):
        """Test listing collections without client"""
        embeddings_service.client = None
        
        collections = embeddings_service.list_collections()
        assert collections == []
    
    def test_get_collection_info(self, embeddings_service):
        """Test getting collection information"""
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.count.return_value = 100
        mock_collection.metadata = {"description": "Test"}
        
        with patch.object(embeddings_service, 'get_or_create_collection', return_value=mock_collection):
            info = embeddings_service.get_collection_info("test_collection")
            
            assert info['count'] == 100
            assert info['metadata'] == {"description": "Test"}
            assert info['name'] == "test_collection"
    
    def test_get_collection_info_no_collection(self, embeddings_service):
        """Test getting info for non-existent collection"""
        with patch.object(embeddings_service, 'get_or_create_collection', return_value=None):
            info = embeddings_service.get_collection_info("test_collection")
            assert info is None
    
    def test_clear_collection(self, embeddings_service):
        """Test clearing a collection"""
        result = embeddings_service.clear_collection("test_collection")
        
        assert result is True
        # Should delete and recreate
        assert embeddings_service.client.delete_collection.called
    
    def test_update_documents(self, embeddings_service):
        """Test updating documents in a collection"""
        mock_collection = MagicMock()
        
        with patch.object(embeddings_service, 'get_or_create_collection', return_value=mock_collection):
            documents = ["updated doc1"]
            embeddings_list = [[0.7, 0.8]]
            metadatas = [{"updated": True}]
            ids = ["id1"]
            
            result = embeddings_service.update_documents(
                "test_collection",
                documents,
                embeddings_list,
                metadatas,
                ids
            )
            
            assert result is True
            mock_collection.update.assert_called_once_with(
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=ids
            )
    
    def test_delete_documents(self, embeddings_service):
        """Test deleting documents from a collection"""
        mock_collection = MagicMock()
        
        with patch.object(embeddings_service, 'get_or_create_collection', return_value=mock_collection):
            ids = ["id1", "id2"]
            result = embeddings_service.delete_documents("test_collection", ids)
            
            assert result is True
            mock_collection.delete.assert_called_once_with(ids=ids)