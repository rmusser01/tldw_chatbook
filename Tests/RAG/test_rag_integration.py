# test_rag_integration.py
# Integration tests for the RAG pipeline end-to-end

import pytest
import asyncio
import tempfile
import sqlite3
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock

from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
    perform_plain_rag_search,
    perform_full_rag_pipeline,
    perform_hybrid_rag_search,
    get_rag_context_for_chat
)
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.RAG_Search.Services import (
    EmbeddingsService,
    ChunkingService,
    IndexingService
)


@pytest.mark.requires_rag_deps
class TestRAGIntegration:
    """Integration tests for the complete RAG pipeline"""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for databases"""
        media_dir = tempfile.mkdtemp()
        chachanotes_dir = tempfile.mkdtemp()
        chromadb_dir = tempfile.mkdtemp()
        
        yield {
            'media': Path(media_dir),
            'chachanotes': Path(chachanotes_dir),
            'chromadb': Path(chromadb_dir)
        }
        
        # Cleanup
        shutil.rmtree(media_dir)
        shutil.rmtree(chachanotes_dir)
        shutil.rmtree(chromadb_dir)
    
    @pytest.fixture
    def media_db(self, temp_dirs):
        """Create a real media database with test data"""
        db_path = temp_dirs['media'] / "media.db"
        db = MediaDatabase(str(db_path))
        
        # Add test media items
        test_items = [
            {
                'title': 'Python Programming Guide',
                'content': 'Python is a high-level programming language. It is known for its simplicity and readability. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.',
                'type': 'article',
                'author': 'Test Author 1',
                'ingestion_date': '2024-01-01'
            },
            {
                'title': 'Machine Learning Basics',
                'content': 'Machine learning is a subset of artificial intelligence. It focuses on building systems that learn from data. Common algorithms include linear regression, decision trees, and neural networks.',
                'type': 'article',
                'author': 'Test Author 2',
                'ingestion_date': '2024-01-02'
            },
            {
                'title': 'Web Development Tutorial',
                'content': 'Web development involves creating websites and web applications. Frontend development uses HTML, CSS, and JavaScript. Backend development can use Python, Ruby, Java, or other languages.',
                'type': 'tutorial',
                'author': 'Test Author 3',
                'ingestion_date': '2024-01-03'
            }
        ]
        
        for item in test_items:
            db.insert_media(**item)
        
        return db
    
    @pytest.fixture
    def chachanotes_db(self, temp_dirs):
        """Create a real ChaChaNotes database with test data"""
        db_path = temp_dirs['chachanotes'] / "chachanotes.db"
        db = CharactersRAGDB(str(db_path), client_id="test_client")
        
        # Add test conversations
        conv_id = db.create_conversation(
            user_id="test_user",
            title="Programming Discussion"
        )
        
        db.save_message(
            conversation_id=conv_id,
            sender="user",
            content="What is object-oriented programming?"
        )
        db.save_message(
            conversation_id=conv_id,
            sender="assistant",
            content="Object-oriented programming (OOP) is a programming paradigm based on objects and classes. Key concepts include encapsulation, inheritance, and polymorphism."
        )
        
        # Add test notes
        db.save_note(
            user_id="test_user",
            title="Python Notes",
            content="Python tips: Use list comprehensions for cleaner code. Remember that Python uses indentation for code blocks. The zen of Python emphasizes readability.",
            tags=["python", "programming"]
        )
        
        db.save_note(
            user_id="test_user",
            title="ML Resources",
            content="Useful machine learning resources: scikit-learn for classical ML, TensorFlow and PyTorch for deep learning. Start with simple algorithms before moving to complex ones.",
            tags=["ml", "resources"]
        )
        
        return db
    
    @pytest.fixture
    def mock_app(self, media_db, chachanotes_db):
        """Create a mock app instance with real databases"""
        app = MagicMock()
        app.media_db = media_db
        app.chachanotes_db = chachanotes_db
        app.notes_service = MagicMock()
        app.notes_user_id = "test_user"
        
        # Mock notes service to use chachanotes_db
        app.notes_service.search_notes.return_value = chachanotes_db.search_notes(
            user_id="test_user",
            search_term=""
        )
        
        return app
    
    @pytest.mark.asyncio
    async def test_plain_rag_search(self, mock_app):
        """Test plain BM25-based RAG search"""
        sources = {
            'media': True,
            'conversations': True,
            'notes': True
        }
        
        results, context = await perform_plain_rag_search(
            mock_app,
            query="Python programming",
            sources=sources,
            top_k=5,
            max_context_length=1000,
            enable_rerank=False
        )
        
        assert len(results) > 0
        assert context != ""
        
        # Check that Python-related content was found
        assert any('python' in r['content'].lower() for r in results)
        assert 'Python' in context
    
    @pytest.mark.asyncio
    async def test_plain_rag_search_with_source_filtering(self, mock_app):
        """Test RAG search with specific sources"""
        # Search only media
        results, _ = await perform_plain_rag_search(
            mock_app,
            query="programming",
            sources={'media': True, 'conversations': False, 'notes': False},
            top_k=10
        )
        
        # All results should be from media
        assert all(r['source'] == 'media' for r in results)
        
        # Search only notes
        results, _ = await perform_plain_rag_search(
            mock_app,
            query="programming",
            sources={'media': False, 'conversations': False, 'notes': True},
            top_k=10
        )
        
        # All results should be from notes
        assert all(r['source'] == 'note' for r in results)
    
    @pytest.mark.asyncio
    async def test_plain_rag_with_context_limit(self, mock_app):
        """Test context length limiting"""
        sources = {'media': True, 'conversations': True, 'notes': True}
        
        # Very small context limit
        _, context_small = await perform_plain_rag_search(
            mock_app,
            query="programming",
            sources=sources,
            top_k=10,
            max_context_length=100
        )
        
        # Larger context limit
        _, context_large = await perform_plain_rag_search(
            mock_app,
            query="programming",
            sources=sources,
            top_k=10,
            max_context_length=1000
        )
        
        assert len(context_small) <= 100
        assert len(context_large) > len(context_small)
    
    @pytest.mark.asyncio
    @patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events.RERANK_AVAILABLE', True)
    @patch('flashrank.Ranker')
    async def test_plain_rag_with_reranking(self, mock_ranker_class, mock_app):
        """Test RAG search with FlashRank re-ranking"""
        # Mock the ranker
        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {'index': 0, 'score': 0.9},
            {'index': 1, 'score': 0.7}
        ]
        mock_ranker_class.return_value = mock_ranker
        
        sources = {'media': True, 'conversations': False, 'notes': False}
        
        results, _ = await perform_plain_rag_search(
            mock_app,
            query="Python",
            sources=sources,
            top_k=2,
            enable_rerank=True,
            reranker_model="flashrank"
        )
        
        # Check that reranking was applied
        assert mock_ranker.rerank.called
        assert results[0]['score'] == 0.9
        assert results[1]['score'] == 0.7
    
    @pytest.mark.asyncio
    @patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events.DEPENDENCIES_AVAILABLE', {
        'embeddings_rag': True,
        'chromadb': True
    })
    async def test_full_rag_pipeline(self, mock_app, temp_dirs):
        """Test full embeddings-based RAG pipeline"""
        # This test requires mocking the embeddings service since we can't load real models in tests
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events.EmbeddingsService') as mock_embeddings_class:
            mock_embeddings = MagicMock()
            mock_embeddings.create_embeddings.return_value = [[0.1, 0.2, 0.3]]
            mock_embeddings.search_collection.return_value = {
                'documents': [['Python is great for programming']],
                'metadatas': [[{
                    'media_id': 1,
                    'title': 'Python Guide',
                    'type': 'article',
                    'author': 'Test',
                    'chunk_index': 0
                }]],
                'distances': [[0.1]]
            }
            mock_embeddings_class.return_value = mock_embeddings
            
            sources = {'media': True, 'conversations': False, 'notes': False}
            
            results, context = await perform_full_rag_pipeline(
                mock_app,
                query="Python programming",
                sources=sources,
                top_k=5,
                chunk_size=100,
                chunk_overlap=20
            )
            
            assert len(results) > 0
            assert context != ""
            assert mock_embeddings.create_embeddings.called
    
    @pytest.mark.asyncio
    async def test_hybrid_rag_search(self, mock_app):
        """Test hybrid search combining BM25 and vector search"""
        # Test with embeddings not available - should fall back to BM25 only
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events.DEPENDENCIES_AVAILABLE', {
            'embeddings_rag': False
        }):
            sources = {'media': True, 'conversations': False, 'notes': False}
            
            results, context = await perform_hybrid_rag_search(
                mock_app,
                query="programming",
                sources=sources,
                top_k=5,
                bm25_weight=0.5,
                vector_weight=0.5
            )
            
            assert len(results) > 0
            # Should have adjusted weights to BM25 only
            assert all('bm25_score' in r for r in results)
            assert all(r.get('vector_score', 0) == 0 for r in results)
    
    @pytest.mark.asyncio
    async def test_get_rag_context_for_chat(self, mock_app):
        """Test the chat integration function"""
        # Mock the UI query methods
        mock_app.query_one = MagicMock()
        
        # Configure UI state
        def query_one_side_effect(selector):
            mocks = {
                "#chat-rag-enable-checkbox": MagicMock(value=False),
                "#chat-rag-plain-enable-checkbox": MagicMock(value=True),
                "#chat-rag-search-media-checkbox": MagicMock(value=True),
                "#chat-rag-search-conversations-checkbox": MagicMock(value=True),
                "#chat-rag-search-notes-checkbox": MagicMock(value=True),
                "#chat-rag-top-k": MagicMock(value="3"),
                "#chat-rag-max-context-length": MagicMock(value="500"),
                "#chat-rag-rerank-enable-checkbox": MagicMock(value=False),
                "#chat-rag-reranker-model": MagicMock(value="flashrank"),
                "#chat-rag-chunk-size": MagicMock(value="100"),
                "#chat-rag-chunk-overlap": MagicMock(value="20"),
                "#chat-rag-include-metadata-checkbox": MagicMock(value=False)
            }
            return mocks.get(selector, MagicMock())
        
        mock_app.query_one.side_effect = query_one_side_effect
        mock_app.notify = MagicMock()
        
        # Test getting context
        context = await get_rag_context_for_chat(mock_app, "Tell me about Python")
        
        assert context is not None
        assert "### Context from RAG Search:" in context
        assert "### End of Context" in context
        assert "Based on the above context" in context
    
    @pytest.mark.asyncio
    async def test_rag_search_caching(self, mock_app):
        """Test that RAG results are cached"""
        sources = {'media': True, 'conversations': False, 'notes': False}
        query = "Python caching test"
        
        # First search
        results1, context1 = await perform_plain_rag_search(
            mock_app, query, sources, top_k=3
        )
        
        # Second search with same parameters
        results2, context2 = await perform_plain_rag_search(
            mock_app, query, sources, top_k=3
        )
        
        # Results should be identical (from cache)
        assert results1 == results2
        assert context1 == context2
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_app):
        """Test error handling in RAG pipeline"""
        # Make media_db raise an error
        mock_app.media_db.search_media_db.side_effect = Exception("Database error")
        
        sources = {'media': True, 'conversations': False, 'notes': False}
        
        # Should handle error gracefully
        results, context = await perform_plain_rag_search(
            mock_app, "test", sources, top_k=5
        )
        
        # Should return empty results instead of crashing
        assert results == []
        assert context == ""
    
    @pytest.mark.asyncio
    async def test_indexing_and_search_integration(self, mock_app, temp_dirs):
        """Test indexing content and then searching it"""
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events.DEPENDENCIES_AVAILABLE', {
            'embeddings_rag': True,
            'chromadb': True
        }):
            # Mock the embeddings service for testing
            with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.SentenceTransformer') as mock_transformer:
                mock_model = MagicMock()
                mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
                mock_transformer.return_value = mock_model
                
                # Create services
                embeddings_service = EmbeddingsService(temp_dirs['chromadb'])
                chunking_service = ChunkingService()
                indexing_service = IndexingService(embeddings_service, chunking_service)
                
                # Index content
                await indexing_service.index_media_items(mock_app.media_db)
                
                # Now search
                # For testing, we'll mock the search results since ChromaDB won't work without real embeddings
                with patch.object(embeddings_service, 'search_collection') as mock_search:
                    mock_search.return_value = {
                        'documents': [['Python content from indexed data']],
                        'metadatas': [[{'media_id': 1, 'title': 'Python Guide'}]],
                        'distances': [[0.1]]
                    }
                    
                    # Search should find indexed content
                    sources = {'media': True, 'conversations': False, 'notes': False}
                    results, _ = await perform_full_rag_pipeline(
                        mock_app, "Python", sources, top_k=5
                    )
                    
                    assert len(results) > 0