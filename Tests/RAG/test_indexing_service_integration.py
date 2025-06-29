# test_indexing_service_integration.py
# Integration tests for the RAG indexing service using real components

import pytest
import pytest_asyncio
import tempfile
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any

from tldw_chatbook.RAG_Search.Services.indexing_service import IndexingService
from tldw_chatbook.RAG_Search.Services.embeddings_service import EmbeddingsService
from tldw_chatbook.RAG_Search.Services.chunking_service import ChunkingService
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE

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
    """Create temporary directories for testing"""
    base_dir = tempfile.mkdtemp()
    dirs = {
        'embeddings': Path(base_dir) / 'embeddings',
        'db': Path(base_dir) / 'db'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    yield dirs
    
    shutil.rmtree(base_dir)


@pytest.fixture
def real_media_db(temp_dirs):
    """Create a real MediaDatabase instance"""
    db_path = temp_dirs['db'] / 'media.db'
    db = MediaDatabase(str(db_path), 'test_client')
    
    # Add test media items
    test_items = [
        {
            'title': 'Introduction to Machine Learning',
            'content': 'Machine learning is a subset of artificial intelligence that focuses on the ability of machines to receive data and learn for themselves. It involves algorithms that parse data, learn from it, and then apply what they have learned to make informed decisions.',
            'media_type': 'article',
            'author': 'AI Researcher',
            'url': 'https://example.com/ml-intro'
        },
        {
            'title': 'Deep Learning Fundamentals',
            'content': 'Deep learning is a specialized form of machine learning that uses neural networks with multiple layers. These deep neural networks attempt to simulate the behavior of the human brain in processing data and creating patterns for use in decision making.',
            'media_type': 'tutorial',
            'author': 'DL Expert',
            'url': 'https://example.com/dl-fundamentals'
        },
        {
            'title': 'Natural Language Processing Overview',
            'content': 'Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics.',
            'media_type': 'article',
            'author': 'NLP Specialist',
            'url': 'https://example.com/nlp-overview'
        }
    ]
    
    for item in test_items:
        db.insert_media_item(**item)
    
    yield db
    db.close()


@pytest.fixture
def real_chachanotes_db(temp_dirs):
    """Create a real CharactersRAGDB instance"""
    db_path = temp_dirs['db'] / 'chachanotes.db'
    db = CharactersRAGDB(str(db_path), 'test_client')
    
    # Add test conversations
    conv1_id = db.create_conversation(
        title="Chat about AI",
        messages=[
            {"role": "user", "content": "What is artificial intelligence?"},
            {"role": "assistant", "content": "Artificial intelligence is the simulation of human intelligence in machines."},
            {"role": "user", "content": "How does machine learning relate to AI?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data."}
        ]
    )
    
    conv2_id = db.create_conversation(
        title="Programming Discussion",
        messages=[
            {"role": "user", "content": "What's the best programming language for beginners?"},
            {"role": "assistant", "content": "Python is often recommended for beginners due to its simple syntax."},
            {"role": "user", "content": "What about for web development?"},
            {"role": "assistant", "content": "JavaScript is essential for web development, along with HTML and CSS."}
        ]
    )
    
    # Add test notes
    db.add_note(
        title="AI Research Notes",
        content="Key concepts in AI:\n- Machine Learning\n- Deep Learning\n- Neural Networks\n- Natural Language Processing"
    )
    
    db.add_note(
        title="Project Ideas",
        content="1. Build a chatbot using NLP\n2. Create an image classifier\n3. Develop a recommendation system"
    )
    
    yield db
    db.close()


@pytest.fixture
def real_indexing_db(temp_dirs):
    """Create a real RAGIndexingDB instance"""
    db_path = temp_dirs['db'] / 'rag_indexing.db'
    return RAGIndexingDB(str(db_path))


@pytest.fixture
def real_embeddings_service(temp_dirs):
    """Create a real embeddings service"""
    service = EmbeddingsService(persist_directory=str(temp_dirs['embeddings']))
    
    if DEPENDENCIES_AVAILABLE.get('sentence_transformers', False):
        # Initialize with a small model for testing
        service.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    
    return service


@pytest.fixture
def real_chunking_service():
    """Create a real chunking service"""
    return ChunkingService(
        chunk_size=200,  # Smaller chunks for testing
        chunk_overlap=50
    )


@pytest.fixture
def real_indexing_service(real_embeddings_service, real_chunking_service, real_indexing_db):
    """Create a real indexing service with all dependencies"""
    return IndexingService(
        embeddings_service=real_embeddings_service,
        chunking_service=real_chunking_service,
        indexing_db=real_indexing_db
    )


#######################################################################################################################
#
# Test Classes

class TestRealIndexingService:
    """Test indexing service with real components"""
    
    @pytest.mark.asyncio
    @requires_embeddings
    async def test_real_index_media_items(self, real_indexing_service, real_media_db):
        """Test indexing real media items"""
        # Index all media items
        indexed_count = await real_indexing_service.index_media_items(real_media_db)
        
        assert indexed_count == 3  # We added 3 test items
        
        # Verify items were indexed in the database
        indexed_items = real_indexing_service.indexing_db.get_indexed_items_by_type('media')
        assert len(indexed_items) == 3
        
        # Verify chunks were created and stored
        collections = real_indexing_service.embeddings_service.list_collections()
        assert "media_chunks" in collections
        
        # Search for indexed content
        query = "What is machine learning?"
        query_embeddings = real_indexing_service.embeddings_service.create_embeddings([query])
        
        results = real_indexing_service.embeddings_service.search_collection(
            "media_chunks",
            query_embeddings,
            n_results=5
        )
        
        assert results is not None
        assert len(results["documents"][0]) > 0
        
        # Should find ML-related content
        found_ml = any("machine learning" in doc.lower() for doc in results["documents"][0])
        assert found_ml
    
    @pytest.mark.asyncio
    @requires_embeddings
    async def test_real_index_conversations(self, real_indexing_service, real_chachanotes_db):
        """Test indexing real conversations"""
        # Index all conversations
        indexed_count = await real_indexing_service.index_conversations(real_chachanotes_db)
        
        assert indexed_count == 2  # We added 2 test conversations
        
        # Verify items were indexed
        indexed_items = real_indexing_service.indexing_db.get_indexed_items_by_type('conversation')
        assert len(indexed_items) == 2
        
        # Verify conversation chunks exist
        collections = real_indexing_service.embeddings_service.list_collections()
        assert "conversation_chunks" in collections
        
        # Search for conversation content
        query = "programming language for beginners"
        query_embeddings = real_indexing_service.embeddings_service.create_embeddings([query])
        
        results = real_indexing_service.embeddings_service.search_collection(
            "conversation_chunks",
            query_embeddings,
            n_results=5
        )
        
        assert results is not None
        assert len(results["documents"][0]) > 0
        
        # Should find Python recommendation
        found_python = any("python" in doc.lower() for doc in results["documents"][0])
        assert found_python
    
    @pytest.mark.asyncio
    @requires_embeddings
    async def test_real_index_notes(self, real_indexing_service, real_chachanotes_db):
        """Test indexing real notes"""
        # Index all notes
        indexed_count = await real_indexing_service.index_notes(real_chachanotes_db)
        
        assert indexed_count == 2  # We added 2 test notes
        
        # Verify items were indexed
        indexed_items = real_indexing_service.indexing_db.get_indexed_items_by_type('note')
        assert len(indexed_items) == 2
        
        # Search for note content
        query = "chatbot NLP project"
        query_embeddings = real_indexing_service.embeddings_service.create_embeddings([query])
        
        results = real_indexing_service.embeddings_service.search_collection(
            "note_chunks",
            query_embeddings,
            n_results=3
        )
        
        assert results is not None
        assert len(results["documents"][0]) > 0
    
    @pytest.mark.asyncio
    @requires_embeddings
    async def test_real_progress_callback(self, real_indexing_service, real_media_db):
        """Test progress callback with real indexing"""
        progress_updates = []
        
        def progress_callback(item_type, current, total):
            progress_updates.append({
                'type': item_type,
                'current': current,
                'total': total,
                'percentage': (current / total * 100) if total > 0 else 0
            })
        
        # Index with progress callback
        await real_indexing_service.index_media_items(
            real_media_db,
            progress_callback=progress_callback
        )
        
        # Should have progress updates
        assert len(progress_updates) > 0
        
        # First update should be 0%
        assert progress_updates[0]['current'] == 0
        assert progress_updates[0]['percentage'] == 0
        
        # Last update should be 100%
        assert progress_updates[-1]['current'] == progress_updates[-1]['total']
        assert progress_updates[-1]['percentage'] == 100
    
    @pytest.mark.asyncio
    @requires_embeddings
    async def test_real_incremental_indexing(self, real_indexing_service, real_media_db):
        """Test that already indexed items are skipped"""
        # Index items first time
        first_count = await real_indexing_service.index_media_items(real_media_db)
        assert first_count == 3
        
        # Index again - should skip already indexed items
        second_count = await real_indexing_service.index_media_items(real_media_db)
        assert second_count == 0
        
        # Add a new item
        real_media_db.insert_media_item(
            title="New Article",
            content="This is a new article about reinforcement learning.",
            media_type="article",
            author="RL Expert"
        )
        
        # Index again - should only index the new item
        third_count = await real_indexing_service.index_media_items(real_media_db)
        assert third_count == 1
    
    @pytest.mark.asyncio
    @requires_embeddings
    async def test_real_error_handling(self, real_indexing_service, real_media_db):
        """Test error handling during indexing"""
        # Create a media item with very long content that might cause issues
        real_media_db.insert_media_item(
            title="Problematic Item",
            content="x" * 100000,  # Very long content
            media_type="article",
            author="Test"
        )
        
        # Should handle gracefully
        try:
            indexed_count = await real_indexing_service.index_media_items(real_media_db)
            # Should index at least the original 3 items
            assert indexed_count >= 3
        except Exception as e:
            pytest.fail(f"Indexing should handle errors gracefully: {e}")
    
    @pytest.mark.asyncio
    @requires_embeddings
    async def test_real_chunking_quality(self, real_indexing_service, real_media_db, real_embeddings_service):
        """Test that chunking produces meaningful segments"""
        # Add a document with clear sections
        real_media_db.insert_media_item(
            title="Structured Document",
            content="""
            Introduction: This document covers three main topics.
            
            Topic 1 - Machine Learning:
            Machine learning is a method of data analysis that automates analytical model building.
            It is based on the idea that systems can learn from data.
            
            Topic 2 - Deep Learning:
            Deep learning is part of a broader family of machine learning methods.
            It is based on artificial neural networks with representation learning.
            
            Topic 3 - Natural Language Processing:
            NLP is a subfield of linguistics, computer science, and artificial intelligence.
            It is concerned with the interactions between computers and human language.
            
            Conclusion: These three areas form the foundation of modern AI.
            """,
            media_type="article",
            author="AI Expert"
        )
        
        # Index the new item
        await real_indexing_service.index_media_items(real_media_db)
        
        # Search for specific topics
        topics = ["machine learning", "deep learning", "natural language processing"]
        
        for topic in topics:
            query_embeddings = real_embeddings_service.create_embeddings([f"Tell me about {topic}"])
            results = real_embeddings_service.search_collection(
                "media_chunks",
                query_embeddings,
                n_results=3
            )
            
            # Should find relevant chunks for each topic
            assert results is not None
            found_topic = any(topic.lower() in doc.lower() for doc in results["documents"][0])
            assert found_topic, f"Should find chunks about {topic}"
    
    @pytest.mark.asyncio
    @requires_embeddings
    async def test_real_concurrent_indexing(self, real_embeddings_service, real_chunking_service, real_indexing_db, temp_dirs):
        """Test concurrent indexing operations"""
        # Create multiple indexing services
        services = []
        for i in range(3):
            service = IndexingService(
                embeddings_service=real_embeddings_service,
                chunking_service=real_chunking_service,
                indexing_db=real_indexing_db
            )
            services.append(service)
        
        # Create separate databases for each service
        media_dbs = []
        for i in range(3):
            db_path = temp_dirs['db'] / f'media_{i}.db'
            db = MediaDatabase(str(db_path), f'test_client_{i}')
            
            # Add unique items to each database
            for j in range(2):
                db.insert_media_item(
                    title=f"Document {i}-{j}",
                    content=f"Content for document {i}-{j} about topic {i}",
                    media_type="article",
                    author=f"Author {i}"
                )
            
            media_dbs.append(db)
        
        # Run concurrent indexing
        tasks = []
        for service, db in zip(services, media_dbs):
            task = asyncio.create_task(service.index_media_items(db))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Each should index 2 items
        assert all(count == 2 for count in results)
        
        # Total indexed items should be 6
        all_indexed = real_indexing_db.get_indexed_items_by_type('media')
        assert len(all_indexed) == 6
        
        # Cleanup
        for db in media_dbs:
            db.close()