# test_indexing_service.py
# Unit tests for the RAG indexing service

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, call

from tldw_chatbook.RAG_Search.Services.indexing_service import IndexingService


class TestIndexingService:
    """Test the document indexing service"""
    
    @pytest.fixture
    def mock_embeddings_service(self):
        """Create a mock embeddings service"""
        mock = MagicMock()
        mock.create_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock.add_documents_to_collection.return_value = True
        return mock
    
    @pytest.fixture
    def mock_chunking_service(self):
        """Create a mock chunking service"""
        mock = MagicMock()
        mock.chunk_document.return_value = [
            {
                'chunk_id': 'chunk1',
                'chunk_index': 0,
                'text': 'First chunk',
                'metadata': {}
            },
            {
                'chunk_id': 'chunk2',
                'chunk_index': 1,
                'text': 'Second chunk',
                'metadata': {}
            }
        ]
        return mock
    
    @pytest.fixture
    def indexing_service(self, mock_embeddings_service, mock_chunking_service):
        """Create an indexing service instance"""
        return IndexingService(mock_embeddings_service, mock_chunking_service)
    
    @pytest.fixture
    def mock_media_db(self):
        """Create a mock media database"""
        mock = MagicMock()
        # Mock fetch_paginated_media to return data on first call, empty on second
        mock.fetch_paginated_media.side_effect = [
            [  # First page
                {
                    'id': 1,
                    'title': 'Media Item 1',
                    'content': 'Content of media item 1',
                    'type': 'article',
                    'author': 'Author 1'
                },
                {
                    'id': 2,
                    'title': 'Media Item 2',
                    'content': 'Content of media item 2',
                    'type': 'video',
                    'author': 'Author 2'
                }
            ],
            []  # Second page (empty - signals end)
        ]
        # Mock search for getting total count
        mock.search_media_db.return_value = ([], 2)  # 2 total items
        return mock
    
    @pytest.fixture
    def mock_chachanotes_db(self):
        """Create a mock ChaChaNotes database"""
        mock = MagicMock()
        mock.list_conversations.return_value = [
            {'id': 'conv1', 'title': 'Conversation 1', 'character_id': 'char1'},
            {'id': 'conv2', 'title': 'Conversation 2', 'character_id': 'char2'}
        ]
        mock.get_messages_for_conversation.side_effect = [
            [  # Messages for conv1
                {'sender': 'user', 'content': 'Hello'},
                {'sender': 'assistant', 'content': 'Hi there'}
            ],
            [  # Messages for conv2
                {'sender': 'user', 'content': 'How are you?'},
                {'sender': 'assistant', 'content': 'I am fine'}
            ]
        ]
        mock.list_notes.return_value = [
            {'id': 'note1', 'title': 'Note 1', 'content': 'Note content 1', 'tags': ['tag1']},
            {'id': 'note2', 'title': 'Note 2', 'content': 'Note content 2', 'tags': ['tag2']}
        ]
        return mock
    
    @pytest.mark.asyncio
    async def test_index_media_items(self, indexing_service, mock_media_db, mock_embeddings_service, mock_chunking_service):
        """Test indexing media items"""
        result = await indexing_service.index_media_items(mock_media_db)
        
        assert result == 2  # Two items indexed
        
        # Check that fetch was called correctly
        assert mock_media_db.fetch_paginated_media.call_count == 2
        
        # Check that chunking was called for each item
        assert mock_chunking_service.chunk_document.call_count == 2
        
        # Check that embeddings were created
        assert mock_embeddings_service.create_embeddings.called
        
        # Check that documents were added to collection
        assert mock_embeddings_service.add_documents_to_collection.called
        call_args = mock_embeddings_service.add_documents_to_collection.call_args
        assert call_args[0][0] == "media_chunks"  # Collection name
    
    @pytest.mark.asyncio
    async def test_index_media_items_with_progress(self, indexing_service, mock_media_db):
        """Test indexing with progress callback"""
        progress_calls = []
        
        def progress_callback(type, current, total):
            progress_calls.append((type, current, total))
        
        await indexing_service.index_media_items(
            mock_media_db,
            progress_callback=progress_callback
        )
        
        # Check progress was reported
        assert len(progress_calls) > 0
        assert progress_calls[0] == ('media', 0, 2)  # Initial call
        assert progress_calls[-1] == ('media', 2, 2)  # Final call
    
    @pytest.mark.asyncio
    async def test_index_media_batch(self, indexing_service, mock_embeddings_service, mock_chunking_service):
        """Test indexing a batch of media items"""
        media_items = [
            {
                'id': 1,
                'title': 'Test Item',
                'content': 'Test content',
                'type': 'article',
                'author': 'Test Author'
            }
        ]
        
        await indexing_service._index_media_batch(media_items, 100, 20)
        
        # Check chunking was called
        mock_chunking_service.chunk_document.assert_called_once()
        
        # Check embeddings were created for chunks
        mock_embeddings_service.create_embeddings.assert_called_once()
        
        # Check documents were added with correct metadata
        add_call = mock_embeddings_service.add_documents_to_collection.call_args
        assert add_call[0][0] == "media_chunks"
        metadatas = add_call[1]['metadatas']
        assert metadatas[0]['media_id'] == 1
        assert metadatas[0]['title'] == 'Test Item'
    
    @pytest.mark.asyncio
    async def test_index_conversations(self, indexing_service, mock_chachanotes_db):
        """Test indexing conversations"""
        result = await indexing_service.index_conversations(mock_chachanotes_db)
        
        assert result == 2  # Two conversations indexed
        
        # Check that conversations were listed
        mock_chachanotes_db.list_conversations.assert_called_once()
        
        # Check that messages were fetched for each conversation
        assert mock_chachanotes_db.get_messages_for_conversation.call_count == 2
    
    @pytest.mark.asyncio
    async def test_index_conversations_with_progress(self, indexing_service, mock_chachanotes_db):
        """Test indexing conversations with progress callback"""
        progress_calls = []
        
        def progress_callback(type, current, total):
            progress_calls.append((type, current, total))
        
        await indexing_service.index_conversations(
            mock_chachanotes_db,
            progress_callback=progress_callback
        )
        
        # Check progress was reported
        assert len(progress_calls) > 0
        assert progress_calls[0] == ('conversations', 0, 2)
        assert progress_calls[-1] == ('conversations', 2, 2)
    
    @pytest.mark.asyncio
    async def test_index_conversation_batch(self, indexing_service, mock_chachanotes_db, mock_embeddings_service):
        """Test indexing a batch of conversations"""
        conversations = [{'id': 'conv1', 'title': 'Test Conversation'}]
        
        await indexing_service._index_conversation_batch(
            mock_chachanotes_db,
            conversations,
            100,
            20
        )
        
        # Check messages were fetched
        mock_chachanotes_db.get_messages_for_conversation.assert_called_once_with(
            conversation_id='conv1'
        )
        
        # Check documents were created with conversation format
        add_call = mock_embeddings_service.add_documents_to_collection.call_args
        documents = add_call[1]['documents']
        assert 'user:' in documents[0]
        assert 'assistant:' in documents[0]
    
    @pytest.mark.asyncio
    async def test_index_notes(self, indexing_service, mock_chachanotes_db):
        """Test indexing notes"""
        result = await indexing_service.index_notes(mock_chachanotes_db)
        
        assert result == 2  # Two notes indexed
        
        # Check that notes were listed
        mock_chachanotes_db.list_notes.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_index_notes_batch(self, indexing_service, mock_embeddings_service):
        """Test indexing a batch of notes"""
        notes = [
            {
                'id': 'note1',
                'title': 'Test Note',
                'content': 'Test note content',
                'tags': ['test', 'example']
            }
        ]
        
        await indexing_service._index_notes_batch(notes, 100, 20)
        
        # Check documents were added with correct metadata
        add_call = mock_embeddings_service.add_documents_to_collection.call_args
        assert add_call[0][0] == "notes_chunks"
        metadatas = add_call[1]['metadatas']
        assert metadatas[0]['note_id'] == 'note1'
        assert metadatas[0]['tags'] == ['test', 'example']
    
    @pytest.mark.asyncio
    async def test_index_all(self, indexing_service, mock_media_db, mock_chachanotes_db):
        """Test indexing all content types"""
        result = await indexing_service.index_all(
            media_db=mock_media_db,
            chachanotes_db=mock_chachanotes_db
        )
        
        assert result['media'] == 2
        assert result['conversations'] == 2
        assert result['notes'] == 2
    
    @pytest.mark.asyncio
    async def test_index_all_with_progress(self, indexing_service, mock_media_db, mock_chachanotes_db):
        """Test indexing all content with progress callback"""
        progress_calls = []
        
        def progress_callback(type, current, total):
            progress_calls.append((type, current, total))
        
        await indexing_service.index_all(
            media_db=mock_media_db,
            chachanotes_db=mock_chachanotes_db,
            progress_callback=progress_callback
        )
        
        # Check that progress was reported for all types
        types_reported = set(call[0] for call in progress_calls)
        assert 'media' in types_reported
        assert 'conversations' in types_reported
        assert 'notes' in types_reported
    
    @pytest.mark.asyncio
    async def test_index_all_partial(self, indexing_service, mock_media_db):
        """Test indexing with only some databases provided"""
        result = await indexing_service.index_all(
            media_db=mock_media_db,
            chachanotes_db=None
        )
        
        assert result['media'] == 2
        assert result['conversations'] == 0
        assert result['notes'] == 0
    
    @pytest.mark.asyncio
    async def test_empty_content_handling(self, indexing_service, mock_embeddings_service):
        """Test handling of empty content"""
        # Media item with no content
        media_items = [
            {'id': 1, 'title': 'Empty', 'content': '', 'type': 'article'}
        ]
        
        await indexing_service._index_media_batch(media_items, 100, 20)
        
        # Should not create embeddings for empty content
        mock_embeddings_service.create_embeddings.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_error_handling_media(self, indexing_service, mock_media_db):
        """Test error handling during media indexing"""
        # Make fetch raise an exception
        mock_media_db.fetch_paginated_media.side_effect = Exception("Database error")
        
        result = await indexing_service.index_media_items(mock_media_db)
        
        # Should handle error and return 0
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_conversations(self, indexing_service, mock_chachanotes_db):
        """Test error handling during conversation indexing"""
        # Make list_conversations raise an exception
        mock_chachanotes_db.list_conversations.side_effect = Exception("Database error")
        
        result = await indexing_service.index_conversations(mock_chachanotes_db)
        
        # Should handle error and return 0
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_chunk_parameters(self, indexing_service, mock_media_db, mock_chunking_service):
        """Test that chunk parameters are passed correctly"""
        chunk_size = 200
        chunk_overlap = 50
        
        await indexing_service.index_media_items(
            mock_media_db,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Check chunking was called with correct parameters
        mock_chunking_service.chunk_document.assert_called()
        call_args = mock_chunking_service.chunk_document.call_args
        assert call_args[0][1] == chunk_size
        assert call_args[0][2] == chunk_overlap
    
    @pytest.mark.asyncio
    async def test_unique_chunk_ids(self, indexing_service, mock_embeddings_service):
        """Test that unique chunk IDs are generated"""
        media_items = [
            {'id': 1, 'title': 'Item 1', 'content': 'Content 1', 'type': 'article'},
            {'id': 2, 'title': 'Item 2', 'content': 'Content 2', 'type': 'article'}
        ]
        
        await indexing_service._index_media_batch(media_items, 100, 20)
        
        # Check that IDs were generated correctly
        add_call = mock_embeddings_service.add_documents_to_collection.call_args
        ids = add_call[1]['ids']
        
        # All IDs should be unique
        assert len(ids) == len(set(ids))
        
        # IDs should follow the expected format
        for id in ids:
            assert id.startswith('media_')