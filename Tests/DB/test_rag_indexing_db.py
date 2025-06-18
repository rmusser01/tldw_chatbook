# test_rag_indexing_db.py
# Description: Unit tests for RAG indexing state tracking database
#
"""
test_rag_indexing_db.py
-----------------------

Unit tests for the RAG indexing database that tracks indexed items
and supports incremental indexing.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
import time

from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB


class TestRAGIndexingDB:
    """Test cases for RAG indexing database."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = Path(tmp.name)
        
        db = RAGIndexingDB(db_path)
        yield db
        
        # Cleanup
        db_path.unlink(missing_ok=True)
    
    def test_initialization(self, temp_db):
        """Test database initialization and schema creation."""
        # Check that tables exist
        with temp_db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            
        assert 'indexed_items' in tables
        assert 'collection_state' in tables
    
    def test_mark_item_indexed(self, temp_db):
        """Test marking an item as indexed."""
        # Mark item as indexed
        item_id = "test_item_1"
        item_type = "media"
        last_modified = datetime.now(timezone.utc)
        
        temp_db.mark_item_indexed(
            item_id=item_id,
            item_type=item_type,
            last_modified=last_modified,
            chunk_count=5
        )
        
        # Verify item was indexed
        indexed_items = temp_db.get_indexed_items_by_type(item_type)
        assert item_id in indexed_items
        assert abs((indexed_items[item_id] - last_modified).total_seconds()) < 1
    
    def test_update_existing_item(self, temp_db):
        """Test updating an already indexed item."""
        item_id = "test_item_1"
        item_type = "media"
        
        # First indexing
        first_modified = datetime.now(timezone.utc)
        temp_db.mark_item_indexed(
            item_id=item_id,
            item_type=item_type,
            last_modified=first_modified,
            chunk_count=5
        )
        
        # Wait a bit and update
        time.sleep(0.1)
        second_modified = datetime.now(timezone.utc)
        temp_db.mark_item_indexed(
            item_id=item_id,
            item_type=item_type,
            last_modified=second_modified,
            chunk_count=10
        )
        
        # Verify update
        indexed_items = temp_db.get_indexed_items_by_type(item_type)
        assert item_id in indexed_items
        assert indexed_items[item_id] > first_modified
    
    def test_get_indexed_items_by_type(self, temp_db):
        """Test retrieving indexed items by type."""
        # Index items of different types
        now = datetime.now(timezone.utc)
        
        temp_db.mark_item_indexed("media_1", "media", now, 5)
        temp_db.mark_item_indexed("media_2", "media", now, 3)
        temp_db.mark_item_indexed("note_1", "note", now, 2)
        temp_db.mark_item_indexed("conv_1", "conversation", now, 4)
        
        # Get items by type
        media_items = temp_db.get_indexed_items_by_type("media")
        note_items = temp_db.get_indexed_items_by_type("note")
        conv_items = temp_db.get_indexed_items_by_type("conversation")
        
        assert len(media_items) == 2
        assert len(note_items) == 1
        assert len(conv_items) == 1
        
        assert "media_1" in media_items
        assert "media_2" in media_items
        assert "note_1" in note_items
        assert "conv_1" in conv_items
    
    def test_is_item_indexed(self, temp_db):
        """Test checking if an item is indexed."""
        item_id = "test_item"
        item_type = "media"
        
        # Check before indexing
        assert not temp_db.is_item_indexed(item_id, item_type)
        
        # Index item
        temp_db.mark_item_indexed(
            item_id=item_id,
            item_type=item_type,
            last_modified=datetime.now(timezone.utc),
            chunk_count=5
        )
        
        # Check after indexing
        assert temp_db.is_item_indexed(item_id, item_type)
    
    def test_needs_reindexing(self, temp_db):
        """Test checking if an item needs reindexing."""
        item_id = "test_item"
        item_type = "media"
        
        # Index item with old timestamp
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        temp_db.mark_item_indexed(
            item_id=item_id,
            item_type=item_type,
            last_modified=old_time,
            chunk_count=5
        )
        
        # Check with newer timestamp
        new_time = datetime.now(timezone.utc)
        assert temp_db.needs_reindexing(item_id, item_type, new_time)
        
        # Check with older timestamp
        older_time = old_time - timedelta(hours=1)
        assert not temp_db.needs_reindexing(item_id, item_type, older_time)
        
        # Check non-existent item (should need indexing)
        assert temp_db.needs_reindexing("non_existent", item_type, new_time)
    
    def test_remove_item(self, temp_db):
        """Test removing an indexed item."""
        item_id = "test_item"
        item_type = "media"
        
        # Index item
        temp_db.mark_item_indexed(
            item_id=item_id,
            item_type=item_type,
            last_modified=datetime.now(timezone.utc),
            chunk_count=5
        )
        
        # Verify indexed
        assert temp_db.is_item_indexed(item_id, item_type)
        
        # Remove item
        assert temp_db.remove_item(item_id, item_type)
        
        # Verify removed
        assert not temp_db.is_item_indexed(item_id, item_type)
        
        # Try removing non-existent item
        assert not temp_db.remove_item("non_existent", item_type)
    
    def test_update_collection_state(self, temp_db):
        """Test updating collection state."""
        collection_name = "media_chunks"
        
        # Update state
        temp_db.update_collection_state(
            collection_name=collection_name,
            total_items=100,
            indexed_items=95,
            last_full_index=datetime.now(timezone.utc)
        )
        
        # Get state
        state = temp_db.get_collection_state(collection_name)
        assert state is not None
        assert state['total_items'] == 100
        assert state['indexed_items'] == 95
        assert state['last_full_index'] is not None
    
    def test_get_indexing_stats(self, temp_db):
        """Test getting indexing statistics."""
        now = datetime.now(timezone.utc)
        
        # Index various items
        temp_db.mark_item_indexed("media_1", "media", now, 5)
        temp_db.mark_item_indexed("media_2", "media", now, 3)
        temp_db.mark_item_indexed("note_1", "note", now, 2)
        
        # Update collection states
        temp_db.update_collection_state("media_chunks", 2, 2)
        temp_db.update_collection_state("notes_chunks", 1, 1)
        
        # Get stats
        stats = temp_db.get_indexing_stats()
        
        assert stats['total_indexed_items'] == 3
        assert stats['items_by_type']['media'] == 2
        assert stats['items_by_type']['note'] == 1
        assert stats['total_chunks'] == 10  # 5 + 3 + 2
        assert len(stats['collection_states']) == 2
    
    def test_clear_all(self, temp_db):
        """Test clearing all indexing data."""
        # Add some data
        now = datetime.now(timezone.utc)
        temp_db.mark_item_indexed("item_1", "media", now, 5)
        temp_db.mark_item_indexed("item_2", "note", now, 3)
        temp_db.update_collection_state("media_chunks", 1, 1)
        
        # Verify data exists
        stats = temp_db.get_indexing_stats()
        assert stats['total_indexed_items'] > 0
        
        # Clear all
        temp_db.clear_all()
        
        # Verify data cleared
        stats = temp_db.get_indexing_stats()
        assert stats['total_indexed_items'] == 0
        assert len(stats['items_by_type']) == 0
        assert len(stats['collection_states']) == 0
    
    def test_concurrent_access(self, temp_db):
        """Test concurrent access to the database."""
        import threading
        import random
        
        errors = []
        
        def index_items(thread_id):
            """Index items from a thread."""
            try:
                for i in range(10):
                    item_id = f"thread_{thread_id}_item_{i}"
                    temp_db.mark_item_indexed(
                        item_id=item_id,
                        item_type="media",
                        last_modified=datetime.now(timezone.utc),
                        chunk_count=random.randint(1, 10)
                    )
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=index_items, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check no errors
        assert len(errors) == 0
        
        # Verify all items indexed
        media_items = temp_db.get_indexed_items_by_type("media")
        assert len(media_items) == 50  # 5 threads * 10 items
    
    def test_timestamp_precision(self, temp_db):
        """Test that timestamp precision is maintained."""
        item_id = "test_item"
        item_type = "media"
        
        # Create timestamp with microseconds
        precise_time = datetime(2024, 1, 1, 12, 30, 45, 123456, tzinfo=timezone.utc)
        
        # Index item
        temp_db.mark_item_indexed(
            item_id=item_id,
            item_type=item_type,
            last_modified=precise_time,
            chunk_count=5
        )
        
        # Retrieve and compare
        indexed_items = temp_db.get_indexed_items_by_type(item_type)
        retrieved_time = indexed_items[item_id]
        
        # Check precision (SQLite stores to microsecond precision)
        assert abs((retrieved_time - precise_time).total_seconds()) < 0.000001
    
    def test_large_batch_operations(self, temp_db):
        """Test handling large batches of items."""
        item_type = "media"
        batch_size = 1000
        now = datetime.now(timezone.utc)
        
        # Index large batch
        start_time = time.time()
        for i in range(batch_size):
            temp_db.mark_item_indexed(
                item_id=f"item_{i}",
                item_type=item_type,
                last_modified=now,
                chunk_count=1
            )
        index_time = time.time() - start_time
        
        # Verify all indexed
        indexed_items = temp_db.get_indexed_items_by_type(item_type)
        assert len(indexed_items) == batch_size
        
        # Performance check - should complete reasonably fast
        assert index_time < 10.0  # 10 seconds for 1000 items
        
        # Test retrieval performance
        start_time = time.time()
        indexed_items = temp_db.get_indexed_items_by_type(item_type)
        retrieve_time = time.time() - start_time
        
        assert retrieve_time < 1.0  # Should retrieve in under 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])