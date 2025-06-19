"""
Unit tests for pagination functionality in database functions.
"""

import logging
import sqlite3
import tempfile
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock

from tldw_chatbook.DB.Client_Media_DB_v2 import (
    MediaDatabase, get_unprocessed_media, get_all_content_from_database
)


class TestMediaDatabasePagination:
    """Test pagination in Media database functions."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            yield tmp.name
        # Cleanup
        Path(tmp.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def mock_media_db(self, temp_db_path):
        """Create a mock MediaDatabase with test data."""
        db = MediaDatabase(temp_db_path, client_id="test_client")
        
        # The MediaDatabase constructor automatically creates the schema
        # We just need to insert test data
        conn = db.get_connection()
        try:
            # Insert test data
            for i in range(250):  # Create 250 test media items
                # Note: content_hash column is part of the actual schema
                db.execute_query("""
                    INSERT INTO Media (uuid, title, content, type, vector_processing, deleted, is_trash, 
                                     content_hash, last_modified, version, client_id)
                    VALUES (?, ?, ?, ?, ?, 0, 0, ?, datetime('now'), 1, ?)
                """, (
                    f"uuid_{i}",
                    f"Title {i}",
                    f"Content for item {i}",
                    "document",
                    0 if i < 150 else 1,  # First 150 need processing
                    f"hash_{i}",  # unique content_hash
                    "test_client"
                ), commit=True)
            
            # Insert test keywords
            for i in range(500):  # Create 500 test keywords
                db.execute_query("""
                    INSERT INTO Keywords (uuid, keyword, deleted, last_modified, version, client_id)
                    VALUES (?, ?, 0, datetime('now'), 1, ?)
                """, (f"kw_uuid_{i}", f"keyword_{i}", "test_client"), commit=True)
            
        except Exception as e:
            # Log error and re-raise
            logging.error(f"Error creating test data: {e}")
            raise
        
        return db
    
    def test_get_all_active_media_for_embedding_pagination(self, mock_media_db):
        """Test pagination in get_all_active_media_for_embedding."""
        # Test default behavior (no limit)
        results = mock_media_db.get_all_active_media_for_embedding()
        assert len(results) == 250  # All non-deleted, non-trash items
        
        # Test with limit
        results = mock_media_db.get_all_active_media_for_embedding(limit=10)
        assert len(results) == 10
        assert results[0]['id'] == 1
        assert results[9]['id'] == 10
        
        # Test with limit and offset
        results = mock_media_db.get_all_active_media_for_embedding(limit=10, offset=20)
        assert len(results) == 10
        assert results[0]['id'] == 21
        assert results[9]['id'] == 30
        
        # Test offset beyond data
        results = mock_media_db.get_all_active_media_for_embedding(limit=10, offset=300)
        assert len(results) == 0
    
    def test_fetch_all_keywords_no_pagination(self, mock_media_db):
        """Test fetch_all_keywords returns all keywords (no pagination support)."""
        # fetch_all_keywords doesn't support pagination parameters
        results = mock_media_db.fetch_all_keywords()
        assert len(results) == 500  # All keywords returned
        assert results[0] == "keyword_0"  # Verify ordering
        assert results[-1] == "keyword_99"  # Keywords are sorted
    
    def test_get_unprocessed_media_no_pagination(self, mock_media_db):
        """Test get_unprocessed_media returns all unprocessed items."""
        # get_unprocessed_media doesn't support pagination parameters
        results = get_unprocessed_media(mock_media_db)
        assert len(results) == 150  # All unprocessed items (vector_processing=0)
        
        # Verify they are the correct items
        for result in results:
            assert result['id'] <= 150  # First 150 have vector_processing=0
    
    def test_get_all_content_from_database_no_pagination(self, mock_media_db):
        """Test get_all_content_from_database returns all active items."""
        # get_all_content_from_database doesn't support pagination parameters
        results = get_all_content_from_database(mock_media_db)
        assert len(results) == 250  # All active, non-trashed items
        
        # Verify we got all items
        # Note: The ordering is by last_modified DESC, but since all items have the same timestamp,
        # the actual order may vary. Just verify we got all 250 items.
        all_ids = {r['id'] for r in results}
        assert all_ids == set(range(1, 251))  # All IDs from 1 to 250
    
    def test_get_all_active_media_for_embedding_supports_pagination(self, mock_media_db):
        """Test that get_all_active_media_for_embedding supports pagination."""
        # This method DOES support pagination
        results = mock_media_db.get_all_active_media_for_embedding(limit=None)
        assert len(results) == 250  # All media items
        
        # Test with limit
        results = mock_media_db.get_all_active_media_for_embedding(limit=10)
        assert len(results) == 10
        
        # Test with limit and offset
        results = mock_media_db.get_all_active_media_for_embedding(limit=10, offset=20)
        assert len(results) == 10


class TestPaginationEdgeCases:
    """Test edge cases in pagination."""
    
    def test_pagination_with_empty_database(self):
        """Test pagination when database is empty."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db = MediaDatabase(tmp.name, client_id="test_client")
            # Database schema is automatically created by MediaDatabase constructor
            # No need to manually create tables
            
            # Test with empty database
            assert db.get_all_active_media_for_embedding(limit=10) == []
            assert db.fetch_all_keywords() == []  # No pagination params
            assert get_unprocessed_media(db) == []  # No pagination params
            assert get_all_content_from_database(db) == []  # No pagination params
            
            # Cleanup
            Path(tmp.name).unlink(missing_ok=True)
    
    def test_pagination_with_negative_values(self):
        """Test pagination with negative limit/offset values."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db = MediaDatabase(tmp.name, client_id="test_client")
            
            # Add one item using the proper database method
            db.execute_query("""
                INSERT INTO Media (uuid, title, content, type, content_hash, last_modified, version, client_id)
                VALUES ('test_uuid', 'Test Title', 'Test Content', 'document', 'test_hash', datetime('now'), 1, 'test_client')
            """, commit=True)
            
            # SQLite treats negative LIMIT as no limit
            # SQLite treats negative OFFSET as 0
            results = db.get_all_active_media_for_embedding(limit=-1, offset=-1)
            assert len(results) == 1
            
            # Cleanup
            Path(tmp.name).unlink(missing_ok=True)


class TestBatchQueryOptimization:
    """Test query patterns in search results."""
    
    def test_search_prompts_keyword_fetch_pattern(self):
        """Test how search_prompts fetches keywords for results."""
        from tldw_chatbook.DB.Prompts_DB import PromptsDatabase as PromptsDB
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db = PromptsDB(tmp.name, client_id="test_client")
            
            # Mock the database to track queries
            original_execute = db.execute_query
            query_log = []
            
            def mock_execute(query, params=None):
                query_log.append(query)
                return original_execute(query, params)
            
            db.execute_query = mock_execute
            
            # PromptsDatabase automatically initializes schema in __init__
            # Add test data using PromptsDB methods
            # Add prompts
            for i in range(10):
                db.add_prompt(
                    name=f"Prompt {i}",
                    author="Test Author",
                    details="Test Details",
                    system_prompt="System",
                    user_prompt="User",
                    keywords=[f"keyword_{j}" for j in range(2)] if i < 5 else []
                )
            
            # Clear query log
            query_log.clear()
            
            # Perform search
            results, total = db.search_prompts(search_query="Prompt")
            
            # Check keyword fetch pattern - currently uses individual queries per prompt
            keyword_queries = [q for q in query_log if "PromptKeywordsTable" in q and "JOIN PromptKeywordLinks" in q]
            
            # Current implementation fetches keywords individually for each prompt
            # This documents the current behavior (N+1 query pattern)
            assert len(keyword_queries) == len(results)  # One query per result
            
            # Verify results have keywords attached
            for result in results:
                if result['id'] <= 5:  # First 5 prompts have keywords
                    assert len(result['keywords']) == 2
                else:
                    assert len(result['keywords']) == 0
            
            # Cleanup
            Path(tmp.name).unlink(missing_ok=True)


class TestPaginationConsistency:
    """Test that pagination maintains consistency across queries."""
    
    def test_pagination_result_consistency(self):
        """Test that paginated results are consistent."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db = MediaDatabase(tmp.name, client_id="test_client")
            
            # MediaDatabase creates schema automatically
            # Insert ordered data using execute_query
            for i in range(100):
                db.execute_query("""
                    INSERT INTO Media (uuid, title, content, type, content_hash, last_modified, version, client_id)
                    VALUES (?, ?, ?, ?, ?, datetime('now'), 1, ?)
                """, (
                    f"uuid_{i:03d}", 
                    f"Title {i:03d}", 
                    f"Content {i}",
                    "document",
                    f"hash_{i}",
                    "test_client"
                ), commit=True)
            
            # Get results in pages
            page_size = 10
            all_ids = []
            
            for offset in range(0, 100, page_size):
                results = db.get_all_active_media_for_embedding(limit=page_size, offset=offset)
                all_ids.extend([r['id'] for r in results])
            
            # Verify we got all items without duplicates
            assert len(all_ids) == 100
            assert len(set(all_ids)) == 100  # No duplicates
            assert sorted(all_ids) == list(range(1, 101))  # All IDs present
            
            # Cleanup
            Path(tmp.name).unlink(missing_ok=True)