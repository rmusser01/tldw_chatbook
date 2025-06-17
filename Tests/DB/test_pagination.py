"""
Unit tests for pagination functionality in database functions.
"""

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
        
        # Create schema
        with db.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS Media (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT UNIQUE NOT NULL,
                    title TEXT,
                    content TEXT,
                    type TEXT,
                    url TEXT,
                    author TEXT,
                    ingestion_date TEXT,
                    last_modified TEXT,
                    deleted INTEGER DEFAULT 0,
                    is_trash INTEGER DEFAULT 0,
                    vector_processing INTEGER DEFAULT 0,
                    version INTEGER DEFAULT 1,
                    client_id TEXT
                );
                
                CREATE TABLE IF NOT EXISTS Keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT UNIQUE NOT NULL,
                    keyword TEXT UNIQUE COLLATE NOCASE,
                    deleted INTEGER DEFAULT 0,
                    last_modified TEXT,
                    version INTEGER DEFAULT 1,
                    client_id TEXT
                );
            """)
            
            # Insert test data
            for i in range(250):  # Create 250 test media items
                conn.execute("""
                    INSERT INTO Media (uuid, title, content, type, vector_processing, deleted, is_trash)
                    VALUES (?, ?, ?, ?, ?, 0, 0)
                """, (
                    f"uuid_{i}",
                    f"Title {i}",
                    f"Content for item {i}",
                    "document",
                    0 if i < 150 else 1  # First 150 need processing
                ))
            
            # Insert test keywords
            for i in range(500):  # Create 500 test keywords
                conn.execute("""
                    INSERT INTO Keywords (uuid, keyword, deleted)
                    VALUES (?, ?, 0)
                """, (f"kw_uuid_{i}", f"keyword_{i}"))
            
            conn.commit()
        
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
    
    def test_fetch_all_keywords_pagination(self, mock_media_db):
        """Test pagination in fetch_all_keywords."""
        # Test default behavior (no limit)
        results = mock_media_db.fetch_all_keywords()
        assert len(results) == 500  # All keywords
        
        # Test with limit
        results = mock_media_db.fetch_all_keywords(limit=50)
        assert len(results) == 50
        
        # Test with limit and offset
        results = mock_media_db.fetch_all_keywords(limit=50, offset=100)
        assert len(results) == 50
        assert results[0] == "keyword_100"
        
        # Test edge case: offset at boundary
        results = mock_media_db.fetch_all_keywords(limit=50, offset=480)
        assert len(results) == 20  # Only 20 items left
    
    def test_get_unprocessed_media_pagination(self, mock_media_db):
        """Test pagination in get_unprocessed_media."""
        # Test default behavior (limit=100)
        results = get_unprocessed_media(mock_media_db)
        assert len(results) == 100  # Default limit
        
        # Test custom limit
        results = get_unprocessed_media(mock_media_db, limit=50)
        assert len(results) == 50
        
        # Test with offset
        results = get_unprocessed_media(mock_media_db, limit=50, offset=50)
        assert len(results) == 50
        assert results[0]['id'] == 51
        
        # Test retrieving all with high limit
        results = get_unprocessed_media(mock_media_db, limit=200)
        assert len(results) == 150  # Total unprocessed items
    
    def test_get_all_content_from_database_pagination(self, mock_media_db):
        """Test pagination in get_all_content_from_database."""
        # Test default behavior (limit=100)
        results = get_all_content_from_database(mock_media_db)
        assert len(results) == 100  # Default limit
        
        # Test custom limit
        results = get_all_content_from_database(mock_media_db, limit=25)
        assert len(results) == 25
        
        # Test with offset
        results = get_all_content_from_database(mock_media_db, limit=25, offset=25)
        assert len(results) == 25
        
        # Verify ordering (should be by last_modified DESC)
        # Since we didn't set last_modified, it should fall back to ID ordering
        assert results[0]['id'] == 226  # IDs 251-25 = 226
    
    def test_pagination_with_none_limit(self, mock_media_db):
        """Test that None limit returns all results."""
        # fetch_all_keywords with None limit
        results = mock_media_db.fetch_all_keywords(limit=None)
        assert len(results) == 500  # All keywords
        
        # get_all_active_media_for_embedding with None limit
        results = mock_media_db.get_all_active_media_for_embedding(limit=None)
        assert len(results) == 250  # All media items


class TestPaginationEdgeCases:
    """Test edge cases in pagination."""
    
    def test_pagination_with_empty_database(self):
        """Test pagination when database is empty."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db = MediaDatabase(tmp.name, client_id="test_client")
            
            # Create empty schema
            with db.get_connection() as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS Media (
                        id INTEGER PRIMARY KEY,
                        uuid TEXT,
                        title TEXT,
                        content TEXT,
                        type TEXT,
                        deleted INTEGER DEFAULT 0,
                        is_trash INTEGER DEFAULT 0,
                        vector_processing INTEGER DEFAULT 0
                    );
                    CREATE TABLE IF NOT EXISTS Keywords (
                        id INTEGER PRIMARY KEY,
                        uuid TEXT,
                        keyword TEXT,
                        deleted INTEGER DEFAULT 0
                    );
                """)
            
            # Test with empty database
            assert db.get_all_active_media_for_embedding(limit=10) == []
            assert db.fetch_all_keywords(limit=10) == []
            assert get_unprocessed_media(db, limit=10) == []
            assert get_all_content_from_database(db, limit=10) == []
            
            # Cleanup
            Path(tmp.name).unlink(missing_ok=True)
    
    def test_pagination_with_negative_values(self):
        """Test pagination with negative limit/offset values."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db = MediaDatabase(tmp.name, client_id="test_client")
            
            # Create schema and add one item
            with db.get_connection() as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS Media (
                        id INTEGER PRIMARY KEY,
                        uuid TEXT,
                        title TEXT,
                        content TEXT,
                        deleted INTEGER DEFAULT 0,
                        is_trash INTEGER DEFAULT 0
                    );
                """)
                conn.execute("""
                    INSERT INTO Media (uuid, title, content)
                    VALUES ('test_uuid', 'Test Title', 'Test Content')
                """)
            
            # SQLite treats negative LIMIT as no limit
            # SQLite treats negative OFFSET as 0
            results = db.get_all_active_media_for_embedding(limit=-1, offset=-1)
            assert len(results) == 1
            
            # Cleanup
            Path(tmp.name).unlink(missing_ok=True)


class TestBatchQueryOptimization:
    """Test the N+1 query optimization for search results."""
    
    def test_search_prompts_batch_keyword_fetch(self):
        """Test that search_prompts fetches keywords in batch."""
        from tldw_chatbook.DB.Prompts_DB import PromptsDB
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db = PromptsDB(tmp.name, client_id="test_client")
            
            # Mock the database to track queries
            original_execute = db.execute_query
            query_log = []
            
            def mock_execute(query, params=None):
                query_log.append(query)
                return original_execute(query, params)
            
            db.execute_query = mock_execute
            
            # Create schema
            db._init_schema()
            
            # Add test data
            with db.get_connection() as conn:
                # Add prompts
                for i in range(10):
                    conn.execute("""
                        INSERT INTO Prompts (uuid, name, system_prompt, user_prompt)
                        VALUES (?, ?, ?, ?)
                    """, (f"prompt_{i}", f"Prompt {i}", "System", "User"))
                
                # Add keywords
                for i in range(5):
                    conn.execute("""
                        INSERT INTO Keywords (uuid, keyword)
                        VALUES (?, ?)
                    """, (f"kw_{i}", f"keyword_{i}"))
                
                # Link some prompts to keywords
                for prompt_id in range(1, 6):
                    for keyword_id in range(1, 3):
                        conn.execute("""
                            INSERT INTO PromptKeywordLinks (prompt_id, keyword_id)
                            VALUES (?, ?)
                        """, (prompt_id, keyword_id))
            
            # Clear query log
            query_log.clear()
            
            # Perform search
            results, total = db.search_prompts(search_query="Prompt")
            
            # Check that keywords were fetched in one batch query
            keyword_queries = [q for q in query_log if "PromptKeywordLinks" in q and "JOIN Keywords" in q]
            
            # Should be exactly one batch query for keywords, not one per result
            assert len(keyword_queries) == 1
            assert "IN (" in keyword_queries[0]  # Using IN clause for batch fetch
            
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
            
            # Create schema and data
            with db.get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS Media (
                        id INTEGER PRIMARY KEY,
                        uuid TEXT UNIQUE,
                        title TEXT,
                        content TEXT,
                        deleted INTEGER DEFAULT 0,
                        is_trash INTEGER DEFAULT 0
                    );
                """)
                
                # Insert ordered data
                for i in range(100):
                    conn.execute("""
                        INSERT INTO Media (uuid, title, content)
                        VALUES (?, ?, ?)
                    """, (f"uuid_{i:03d}", f"Title {i:03d}", f"Content {i}"))
            
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