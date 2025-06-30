# test_search_history_db.py
# Description: Integration tests for RAG search history database
#
"""
test_search_history_db.py
-------------------------

Integration tests for the search history database that persists RAG search
queries, results, and analytics.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import time

from tldw_chatbook.DB.search_history_db import SearchHistoryDB


@pytest.mark.integration
class TestSearchHistoryDB:
    """Test cases for SearchHistoryDB."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = Path(tmp.name)
        
        db = SearchHistoryDB(db_path)
        yield db
        
        # Cleanup
        db_path.unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        return [
            {
                'title': 'Result 1',
                'content': 'This is the first result content',
                'source': 'media',
                'source_id': 'media_123',
                'score': 0.95,
                'metadata': {'author': 'Test Author', 'date': '2024-01-01'}
            },
            {
                'title': 'Result 2', 
                'content': 'This is the second result content',
                'source': 'conversation',
                'source_id': 'conv_456',
                'score': 0.87,
                'metadata': {'participants': ['user1', 'user2']}
            },
            {
                'title': 'Result 3',
                'content': 'This is the third result content',
                'source': 'note',
                'source_id': 'note_789',
                'score': 0.75,
                'metadata': {'tags': ['important', 'todo']}
            }
        ]
    
    def test_initialization(self, temp_db):
        """Test database initialization and schema creation."""
        # Check that tables exist
        with temp_db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            
        assert 'search_history' in tables
        assert 'search_results' in tables
        assert 'search_analytics' in tables
        assert 'result_feedback' in tables
    
    def test_record_successful_search(self, temp_db, sample_results):
        """Test recording a successful search."""
        search_id = temp_db.record_search(
            query="test query",
            search_type="full",
            results=sample_results,
            execution_time_ms=150,
            search_params={'top_k': 10, 'enable_rerank': True},
            user_session="test_session"
        )
        
        assert search_id > 0
        
        # Verify search was recorded
        history = temp_db.get_search_history(limit=1)
        assert len(history) == 1
        assert history[0]['query'] == "test query"
        assert history[0]['search_type'] == "full"
        assert history[0]['result_count'] == 3
        assert history[0]['success'] is True
        assert history[0]['execution_time_ms'] == 150
    
    def test_record_failed_search(self, temp_db):
        """Test recording a failed search."""
        search_id = temp_db.record_search(
            query="failing query",
            search_type="plain",
            results=[],
            execution_time_ms=50,
            error_message="Connection timeout"
        )
        
        assert search_id > 0
        
        # Verify failed search was recorded
        history = temp_db.get_search_history(limit=1)
        assert len(history) == 1
        assert history[0]['query'] == "failing query"
        assert history[0]['success'] is False
        assert history[0]['error_message'] == "Connection timeout"
        assert history[0]['result_count'] == 0
    
    def test_get_search_results(self, temp_db, sample_results):
        """Test retrieving results for a specific search."""
        # Record search
        search_id = temp_db.record_search(
            query="test query",
            search_type="full",
            results=sample_results,
            execution_time_ms=150
        )
        
        # Get results
        results = temp_db.get_search_results(search_id)
        
        assert len(results) == 3
        assert results[0]['title'] == 'Result 1'
        assert results[0]['source'] == 'media'
        assert results[0]['score'] == 0.95
        assert results[1]['title'] == 'Result 2'
        assert results[2]['title'] == 'Result 3'
        
        # Verify metadata is preserved
        assert results[0]['metadata']['author'] == 'Test Author'
        assert results[1]['metadata']['participants'] == ['user1', 'user2']
        assert results[2]['metadata']['tags'] == ['important', 'todo']
    
    def test_get_search_history_with_filters(self, temp_db, sample_results):
        """Test getting search history with various filters."""
        # Record multiple searches
        temp_db.record_search("query 1", "plain", [], 100)
        time.sleep(0.1)
        temp_db.record_search("query 2", "full", sample_results, 200)
        time.sleep(0.1)
        temp_db.record_search("query 3", "hybrid", sample_results[:2], 300)
        time.sleep(0.1)
        temp_db.record_search("query 4", "full", sample_results[:1], 150)
        
        # Test limit
        history = temp_db.get_search_history(limit=2)
        assert len(history) == 2
        assert history[0]['query'] == "query 4"  # Most recent first
        
        # Test search type filter
        full_searches = temp_db.get_search_history(search_type="full")
        assert len(full_searches) == 2
        assert all(s['search_type'] == "full" for s in full_searches)
        
        # Test days back filter (should get all since they're recent)
        recent = temp_db.get_search_history(days_back=1)
        assert len(recent) == 4
    
    def test_record_result_feedback(self, temp_db, sample_results):
        """Test recording user feedback for results."""
        # Record search
        search_id = temp_db.record_search(
            query="test query",
            search_type="full",
            results=sample_results,
            execution_time_ms=150
        )
        
        # Record feedback for first result
        success = temp_db.record_result_feedback(
            search_id=search_id,
            result_index=0,
            rating=5,
            helpful=True,
            clicked=True,
            comments="Very relevant result"
        )
        
        assert success is True
        
        # Verify feedback was recorded
        with temp_db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM result_feedback WHERE search_id = ?",
                (search_id,)
            )
            feedback = cursor.fetchone()
            
        assert feedback is not None
        assert feedback['rating'] == 5
        assert feedback['helpful'] == 1
        assert feedback['clicked'] == 1
        assert feedback['comments'] == "Very relevant result"
    
    def test_get_popular_queries(self, temp_db, sample_results):
        """Test getting popular queries."""
        # Record searches with different frequencies
        for _ in range(5):
            temp_db.record_search("popular query", "full", sample_results, 100)
        
        for _ in range(3):
            temp_db.record_search("medium query", "plain", [], 50)
            
        temp_db.record_search("rare query", "hybrid", sample_results[:1], 200)
        
        # Get popular queries
        popular = temp_db.get_popular_queries(limit=3)
        
        assert len(popular) <= 3
        assert popular[0]['query'] == "popular query"
        assert popular[0]['count'] == 5
        assert popular[0]['avg_result_count'] == 3.0
        
        if len(popular) > 1:
            assert popular[1]['query'] == "medium query"
            assert popular[1]['count'] == 3
    
    def test_get_search_analytics(self, temp_db, sample_results):
        """Test getting search analytics."""
        # Record various searches
        # Successful searches
        for i in range(10):
            temp_db.record_search(
                query=f"query {i}",
                search_type="full" if i % 2 == 0 else "plain",
                results=sample_results if i % 3 != 0 else [],
                execution_time_ms=100 + i * 10
            )
        
        # Failed searches
        for i in range(2):
            temp_db.record_search(
                query=f"failed {i}",
                search_type="hybrid",
                results=[],
                execution_time_ms=50,
                error_message=f"Error type {i}"
            )
        
        # Get analytics
        analytics = temp_db.get_search_analytics(days_back=30)
        
        assert analytics['total_searches'] == 12
        assert analytics['unique_queries'] == 12
        assert analytics['success_rate'] > 80  # 10 successful out of 12
        assert analytics['avg_execution_time_ms'] > 0
        assert analytics['avg_result_count'] >= 0
        
        # Check search type distribution
        assert 'full' in analytics['search_type_distribution']
        assert 'plain' in analytics['search_type_distribution']
        assert 'hybrid' in analytics['search_type_distribution']
        
        # Check daily counts
        assert len(analytics['daily_search_counts']) > 0
        
        # Check top errors
        assert len(analytics['top_errors']) == 2
        
        # Check popular queries
        assert len(analytics['popular_queries']) > 0
    
    def test_export_search_data(self, temp_db, sample_results):
        """Test exporting search data to JSON."""
        # Record some searches
        search_ids = []
        for i in range(3):
            search_id = temp_db.record_search(
                query=f"query {i}",
                search_type="full",
                results=sample_results[:i+1],
                execution_time_ms=100 + i * 50
            )
            search_ids.append(search_id)
        
        # Export data
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            export_path = Path(tmp.name)
        
        success = temp_db.export_search_data(export_path, days_back=30)
        assert success is True
        
        # Verify export file
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert 'export_timestamp' in exported_data
        assert 'period_days' in exported_data
        assert exported_data['period_days'] == 30
        assert 'analytics' in exported_data
        assert 'search_history' in exported_data
        
        # Verify search history includes results
        assert len(exported_data['search_history']) == 3
        assert exported_data['search_history'][0]['results'] is not None
        
        # Cleanup
        export_path.unlink()
    
    def test_clear_old_data(self, temp_db, sample_results):
        """Test clearing old search data."""
        # Record old and new searches
        now = datetime.now(timezone.utc)
        
        # Manually insert old searches
        with temp_db._get_connection() as conn:
            old_time = (now - timedelta(days=100)).isoformat()
            conn.execute(
                """INSERT INTO search_history 
                   (query, search_type, timestamp, execution_time_ms, result_count, success)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("old query 1", "plain", old_time, 100, 0, 1)
            )
            conn.execute(
                """INSERT INTO search_history 
                   (query, search_type, timestamp, execution_time_ms, result_count, success)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("old query 2", "full", old_time, 200, 5, 1)
            )
            conn.commit()
        
        # Record recent searches
        temp_db.record_search("recent query", "hybrid", sample_results, 150)
        
        # Clear old data
        deleted_count = temp_db.clear_old_data(days_to_keep=90)
        
        assert deleted_count == 2
        
        # Verify only recent search remains
        history = temp_db.get_search_history(limit=10)
        assert len(history) == 1
        assert history[0]['query'] == "recent query"
    
    def test_concurrent_recording(self, temp_db, sample_results):
        """Test concurrent search recording."""
        import threading
        
        errors = []
        search_ids = []
        
        def record_searches(thread_id):
            """Record searches from a thread."""
            try:
                for i in range(5):
                    search_id = temp_db.record_search(
                        query=f"thread_{thread_id}_query_{i}",
                        search_type="full",
                        results=sample_results[:2],
                        execution_time_ms=100 + i
                    )
                    search_ids.append(search_id)
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=record_searches, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check no errors
        assert len(errors) == 0
        
        # Verify all searches were recorded
        history = temp_db.get_search_history(limit=20)
        assert len(history) == 15  # 3 threads * 5 searches
        
        # Verify search IDs are unique
        assert len(set(search_ids)) == 15
    
    def test_search_result_ordering(self, temp_db):
        """Test that search results maintain their order."""
        results = []
        for i in range(10):
            results.append({
                'title': f'Result {i}',
                'content': f'Content {i}',
                'source': 'media',
                'source_id': f'id_{i}',
                'score': 1.0 - (i * 0.1),
                'metadata': {'index': i}
            })
        
        # Record search
        search_id = temp_db.record_search(
            query="ordering test",
            search_type="full",
            results=results,
            execution_time_ms=100
        )
        
        # Retrieve results
        retrieved = temp_db.get_search_results(search_id)
        
        # Verify order is preserved
        assert len(retrieved) == 10
        for i in range(10):
            assert retrieved[i]['index'] == i
            assert retrieved[i]['title'] == f'Result {i}'
            assert retrieved[i]['metadata']['index'] == i
    
    def test_empty_results_handling(self, temp_db):
        """Test handling searches with no results."""
        search_id = temp_db.record_search(
            query="no results query",
            search_type="plain",
            results=[],
            execution_time_ms=50
        )
        
        assert search_id > 0
        
        # Verify search was recorded
        history = temp_db.get_search_history(limit=1)
        assert history[0]['result_count'] == 0
        
        # Verify no results are returned
        results = temp_db.get_search_results(search_id)
        assert len(results) == 0
    
    def test_large_metadata_handling(self, temp_db):
        """Test handling results with large metadata."""
        large_metadata = {
            'large_field': 'x' * 10000,  # 10KB of data
            'nested': {
                'level1': {
                    'level2': {
                        'data': list(range(1000))
                    }
                }
            }
        }
        
        results = [{
            'title': 'Large metadata result',
            'content': 'Content',
            'source': 'media',
            'source_id': '123',
            'score': 0.9,
            'metadata': large_metadata
        }]
        
        # Should handle large metadata without issues
        search_id = temp_db.record_search(
            query="large metadata test",
            search_type="full",
            results=results,
            execution_time_ms=100
        )
        
        # Retrieve and verify
        retrieved = temp_db.get_search_results(search_id)
        assert len(retrieved) == 1
        assert retrieved[0]['metadata']['large_field'] == 'x' * 10000
        assert len(retrieved[0]['metadata']['nested']['level1']['level2']['data']) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])