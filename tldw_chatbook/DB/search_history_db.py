# search_history_db.py
# Description: Database module for RAG search history persistence
#
"""
search_history_db.py
-------------------

A SQLite-based module for persisting RAG search history and analytics.
This module provides functionality to:
- Store search queries and their results
- Track search performance metrics
- Maintain search history with metadata
- Provide search analytics and usage patterns
- Support search result caching and retrieval

The module is designed for single-user applications and focuses on
providing insights into search behavior and result effectiveness.
"""

import sqlite3
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from loguru import logger
from ..Metrics.metrics_logger import log_counter, log_histogram

class SearchHistoryDB:
    """
    Manages SQLite database for RAG search history and analytics.
    
    This class provides methods to store search queries, results, and
    analytics data for the RAG system.
    """
    
    def __init__(self, db_path: Union[str, Path], client_id: str = "default"):
        """
        Initialize the search history database.
        
        Args:
            db_path: Path to the SQLite database file or ':memory:'
            client_id: Client identifier (for future multi-client support)
        """
        # Handle path types consistently
        if isinstance(db_path, Path):
            self.is_memory_db = False
            self.db_path = db_path.resolve()
        else:
            self.is_memory_db = (db_path == ':memory:')
            self.db_path = Path(db_path).resolve() if not self.is_memory_db else Path(":memory:")
        
        self.db_path_str = str(self.db_path) if not self.is_memory_db else ':memory:'
        self.client_id = client_id
        
        # Create directory if needed for file-based DB
        if not self.is_memory_db:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._initialize_schema()
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path_str)
        conn.row_factory = sqlite3.Row
        return conn
        
    def _initialize_schema(self):
        """Initialize the database schema."""
        schema = """
        -- Main search history table
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            search_type TEXT NOT NULL, -- 'plain', 'full', 'hybrid'
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            execution_time_ms INTEGER,
            result_count INTEGER DEFAULT 0,
            success BOOLEAN NOT NULL DEFAULT 1,
            error_message TEXT,
            search_params TEXT, -- JSON blob with search parameters
            user_session TEXT   -- Optional session identifier
        );
        
        -- Search results table (denormalized for performance)
        CREATE TABLE IF NOT EXISTS search_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_id INTEGER NOT NULL REFERENCES search_history(id) ON DELETE CASCADE,
            result_index INTEGER NOT NULL, -- Position in result list
            title TEXT,
            content TEXT,
            source TEXT, -- 'media', 'conversation', 'note'
            source_id TEXT,
            score REAL,
            metadata TEXT -- JSON blob with result metadata
        );
        
        -- Search analytics table for performance tracking
        CREATE TABLE IF NOT EXISTS search_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL, -- YYYY-MM-DD format
            total_searches INTEGER DEFAULT 0,
            avg_execution_time_ms REAL DEFAULT 0,
            avg_result_count REAL DEFAULT 0,
            success_rate REAL DEFAULT 0,
            popular_queries TEXT, -- JSON array of popular queries
            search_type_distribution TEXT -- JSON object with type counts
        );
        
        -- User feedback table for result quality
        CREATE TABLE IF NOT EXISTS result_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_id INTEGER NOT NULL REFERENCES search_history(id) ON DELETE CASCADE,
            result_id INTEGER NOT NULL REFERENCES search_results(id) ON DELETE CASCADE,
            rating INTEGER CHECK(rating BETWEEN 1 AND 5), -- 1-5 star rating
            helpful BOOLEAN, -- Simple helpful/not helpful
            clicked BOOLEAN DEFAULT 0, -- Whether user clicked/expanded result
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            comments TEXT
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_search_history_timestamp 
        ON search_history(timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_search_history_query 
        ON search_history(query);
        
        CREATE INDEX IF NOT EXISTS idx_search_history_type 
        ON search_history(search_type);
        
        CREATE INDEX IF NOT EXISTS idx_search_results_search_id 
        ON search_results(search_id);
        
        CREATE INDEX IF NOT EXISTS idx_search_results_source 
        ON search_results(source, source_id);
        
        CREATE INDEX IF NOT EXISTS idx_search_analytics_date 
        ON search_analytics(date);
        
        CREATE INDEX IF NOT EXISTS idx_result_feedback_search 
        ON result_feedback(search_id);
        """
        
        with self._get_connection() as conn:
            conn.executescript(schema)
            conn.commit()
            
    def record_search(
        self,
        query: str,
        search_type: str,
        results: List[Dict[str, Any]],
        execution_time_ms: int,
        search_params: Optional[Dict[str, Any]] = None,
        user_session: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> int:
        """
        Record a search query and its results.
        
        Args:
            query: The search query string
            search_type: Type of search ('plain', 'full', 'hybrid')
            results: List of search result dictionaries
            execution_time_ms: Search execution time in milliseconds
            search_params: Optional search parameters
            user_session: Optional session identifier
            error_message: Optional error message if search failed
            
        Returns:
            Search ID for the recorded search
        """
        start_time = time.time()
        
        try:
            with self._get_connection() as conn:
                # Insert search record
                search_query = """
                INSERT INTO search_history 
                (query, search_type, execution_time_ms, result_count, success, 
                 error_message, search_params, user_session)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                success = error_message is None
                search_params_json = json.dumps(search_params) if search_params else None
                
                cursor = conn.execute(
                    search_query,
                    (query, search_type, execution_time_ms, len(results), 
                     success, error_message, search_params_json, user_session)
                )
                
                search_id = cursor.lastrowid
                
                # Insert results if search was successful
                if success and results:
                    result_query = """
                    INSERT INTO search_results 
                    (search_id, result_index, title, content, source, source_id, score, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    result_data = []
                    for i, result in enumerate(results):
                        metadata_json = json.dumps(result.get('metadata', {}))
                        result_data.append((
                            search_id,
                            i,
                            result.get('title', ''),
                            result.get('content', ''),
                            result.get('source', ''),
                            str(result.get('source_id', '')),
                            result.get('score', 0.0),
                            metadata_json
                        ))
                    
                    conn.executemany(result_query, result_data)
                
                conn.commit()
                logger.debug(f"Recorded search: '{query}' with {len(results)} results")
                
                # Log success metrics
                duration = time.time() - start_time
                log_histogram("search_history_db_operation_duration", duration, labels={
                    "operation": "record_search",
                    "search_type": search_type,
                    "result_count": str(len(results))
                })
                log_counter("search_history_db_operation_count", labels={
                    "operation": "record_search",
                    "search_type": search_type,
                    "status": "success",
                    "has_error": "true" if error_message else "false"
                })
                # Also log the search execution time from the search itself
                log_histogram("search_history_db_search_execution_time", execution_time_ms / 1000.0, labels={
                    "search_type": search_type
                })
                
                return search_id
                
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram("search_history_db_operation_duration", duration, labels={
                "operation": "record_search",
                "search_type": search_type,
                "result_count": "0"
            })
            log_counter("search_history_db_operation_count", labels={
                "operation": "record_search",
                "search_type": search_type,
                "status": "error",
                "error_type": type(e).__name__
            })
            
            logger.error(f"Error recording search: {e}")
            return -1
            
    def get_search_history(
        self,
        limit: int = 100,
        search_type: Optional[str] = None,
        days_back: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get search history with optional filters.
        
        Args:
            limit: Maximum number of searches to return
            search_type: Filter by search type
            days_back: Only return searches from last N days
            
        Returns:
            List of search history records
        """
        start_time = time.time()
        
        try:
            query = """
            SELECT id, query, search_type, timestamp, execution_time_ms, 
                   result_count, success, error_message, search_params
            FROM search_history
            WHERE 1=1
            """
            params = []
            
            if search_type:
                query += " AND search_type = ?"
                params.append(search_type)
                
            if days_back:
                query += " AND timestamp >= datetime('now', '-{} days')".format(days_back)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            with self._get_connection() as conn:
                cursor = conn.execute(query, params)
                
                history = []
                for row in cursor:
                    search_params = json.loads(row['search_params']) if row['search_params'] else {}
                    
                    history.append({
                        'id': row['id'],
                        'query': row['query'],
                        'search_type': row['search_type'],
                        'timestamp': row['timestamp'],
                        'execution_time_ms': row['execution_time_ms'],
                        'result_count': row['result_count'],
                        'success': bool(row['success']),
                        'error_message': row['error_message'],
                        'search_params': search_params
                    })
                
                # Log success metrics
                duration = time.time() - start_time
                log_histogram("search_history_db_operation_duration", duration, labels={
                    "operation": "get_search_history",
                    "result_count": str(len(history))
                })
                log_counter("search_history_db_operation_count", labels={
                    "operation": "get_search_history",
                    "status": "success",
                    "result_count": str(len(history)),
                    "filtered_by": search_type or "none"
                })
                
                return history
                
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram("search_history_db_operation_duration", duration, labels={
                "operation": "get_search_history",
                "result_count": "0"
            })
            log_counter("search_history_db_operation_count", labels={
                "operation": "get_search_history",
                "status": "error",
                "error_type": type(e).__name__
            })
            
            logger.error(f"Error getting search history: {e}")
            return []
            
    def get_search_results(self, search_id: int) -> List[Dict[str, Any]]:
        """
        Get results for a specific search.
        
        Args:
            search_id: ID of the search
            
        Returns:
            List of search results
        """
        try:
            query = """
            SELECT result_index, title, content, source, source_id, score, metadata
            FROM search_results
            WHERE search_id = ?
            ORDER BY result_index
            """
            
            with self._get_connection() as conn:
                cursor = conn.execute(query, (search_id,))
                
                results = []
                for row in cursor:
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    
                    results.append({
                        'index': row['result_index'],
                        'title': row['title'],
                        'content': row['content'],
                        'source': row['source'],
                        'source_id': row['source_id'],
                        'score': row['score'],
                        'metadata': metadata
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting search results for search {search_id}: {e}")
            return []
            
    def record_result_feedback(
        self,
        search_id: int,
        result_index: int,
        rating: Optional[int] = None,
        helpful: Optional[bool] = None,
        clicked: bool = False,
        comments: Optional[str] = None
    ) -> bool:
        """
        Record user feedback for a search result.
        
        Args:
            search_id: ID of the search
            result_index: Index of the result in the search
            rating: Optional 1-5 star rating
            helpful: Optional helpful/not helpful feedback
            clicked: Whether the user clicked/expanded the result
            comments: Optional text comments
            
        Returns:
            True if feedback was recorded successfully
        """
        start_time = time.time()
        
        try:
            # First get the result_id
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT id FROM search_results WHERE search_id = ? AND result_index = ?",
                    (search_id, result_index)
                )
                result = cursor.fetchone()
                
                if not result:
                    logger.warning(f"No result found for search {search_id} index {result_index}")
                    return False
                    
                result_id = result['id']
                
                # Insert or update feedback
                query = """
                INSERT OR REPLACE INTO result_feedback 
                (search_id, result_id, rating, helpful, clicked, comments)
                VALUES (?, ?, ?, ?, ?, ?)
                """
                
                conn.execute(query, (search_id, result_id, rating, helpful, clicked, comments))
                conn.commit()
                
                logger.debug(f"Recorded feedback for search {search_id} result {result_index}")
                
                # Log success metrics
                duration = time.time() - start_time
                feedback_type = "rating" if rating else "helpful" if helpful is not None else "clicked"
                log_histogram("search_history_db_operation_duration", duration, labels={
                    "operation": "record_feedback",
                    "feedback_type": feedback_type
                })
                log_counter("search_history_db_operation_count", labels={
                    "operation": "record_feedback",
                    "feedback_type": feedback_type,
                    "status": "success"
                })
                
                return True
                
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram("search_history_db_operation_duration", duration, labels={
                "operation": "record_feedback",
                "feedback_type": "unknown"
            })
            log_counter("search_history_db_operation_count", labels={
                "operation": "record_feedback",
                "status": "error",
                "error_type": type(e).__name__
            })
            
            logger.error(f"Error recording result feedback: {e}")
            return False
            
    def get_popular_queries(self, limit: int = 10, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Get most popular search queries.
        
        Args:
            limit: Number of queries to return
            days_back: Look at queries from last N days
            
        Returns:
            List of popular queries with counts
        """
        try:
            query = """
            SELECT query, COUNT(*) as count, AVG(execution_time_ms) as avg_time,
                   AVG(result_count) as avg_results
            FROM search_history
            WHERE timestamp >= datetime('now', '-{} days')
              AND success = 1
            GROUP BY LOWER(query)
            ORDER BY count DESC
            LIMIT ?
            """.format(days_back)
            
            with self._get_connection() as conn:
                cursor = conn.execute(query, (limit,))
                
                popular = []
                for row in cursor:
                    popular.append({
                        'query': row['query'],
                        'count': row['count'],
                        'avg_execution_time_ms': round(row['avg_time'], 2),
                        'avg_result_count': round(row['avg_results'], 1)
                    })
                
                return popular
                
        except Exception as e:
            logger.error(f"Error getting popular queries: {e}")
            return []
            
    def get_search_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get search analytics and performance metrics.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with analytics data
        """
        try:
            with self._get_connection() as conn:
                # Basic statistics
                stats_query = """
                SELECT 
                    COUNT(*) as total_searches,
                    AVG(execution_time_ms) as avg_execution_time,
                    AVG(result_count) as avg_result_count,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    COUNT(DISTINCT query) as unique_queries
                FROM search_history
                WHERE timestamp >= datetime('now', '-{} days')
                """.format(days_back)
                
                cursor = conn.execute(stats_query)
                stats = cursor.fetchone()
                
                # Search type distribution
                type_query = """
                SELECT search_type, COUNT(*) as count
                FROM search_history
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY search_type
                """.format(days_back)
                
                cursor = conn.execute(type_query)
                search_types = {row['search_type']: row['count'] for row in cursor}
                
                # Daily search counts
                daily_query = """
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM search_history
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY DATE(timestamp)
                ORDER BY date
                """.format(days_back)
                
                cursor = conn.execute(daily_query)
                daily_counts = [{'date': row['date'], 'count': row['count']} for row in cursor]
                
                # Top error messages
                error_query = """
                SELECT error_message, COUNT(*) as count
                FROM search_history
                WHERE timestamp >= datetime('now', '-{} days')
                  AND success = 0
                  AND error_message IS NOT NULL
                GROUP BY error_message
                ORDER BY count DESC
                LIMIT 5
                """.format(days_back)
                
                cursor = conn.execute(error_query)
                top_errors = [{'error': row['error_message'], 'count': row['count']} for row in cursor]
                
                return {
                    'period_days': days_back,
                    'total_searches': stats['total_searches'] or 0,
                    'unique_queries': stats['unique_queries'] or 0,
                    'avg_execution_time_ms': round(stats['avg_execution_time'] or 0, 2),
                    'avg_result_count': round(stats['avg_result_count'] or 0, 1),
                    'success_rate': round(stats['success_rate'] or 0, 2),
                    'search_type_distribution': search_types,
                    'daily_search_counts': daily_counts,
                    'top_errors': top_errors,
                    'popular_queries': self.get_popular_queries(limit=5, days_back=days_back)
                }
                
        except Exception as e:
            logger.error(f"Error getting search analytics: {e}")
            return {
                'period_days': days_back,
                'total_searches': 0,
                'unique_queries': 0,
                'avg_execution_time_ms': 0,
                'avg_result_count': 0,
                'success_rate': 0,
                'search_type_distribution': {},
                'daily_search_counts': [],
                'top_errors': [],
                'popular_queries': []
            }
            
    def export_search_data(self, output_path: Path, days_back: int = 30) -> bool:
        """
        Export search data to JSON file.
        
        Args:
            output_path: Path to save the JSON file
            days_back: Number of days of data to export
            
        Returns:
            True if export successful
        """
        try:
            # Get all data
            history = self.get_search_history(limit=10000, days_back=days_back)
            analytics = self.get_search_analytics(days_back=days_back)
            
            # Include results for each search
            for search in history:
                search['results'] = self.get_search_results(search['id'])
            
            export_data = {
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'period_days': days_back,
                'analytics': analytics,
                'search_history': history
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Exported search data to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting search data: {e}")
            return False
            
    def clear_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clear old search history data.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Number of search records deleted
        """
        start_time = time.time()
        
        try:
            with self._get_connection() as conn:
                # Delete old search records (cascades to results and feedback)
                cursor = conn.execute(
                    "DELETE FROM search_history WHERE timestamp < datetime('now', '-{} days')".format(days_to_keep)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Deleted {deleted_count} old search records (older than {days_to_keep} days)")
                
                # Log success metrics
                duration = time.time() - start_time
                log_histogram("search_history_db_operation_duration", duration, labels={
                    "operation": "clear_old_data",
                    "days_to_keep": str(days_to_keep)
                })
                log_counter("search_history_db_operation_count", labels={
                    "operation": "clear_old_data",
                    "status": "success",
                    "records_deleted": str(deleted_count)
                })
                
                return deleted_count
                
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram("search_history_db_operation_duration", duration, labels={
                "operation": "clear_old_data",
                "days_to_keep": str(days_to_keep)
            })
            log_counter("search_history_db_operation_count", labels={
                "operation": "clear_old_data",
                "status": "error",
                "error_type": type(e).__name__
            })
            
            logger.error(f"Error clearing old search data: {e}")
            return 0