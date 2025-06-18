# RAG_Indexing_DB.py
# Description: Database module for tracking RAG indexing state
#
"""
RAG_Indexing_DB.py
------------------

A SQLite-based module for tracking the state of RAG indexing operations.
This module provides functionality to:
- Track which items have been indexed and when
- Support incremental indexing by tracking last_modified timestamps
- Manage indexing state across different content types (media, conversations, notes)

The module uses a simple schema that tracks:
- Item ID and type
- Last indexed timestamp
- Last known modification timestamp
- Indexing status and metadata
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from loguru import logger

class RAGIndexingDB:
    """
    Manages SQLite database for tracking RAG indexing state.
    
    This class provides methods to track which items have been indexed,
    when they were indexed, and their last known modification times.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize the RAG indexing database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
        
    def _initialize_schema(self):
        """Initialize the database schema."""
        schema = """
        CREATE TABLE IF NOT EXISTS indexed_items (
            item_id TEXT NOT NULL,
            item_type TEXT NOT NULL,
            last_indexed DATETIME NOT NULL,
            last_modified DATETIME NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            metadata TEXT,
            PRIMARY KEY (item_id, item_type)
        );
        
        CREATE INDEX IF NOT EXISTS idx_indexed_items_type 
        ON indexed_items(item_type);
        
        CREATE INDEX IF NOT EXISTS idx_indexed_items_modified 
        ON indexed_items(last_modified);
        
        CREATE INDEX IF NOT EXISTS idx_indexed_items_indexed 
        ON indexed_items(last_indexed);
        
        -- Table for tracking collection states
        CREATE TABLE IF NOT EXISTS collection_state (
            collection_name TEXT PRIMARY KEY,
            last_full_index DATETIME,
            total_items INTEGER DEFAULT 0,
            indexed_items INTEGER DEFAULT 0,
            metadata TEXT
        );
        """
        
        with self._get_connection() as conn:
            conn.executescript(schema)
            conn.commit()
            
    def mark_item_indexed(
        self, 
        item_id: str, 
        item_type: str, 
        last_modified: datetime,
        chunk_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Mark an item as indexed.
        
        Args:
            item_id: Unique identifier for the item
            item_type: Type of item (media, conversation, note)
            last_modified: Last modification timestamp of the item
            chunk_count: Number of chunks created for this item
            metadata: Optional metadata about the indexing
        """
        query = """
        INSERT OR REPLACE INTO indexed_items 
        (item_id, item_type, last_indexed, last_modified, chunk_count, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        now = datetime.now(timezone.utc)
        metadata_json = json.dumps(metadata) if metadata else None
        
        with self._get_connection() as conn:
            conn.execute(
                query,
                (item_id, item_type, now, last_modified, chunk_count, metadata_json)
            )
            conn.commit()
            
    def get_items_to_index(
        self, 
        item_type: str,
        modified_since: Optional[datetime] = None
    ) -> List[str]:
        """
        Get list of item IDs that need indexing.
        
        This method is used by the indexing service to determine which items
        are new or have been modified since last indexing.
        
        Args:
            item_type: Type of items to check
            modified_since: Only return items modified after this timestamp
            
        Returns:
            List of item IDs that need indexing
        """
        # This will be implemented by the indexing service
        # by comparing with the source database
        return []
        
    def get_indexed_item_info(
        self, 
        item_id: str, 
        item_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get indexing information for a specific item.
        
        Args:
            item_id: Item identifier
            item_type: Type of item
            
        Returns:
            Dictionary with indexing information or None if not indexed
        """
        query = """
        SELECT * FROM indexed_items 
        WHERE item_id = ? AND item_type = ?
        """
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, (item_id, item_type))
            row = cursor.fetchone()
            
            if row:
                return {
                    'item_id': row['item_id'],
                    'item_type': row['item_type'],
                    'last_indexed': row['last_indexed'],
                    'last_modified': row['last_modified'],
                    'chunk_count': row['chunk_count'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else None
                }
            return None
            
    def get_indexed_items_by_type(
        self, 
        item_type: str
    ) -> Dict[str, datetime]:
        """
        Get all indexed items of a specific type with their last modified times.
        
        Args:
            item_type: Type of items to retrieve
            
        Returns:
            Dictionary mapping item_id to last_modified timestamp
        """
        query = """
        SELECT item_id, last_modified FROM indexed_items 
        WHERE item_type = ?
        """
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, (item_type,))
            return {
                row['item_id']: datetime.fromisoformat(row['last_modified'])
                for row in cursor
            }
            
    def remove_indexed_item(self, item_id: str, item_type: str):
        """
        Remove an item from the indexed items tracking.
        
        Args:
            item_id: Item identifier
            item_type: Type of item
        """
        query = "DELETE FROM indexed_items WHERE item_id = ? AND item_type = ?"
        
        with self._get_connection() as conn:
            conn.execute(query, (item_id, item_type))
            conn.commit()
            
    def update_collection_state(
        self,
        collection_name: str,
        total_items: int,
        indexed_items: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update the state of a collection.
        
        Args:
            collection_name: Name of the collection (e.g., 'media_chunks')
            total_items: Total number of items in the source
            indexed_items: Number of items indexed
            metadata: Optional metadata about the collection
        """
        query = """
        INSERT OR REPLACE INTO collection_state 
        (collection_name, last_full_index, total_items, indexed_items, metadata)
        VALUES (?, ?, ?, ?, ?)
        """
        
        now = datetime.now(timezone.utc)
        metadata_json = json.dumps(metadata) if metadata else None
        
        with self._get_connection() as conn:
            conn.execute(
                query,
                (collection_name, now, total_items, indexed_items, metadata_json)
            )
            conn.commit()
            
    def get_collection_state(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection state or None
        """
        query = "SELECT * FROM collection_state WHERE collection_name = ?"
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, (collection_name,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'collection_name': row['collection_name'],
                    'last_full_index': row['last_full_index'],
                    'total_items': row['total_items'],
                    'indexed_items': row['indexed_items'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else None
                }
            return None
            
    def get_indexing_stats(self) -> Dict[str, Any]:
        """
        Get overall indexing statistics.
        
        Returns:
            Dictionary with indexing statistics
        """
        stats = {
            'total_indexed': 0,
            'by_type': {},
            'collections': {}
        }
        
        with self._get_connection() as conn:
            # Get counts by type
            cursor = conn.execute("""
                SELECT item_type, COUNT(*) as count 
                FROM indexed_items 
                GROUP BY item_type
            """)
            
            for row in cursor:
                stats['by_type'][row['item_type']] = row['count']
                stats['total_indexed'] += row['count']
                
            # Get collection states
            cursor = conn.execute("SELECT * FROM collection_state")
            for row in cursor:
                stats['collections'][row['collection_name']] = {
                    'last_full_index': row['last_full_index'],
                    'total_items': row['total_items'],
                    'indexed_items': row['indexed_items']
                }
                
        return stats
        
    def clear_all(self):
        """Clear all indexing tracking data."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM indexed_items")
            conn.execute("DELETE FROM collection_state")
            conn.commit()
            logger.warning("Cleared all RAG indexing tracking data")