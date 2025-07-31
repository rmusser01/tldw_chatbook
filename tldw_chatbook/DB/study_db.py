# study_db.py
# Description: DB Library for Study Management (Learning Paths, Flashcards, Mindmaps)
#
"""
study_db.py
-----------

A comprehensive SQLite-based library for managing study data including:
- Learning paths and topics
- Flashcards with spaced repetition
- Mindmaps and concept relationships
- Study progress tracking

This library provides:
- Schema management with versioning
- Thread-safe database connections using `threading.local`
- CRUD operations for study entities
- Anki-compatible card format support
- Spaced repetition algorithm (SM-2)
- Full-Text Search (FTS5) capabilities
"""

import sqlite3
import json
import uuid
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
from loguru import logger

from .sql_validation import validate_table_name, validate_column_name
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram

# Database Schema Version
SCHEMA_VERSION = 1

class StudyDBError(Exception):
    """Base exception for StudyDB related errors."""
    pass

class SchemaError(StudyDBError):
    """Exception for schema version mismatches or migration failures."""
    pass

class InputError(ValueError):
    """Custom exception for input validation errors."""
    pass

class ConflictError(StudyDBError):
    """Indicates a conflict due to concurrent modification or unique constraint violation."""
    
    def __init__(self, message="Conflict detected.", entity: Optional[str] = None, entity_id: Any = None):
        super().__init__(message)
        self.entity = entity
        self.entity_id = entity_id

class StudyDB:
    """Database manager for study data including learning paths, flashcards, and mindmaps."""
    
    def __init__(self, db_path: Union[str, Path] = "study.db", client_id: str = "default_client"):
        """
        Initialize the StudyDB with a database path and client ID.
        
        Args:
            db_path: Path to the SQLite database file
            client_id: Identifier for the client making changes (for audit trail)
        """
        # Handle special case for in-memory database
        if db_path == ":memory:":
            self.db_path = db_path
        else:
            self.db_path = Path(db_path)
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.client_id = client_id
        self._local = threading.local()
        
        # Initialize database schema
        self._init_schema()
        
        logger.info(f"StudyDB initialized with path: {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            # Convert Path to string if necessary, but keep :memory: as is
            db_path_str = self.db_path if isinstance(self.db_path, str) else str(self.db_path)
            conn = sqlite3.connect(db_path_str, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.connection = conn
        return self._local.connection
    
    def get_connection(self) -> sqlite3.Connection:
        """Public method to get thread-local database connection."""
        return self._get_connection()
    
    def _init_schema(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Create metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            
            # Check and update schema version
            cursor.execute("SELECT value FROM metadata WHERE key = 'schema_version'")
            row = cursor.fetchone()
            
            if row is None:
                # New database, create schema
                self._create_schema(cursor)
                cursor.execute("INSERT INTO metadata (key, value) VALUES ('schema_version', ?)", 
                            (str(SCHEMA_VERSION),))
            else:
                current_version = int(row[0])
                if current_version < SCHEMA_VERSION:
                    self._migrate_schema(cursor, current_version, SCHEMA_VERSION)
                elif current_version > SCHEMA_VERSION:
                    raise SchemaError(f"Database schema version {current_version} is newer than supported version {SCHEMA_VERSION}")
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Schema initialization failed: {e}")
            raise
    
    def _create_schema(self, cursor: sqlite3.Cursor):
        """Create the initial database schema."""
        
        # Learning paths table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_paths (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                last_modified_by TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                is_deleted INTEGER DEFAULT 0,
                path_order TEXT,  -- JSON array of topic IDs in order
                metadata TEXT     -- JSON for additional data
            )
        """)
        
        # Topics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id TEXT PRIMARY KEY,
                path_id TEXT REFERENCES learning_paths(id) ON DELETE CASCADE,
                parent_id TEXT REFERENCES topics(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                content TEXT,
                topic_order INTEGER DEFAULT 0,
                status TEXT DEFAULT 'not_started',  -- not_started, in_progress, completed
                progress REAL DEFAULT 0.0,  -- 0.0 to 1.0
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                last_modified_by TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                is_deleted INTEGER DEFAULT 0,
                metadata TEXT     -- JSON for additional data
            )
        """)
        
        # Flashcard decks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                last_modified_by TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                is_deleted INTEGER DEFAULT 0,
                card_count INTEGER DEFAULT 0,
                metadata TEXT     -- JSON for Anki export settings, etc.
            )
        """)
        
        # Flashcards table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flashcards (
                id TEXT PRIMARY KEY,
                deck_id TEXT REFERENCES decks(id) ON DELETE CASCADE,
                front TEXT NOT NULL,  -- Question/prompt
                back TEXT NOT NULL,   -- Answer
                tags TEXT,            -- Space-separated tags
                type TEXT DEFAULT 'basic',  -- basic, cloze, reverse, etc.
                
                -- Spaced repetition fields (SM-2 algorithm)
                interval INTEGER DEFAULT 0,  -- Days until next review
                repetitions INTEGER DEFAULT 0,
                ease_factor REAL DEFAULT 2.5,
                next_review TIMESTAMP,
                last_review TIMESTAMP,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                last_modified_by TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                is_deleted INTEGER DEFAULT 0,
                is_suspended INTEGER DEFAULT 0,
                metadata TEXT     -- JSON for additional Anki fields
            )
        """)
        
        # Review history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_history (
                id TEXT PRIMARY KEY,
                flashcard_id TEXT REFERENCES flashcards(id) ON DELETE CASCADE,
                reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                rating INTEGER NOT NULL,  -- 0-5 (Again, Hard, Good, Easy, etc.)
                time_taken INTEGER,       -- Seconds to answer
                interval_before INTEGER,  -- Previous interval
                interval_after INTEGER,   -- New interval
                ease_before REAL,
                ease_after REAL
            )
        """)
        
        # Mindmaps table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mindmaps (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                root_node_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                last_modified_by TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                is_deleted INTEGER DEFAULT 0,
                metadata TEXT     -- JSON for styling, layout preferences
            )
        """)
        
        # Mindmap nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mindmap_nodes (
                id TEXT PRIMARY KEY,
                mindmap_id TEXT REFERENCES mindmaps(id) ON DELETE CASCADE,
                parent_id TEXT REFERENCES mindmap_nodes(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                position_x REAL DEFAULT 0,
                position_y REAL DEFAULT 0,
                node_order INTEGER DEFAULT 0,
                color TEXT,
                icon TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT     -- JSON for additional properties
            )
        """)
        
        # Study sessions table (for tracking study time and progress)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS study_sessions (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,  -- topic, flashcard_deck, mindmap
                entity_id TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                duration INTEGER,  -- Seconds
                cards_reviewed INTEGER DEFAULT 0,
                topics_completed INTEGER DEFAULT 0,
                metadata TEXT     -- JSON for session details
            )
        """)
        
        # Create FTS5 virtual tables for search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS topics_fts USING fts5(
                title, content, content=topics, content_rowid=rowid
            )
        """)
        
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS flashcards_fts USING fts5(
                front, back, tags, content=flashcards, content_rowid=rowid
            )
        """)
        
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS mindmap_nodes_fts USING fts5(
                text, content=mindmap_nodes, content_rowid=rowid
            )
        """)
        
        # Create triggers to keep FTS tables in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS topics_ai AFTER INSERT ON topics BEGIN
                INSERT INTO topics_fts(rowid, title, content) 
                VALUES (new.rowid, new.title, new.content);
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS topics_ad AFTER DELETE ON topics BEGIN
                DELETE FROM topics_fts WHERE rowid = old.rowid;
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS topics_au AFTER UPDATE ON topics BEGIN
                UPDATE topics_fts SET title = new.title, content = new.content 
                WHERE rowid = new.rowid;
            END
        """)
        
        # Similar triggers for flashcards
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS flashcards_ai AFTER INSERT ON flashcards BEGIN
                INSERT INTO flashcards_fts(rowid, front, back, tags) 
                VALUES (new.rowid, new.front, new.back, new.tags);
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS flashcards_ad AFTER DELETE ON flashcards BEGIN
                DELETE FROM flashcards_fts WHERE rowid = old.rowid;
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS flashcards_au AFTER UPDATE ON flashcards BEGIN
                UPDATE flashcards_fts SET front = new.front, back = new.back, tags = new.tags 
                WHERE rowid = new.rowid;
            END
        """)
        
        # Similar triggers for mindmap_nodes
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS mindmap_nodes_ai AFTER INSERT ON mindmap_nodes BEGIN
                INSERT INTO mindmap_nodes_fts(rowid, text) 
                VALUES (new.rowid, new.text);
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS mindmap_nodes_ad AFTER DELETE ON mindmap_nodes BEGIN
                DELETE FROM mindmap_nodes_fts WHERE rowid = old.rowid;
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS mindmap_nodes_au AFTER UPDATE ON mindmap_nodes BEGIN
                UPDATE mindmap_nodes_fts SET text = new.text 
                WHERE rowid = new.rowid;
            END
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_path_id ON topics(path_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_parent_id ON topics(parent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_deck_id ON flashcards(deck_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_next_review ON flashcards(next_review)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_history_flashcard_id ON review_history(flashcard_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mindmap_nodes_mindmap_id ON mindmap_nodes(mindmap_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_study_sessions_entity ON study_sessions(entity_type, entity_id)")
        
        logger.info("Study database schema created successfully")
    
    def _migrate_schema(self, cursor: sqlite3.Cursor, from_version: int, to_version: int):
        """Migrate database schema from one version to another."""
        logger.info(f"Migrating database schema from version {from_version} to {to_version}")
        
        # Add migration logic here as schema evolves
        # For now, just update the version
        cursor.execute("UPDATE metadata SET value = ? WHERE key = 'schema_version'", (str(to_version),))
    
    def close(self):
        """Close database connection for current thread."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')