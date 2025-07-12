"""
db_test_utilities.py
--------------------

Specialized database testing utilities for tldw_chatbook.
Provides fixtures and helpers specific to database testing patterns.
"""

import pytest
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from contextlib import contextmanager
import uuid


# ===========================================
# Database Schema Helpers
# ===========================================

class TestDatabaseSchema:
    """Common test database schemas."""
    
    CONVERSATIONS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT,
        version INTEGER DEFAULT 1,
        client_id TEXT NOT NULL,
        character_id TEXT,
        metadata TEXT
    );
    
    CREATE INDEX IF NOT EXISTS idx_conversations_client 
    ON conversations(client_id, deleted_at);
    """
    
    MESSAGES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        metadata TEXT,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
    );
    
    CREATE INDEX IF NOT EXISTS idx_messages_conversation 
    ON messages(conversation_id, timestamp);
    """
    
    NOTES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS notes (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted_at TEXT,
        version INTEGER DEFAULT 1,
        client_id TEXT NOT NULL,
        file_path TEXT,
        metadata TEXT
    );
    
    CREATE INDEX IF NOT EXISTS idx_notes_client 
    ON notes(client_id, deleted_at);
    """
    
    FTS5_SCHEMA = """
    CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
        title, content, content=notes, content_rowid=rowid
    );
    
    CREATE TRIGGER IF NOT EXISTS notes_fts_insert 
    AFTER INSERT ON notes BEGIN
        INSERT INTO notes_fts(rowid, title, content) 
        VALUES (new.rowid, new.title, new.content);
    END;
    
    CREATE TRIGGER IF NOT EXISTS notes_fts_update 
    AFTER UPDATE ON notes BEGIN
        UPDATE notes_fts 
        SET title = new.title, content = new.content 
        WHERE rowid = new.rowid;
    END;
    
    CREATE TRIGGER IF NOT EXISTS notes_fts_delete 
    AFTER DELETE ON notes BEGIN
        DELETE FROM notes_fts WHERE rowid = old.rowid;
    END;
    """
    
    SYNC_LOG_SCHEMA = """
    CREATE TABLE IF NOT EXISTS sync_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_name TEXT NOT NULL,
        record_id TEXT NOT NULL,
        operation TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        client_id TEXT NOT NULL,
        data TEXT
    );
    
    CREATE INDEX IF NOT EXISTS idx_sync_log_timestamp 
    ON sync_log(timestamp);
    """


@pytest.fixture
def test_db_schema():
    """Provide test database schemas."""
    return TestDatabaseSchema()


# ===========================================
# Database Setup Fixtures
# ===========================================

@pytest.fixture
def setup_test_db():
    """Setup a complete test database with all tables."""
    def _setup(conn: sqlite3.Connection, include_fts: bool = True):
        schema = TestDatabaseSchema()
        
        # Create tables
        conn.executescript(schema.CONVERSATIONS_SCHEMA)
        conn.executescript(schema.MESSAGES_SCHEMA)
        conn.executescript(schema.NOTES_SCHEMA)
        
        if include_fts:
            conn.executescript(schema.FTS5_SCHEMA)
        
        conn.executescript(schema.SYNC_LOG_SCHEMA)
        
        # Create additional common tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS characters (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                personality TEXT,
                created_at TEXT NOT NULL,
                client_id TEXT NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL UNIQUE
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS note_keywords (
                note_id TEXT NOT NULL,
                keyword_id INTEGER NOT NULL,
                PRIMARY KEY (note_id, keyword_id),
                FOREIGN KEY (note_id) REFERENCES notes(id),
                FOREIGN KEY (keyword_id) REFERENCES keywords(id)
            )
        """)
        
        conn.commit()
    
    return _setup


# ===========================================
# Test Data Population
# ===========================================

class DatabasePopulator:
    """Helper class to populate test databases with data."""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    
    def add_conversation(
        self, 
        title: str = "Test Conversation",
        client_id: str = "test_client",
        character_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Add a conversation and return its ID."""
        conv_id = kwargs.get('id', str(uuid.uuid4()))
        now = datetime.now(timezone.utc).isoformat()
        
        self.conn.execute("""
            INSERT INTO conversations (id, title, created_at, updated_at, client_id, character_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (conv_id, title, now, now, client_id, character_id))
        
        return conv_id
    
    def add_message(
        self,
        conversation_id: str,
        role: str = "user",
        content: str = "Test message",
        **kwargs
    ) -> str:
        """Add a message to a conversation."""
        msg_id = kwargs.get('id', str(uuid.uuid4()))
        timestamp = kwargs.get('timestamp', datetime.now(timezone.utc).isoformat())
        
        self.conn.execute("""
            INSERT INTO messages (id, conversation_id, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (msg_id, conversation_id, role, content, timestamp))
        
        return msg_id
    
    def add_note(
        self,
        title: str = "Test Note",
        content: str = "Test content",
        client_id: str = "test_client",
        **kwargs
    ) -> str:
        """Add a note and return its ID."""
        note_id = kwargs.get('id', str(uuid.uuid4()))
        now = datetime.now(timezone.utc).isoformat()
        
        self.conn.execute("""
            INSERT INTO notes (id, title, content, created_at, updated_at, client_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (note_id, title, content, now, now, client_id))
        
        return note_id
    
    def add_character(
        self,
        name: str = "Test Character",
        client_id: str = "test_client",
        **kwargs
    ) -> str:
        """Add a character and return its ID."""
        char_id = kwargs.get('id', str(uuid.uuid4()))
        now = datetime.now(timezone.utc).isoformat()
        
        self.conn.execute("""
            INSERT INTO characters (id, name, description, personality, created_at, client_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            char_id, name,
            kwargs.get('description', 'Test character description'),
            kwargs.get('personality', 'Helpful and friendly'),
            now, client_id
        ))
        
        return char_id
    
    def add_keywords_to_note(self, note_id: str, keywords: List[str]):
        """Add keywords to a note."""
        for keyword in keywords:
            # Insert or get keyword
            cursor = self.conn.execute(
                "INSERT OR IGNORE INTO keywords (keyword) VALUES (?)", (keyword,)
            )
            
            if cursor.lastrowid:
                keyword_id = cursor.lastrowid
            else:
                keyword_id = self.conn.execute(
                    "SELECT id FROM keywords WHERE keyword = ?", (keyword,)
                ).fetchone()[0]
            
            # Link to note
            self.conn.execute(
                "INSERT INTO note_keywords (note_id, keyword_id) VALUES (?, ?)",
                (note_id, keyword_id)
            )
    
    def populate_test_data(self, num_conversations: int = 3, messages_per_conv: int = 5):
        """Populate database with test data."""
        # Add characters
        char1_id = self.add_character("Alice", "test_client")
        char2_id = self.add_character("Bob", "test_client")
        
        # Add conversations with messages
        for i in range(num_conversations):
            conv_id = self.add_conversation(
                f"Conversation {i+1}",
                character_id=char1_id if i % 2 == 0 else char2_id
            )
            
            for j in range(messages_per_conv):
                role = "user" if j % 2 == 0 else "assistant"
                self.add_message(conv_id, role, f"Message {j+1} in conv {i+1}")
        
        # Add notes with keywords
        for i in range(5):
            note_id = self.add_note(f"Note {i+1}", f"Content for note {i+1}")
            self.add_keywords_to_note(note_id, [f"tag{i}", "common"])
        
        self.conn.commit()


@pytest.fixture
def db_populator():
    """Provide database populator factory."""
    def _create_populator(conn: sqlite3.Connection) -> DatabasePopulator:
        return DatabasePopulator(conn)
    return _create_populator


# ===========================================
# Database Testing Helpers
# ===========================================

@contextmanager
def db_transaction(conn: sqlite3.Connection):
    """Context manager for database transactions with rollback on error."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def assert_table_exists(conn: sqlite3.Connection, table_name: str):
    """Assert that a table exists in the database."""
    result = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    ).fetchone()
    assert result is not None, f"Table '{table_name}' does not exist"


def assert_row_count(conn: sqlite3.Connection, table_name: str, expected_count: int):
    """Assert the number of rows in a table."""
    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    assert count == expected_count, \
        f"Expected {expected_count} rows in {table_name}, got {count}"


def assert_record_exists(
    conn: sqlite3.Connection,
    table_name: str,
    conditions: Dict[str, Any]
) -> sqlite3.Row:
    """Assert that a record exists with given conditions."""
    where_clause = " AND ".join([f"{k} = ?" for k in conditions.keys()])
    query = f"SELECT * FROM {table_name} WHERE {where_clause}"
    
    result = conn.execute(query, list(conditions.values())).fetchone()
    assert result is not None, \
        f"No record found in {table_name} with conditions: {conditions}"
    
    return result


def get_table_schema(conn: sqlite3.Connection, table_name: str) -> List[Tuple]:
    """Get the schema of a table."""
    return conn.execute(f"PRAGMA table_info({table_name})").fetchall()


# ===========================================
# Database Migration Testing
# ===========================================

@pytest.fixture
def migration_test_helper():
    """Helper for testing database migrations."""
    class MigrationTestHelper:
        def __init__(self):
            self.versions = {}
        
        def save_version(self, conn: sqlite3.Connection, version: int):
            """Save database version."""
            conn.execute("""
                CREATE TABLE IF NOT EXISTS db_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            """)
            conn.execute(
                "INSERT INTO db_version (version, applied_at) VALUES (?, ?)",
                (version, datetime.now(timezone.utc).isoformat())
            )
        
        def get_version(self, conn: sqlite3.Connection) -> Optional[int]:
            """Get current database version."""
            try:
                result = conn.execute(
                    "SELECT MAX(version) FROM db_version"
                ).fetchone()
                return result[0] if result and result[0] else None
            except sqlite3.OperationalError:
                return None
        
        def apply_migration(
            self, 
            conn: sqlite3.Connection,
            from_version: int,
            to_version: int,
            migration_sql: str
        ):
            """Apply a migration and update version."""
            current = self.get_version(conn)
            assert current == from_version, \
                f"Expected version {from_version}, got {current}"
            
            conn.executescript(migration_sql)
            self.save_version(conn, to_version)
    
    return MigrationTestHelper()


# ===========================================
# Performance Testing for Databases
# ===========================================

@pytest.fixture
def db_performance_tester():
    """Database performance testing utilities."""
    class DBPerformanceTester:
        def __init__(self):
            self.query_times = {}
        
        @contextmanager
        def measure_query(self, name: str, conn: sqlite3.Connection):
            """Measure query execution time."""
            import time
            
            # Enable query timing
            conn.execute("PRAGMA query_only = OFF")
            
            start = time.perf_counter()
            yield
            duration = time.perf_counter() - start
            
            self.query_times[name] = duration
        
        def assert_query_performance(self, name: str, max_seconds: float):
            """Assert query completed within time limit."""
            assert name in self.query_times, f"Query '{name}' not measured"
            duration = self.query_times[name]
            assert duration <= max_seconds, \
                f"Query '{name}' took {duration:.3f}s, max allowed: {max_seconds}s"
        
        def create_large_dataset(
            self, 
            conn: sqlite3.Connection,
            populator: DatabasePopulator,
            num_records: int
        ):
            """Create a large dataset for performance testing."""
            with db_transaction(conn):
                for i in range(num_records):
                    if i % 1000 == 0:
                        conn.commit()  # Periodic commits for large datasets
                    
                    conv_id = populator.add_conversation(f"Conv {i}")
                    populator.add_message(conv_id, "user", f"Message {i}")
                    
                    if i % 10 == 0:
                        note_id = populator.add_note(f"Note {i}", f"Content {i}")
                        populator.add_keywords_to_note(note_id, ["perf", f"tag{i%5}"])
    
    return DBPerformanceTester()


# ===========================================
# Concurrent Access Testing
# ===========================================

@pytest.fixture
def concurrent_db_tester():
    """Test concurrent database access patterns."""
    import threading
    
    class ConcurrentDBTester:
        def __init__(self):
            self.errors = []
            self.results = {}
        
        def run_concurrent_operations(
            self,
            db_path: str,
            operations: List[Callable],
            num_threads: int = 5
        ):
            """Run database operations concurrently."""
            threads = []
            
            def worker(op_func, thread_id):
                try:
                    conn = sqlite3.connect(db_path)
                    conn.row_factory = sqlite3.Row
                    result = op_func(conn, thread_id)
                    self.results[thread_id] = result
                    conn.close()
                except Exception as e:
                    self.errors.append((thread_id, e))
            
            for i in range(num_threads):
                op_func = operations[i % len(operations)]
                thread = threading.Thread(target=worker, args=(op_func, i))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            assert not self.errors, f"Concurrent operations failed: {self.errors}"
    
    return ConcurrentDBTester()


# ===========================================
# Database Integrity Checks
# ===========================================

def check_referential_integrity(conn: sqlite3.Connection) -> List[str]:
    """Check for referential integrity violations."""
    violations = []
    
    # Check messages -> conversations
    orphaned = conn.execute("""
        SELECT m.id, m.conversation_id 
        FROM messages m
        LEFT JOIN conversations c ON m.conversation_id = c.id
        WHERE c.id IS NULL
    """).fetchall()
    
    if orphaned:
        violations.append(f"Orphaned messages: {[row[0] for row in orphaned]}")
    
    # Check note_keywords -> notes
    orphaned = conn.execute("""
        SELECT nk.note_id, nk.keyword_id
        FROM note_keywords nk
        LEFT JOIN notes n ON nk.note_id = n.id
        WHERE n.id IS NULL
    """).fetchall()
    
    if orphaned:
        violations.append(f"Orphaned note keywords: {[row[0] for row in orphaned]}")
    
    return violations


def vacuum_and_analyze(conn: sqlite3.Connection):
    """Vacuum and analyze database for testing."""
    conn.execute("VACUUM")
    conn.execute("ANALYZE")


# ===========================================
# Example Usage
# ===========================================

"""
Example Usage:

1. Basic Database Setup:
   ```python
   def test_database_creation(memory_db, setup_test_db):
       setup_test_db(memory_db)
       assert_table_exists(memory_db, "conversations")
   ```

2. Populating Test Data:
   ```python
   def test_with_data(memory_db, setup_test_db, db_populator):
       setup_test_db(memory_db)
       populator = db_populator(memory_db)
       populator.populate_test_data(num_conversations=5)
       assert_row_count(memory_db, "conversations", 5)
   ```

3. Testing Queries:
   ```python
   def test_search_performance(memory_db, db_performance_tester):
       with db_performance_tester.measure_query("search", memory_db):
           results = memory_db.execute("SELECT * FROM notes_fts WHERE content MATCH ?", ("test",))
       db_performance_tester.assert_query_performance("search", max_seconds=0.1)
   ```

4. Migration Testing:
   ```python
   def test_migration(memory_db, migration_test_helper):
       migration_test_helper.apply_migration(
           memory_db, 
           from_version=1, 
           to_version=2,
           migration_sql="ALTER TABLE notes ADD COLUMN tags TEXT;"
       )
   ```

5. Concurrent Access:
   ```python
   def test_concurrent_writes(temp_db_path, concurrent_db_tester):
       def write_op(conn, thread_id):
           conn.execute("INSERT INTO notes (id, title) VALUES (?, ?)", 
                       (f"note_{thread_id}", f"Title {thread_id}"))
           conn.commit()
       
       concurrent_db_tester.run_concurrent_operations(
           temp_db_path,
           [write_op],
           num_threads=10
       )
   ```
"""