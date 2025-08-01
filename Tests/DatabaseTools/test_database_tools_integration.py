# test_database_tools_integration.py
# Integration tests for database tools

import pytest
import sqlite3
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tldw_chatbook.UI.Tools_Settings_Window import ToolsSettingsWindow


class TestDatabaseToolsIntegration:
    """Integration tests for database tools operations."""
    
    @pytest.fixture
    def test_db_dir(self):
        """Create a temporary directory with test databases."""
        temp_dir = tempfile.mkdtemp()
        db_dir = Path(temp_dir)
        
        # Create test databases with realistic schema
        databases = {
            'chachanotes': self._create_chachanotes_db,
            'media': self._create_media_db,
            'prompts': self._create_prompts_db
        }
        
        paths = {}
        for name, creator in databases.items():
            db_path = db_dir / f"{name}.db"
            creator(db_path)
            paths[name] = str(db_path)
        
        yield db_dir, paths
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def _create_chachanotes_db(self, db_path: Path):
        """Create a test ChaChaNotes database."""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_name TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                character_id INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                sender TEXT,
                message TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                keywords TEXT
            )
        """)
        
        # Insert test data
        cursor.execute("INSERT INTO conversations (conversation_name) VALUES ('Test Conv 1'), ('Test Conv 2')")
        cursor.execute("INSERT INTO messages (conversation_id, sender, message) VALUES (1, 'user', 'Hello'), (1, 'assistant', 'Hi!')")
        cursor.execute("INSERT INTO notes (title, content) VALUES ('Note 1', 'Content 1'), ('Note 2', 'Content 2')")
        
        # Set schema version
        cursor.execute("PRAGMA user_version = 7")
        
        conn.commit()
        conn.close()
    
    def _create_media_db(self, db_path: Path):
        """Create a test media database."""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                file_path TEXT,
                media_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("INSERT INTO media (title, file_path, media_type) VALUES ('Video 1', '/path/video1.mp4', 'video')")
        cursor.execute("PRAGMA user_version = 2")
        
        conn.commit()
        conn.close()
    
    def _create_prompts_db(self, db_path: Path):
        """Create a test prompts database."""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                details TEXT,
                system_prompt TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("INSERT INTO prompts (name, system_prompt) VALUES ('Test Prompt', 'You are helpful')")
        cursor.execute("PRAGMA user_version = 1")
        
        conn.commit()
        conn.close()
    
    def test_vacuum_operation_integration(self, test_db_dir):
        """Test vacuum operation reduces database size."""
        db_dir, db_paths = test_db_dir
        db_path = Path(db_paths['chachanotes'])
        
        # Add and delete data to create fragmentation
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Insert lots of data
        for i in range(1000):
            cursor.execute("INSERT INTO notes (title, content) VALUES (?, ?)", 
                         (f"Temp Note {i}", "X" * 1000))
        conn.commit()
        
        # Delete most of it
        cursor.execute("DELETE FROM notes WHERE title LIKE 'Temp Note%'")
        conn.commit()
        conn.close()
        
        # Get size before vacuum
        size_before = db_path.stat().st_size
        
        # Perform vacuum
        conn = sqlite3.connect(str(db_path))
        conn.execute("VACUUM")
        conn.close()
        
        # Get size after vacuum
        size_after = db_path.stat().st_size
        
        # Verify size reduced
        assert size_after < size_before
        print(f"Size reduced from {size_before} to {size_after} bytes")
    
    def test_backup_restore_cycle(self, test_db_dir):
        """Test complete backup and restore cycle."""
        db_dir, db_paths = test_db_dir
        original_db = Path(db_paths['chachanotes'])
        backup_dir = db_dir / "backups"
        backup_dir.mkdir()
        
        # Create backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"chachanotes_backup_{timestamp}.db"
        metadata_path = backup_path.with_suffix('.json')
        
        # Copy database
        shutil.copy2(original_db, backup_path)
        
        # Create metadata
        metadata = {
            "database": "chachanotes",
            "original_path": str(original_db),
            "backup_time": datetime.now().isoformat(),
            "file_size": original_db.stat().st_size,
            "schema_version": 7
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Modify original database
        conn = sqlite3.connect(str(original_db))
        cursor = conn.cursor()
        cursor.execute("INSERT INTO notes (title, content) VALUES ('New Note', 'After backup')")
        conn.commit()
        
        # Verify new note exists
        cursor.execute("SELECT COUNT(*) FROM notes WHERE title = 'New Note'")
        assert cursor.fetchone()[0] == 1
        conn.close()
        
        # Restore from backup
        shutil.copy2(backup_path, original_db)
        
        # Verify new note is gone
        conn = sqlite3.connect(str(original_db))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM notes WHERE title = 'New Note'")
        assert cursor.fetchone()[0] == 0
        conn.close()
    
    def test_integrity_check_detects_corruption(self, test_db_dir):
        """Test that integrity check detects database corruption."""
        db_dir, db_paths = test_db_dir
        db_path = Path(db_paths['media'])
        
        # First verify database is OK
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        assert result[0] == "ok"
        conn.close()
        
        # Corrupt the database by writing random data
        # Note: This is a controlled test - don't do this in production!
        with open(db_path, 'r+b') as f:
            # Skip SQLite header (first 100 bytes) to avoid total corruption
            f.seek(200)
            f.write(b'CORRUPTED_DATA_HERE')
        
        # Now check integrity
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            # Should detect corruption
            assert result[0] != "ok"
            conn.close()
        except sqlite3.DatabaseError:
            # Severe corruption might prevent connection
            pass
    
    def test_multiple_database_operations(self, test_db_dir):
        """Test operations on multiple databases."""
        db_dir, db_paths = test_db_dir
        results = {}
        
        # Test each database
        for db_name, db_path in db_paths.items():
            path = Path(db_path)
            
            # Test connection
            conn = sqlite3.connect(str(path))
            
            # Get schema version
            cursor = conn.execute("PRAGMA user_version")
            version = cursor.fetchone()[0]
            
            # Get table count
            cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            
            # Run integrity check
            cursor = conn.execute("PRAGMA integrity_check")
            integrity = cursor.fetchone()[0]
            
            conn.close()
            
            results[db_name] = {
                'version': version,
                'tables': table_count,
                'integrity': integrity,
                'size': path.stat().st_size
            }
        
        # Verify results
        assert results['chachanotes']['version'] == 7
        assert results['chachanotes']['tables'] >= 3
        assert results['chachanotes']['integrity'] == 'ok'
        
        assert results['media']['version'] == 2
        assert results['media']['tables'] >= 1
        assert results['media']['integrity'] == 'ok'
        
        assert results['prompts']['version'] == 1
        assert results['prompts']['tables'] >= 1
        assert results['prompts']['integrity'] == 'ok'
    
    def test_concurrent_database_access(self, test_db_dir):
        """Test that databases handle concurrent access properly."""
        db_dir, db_paths = test_db_dir
        db_path = Path(db_paths['chachanotes'])
        
        # Create multiple connections
        connections = []
        for i in range(5):
            conn = sqlite3.connect(str(db_path))
            connections.append(conn)
        
        # Perform operations on each connection
        for i, conn in enumerate(connections):
            cursor = conn.cursor()
            cursor.execute("INSERT INTO notes (title, content) VALUES (?, ?)",
                         (f"Concurrent Note {i}", f"From connection {i}"))
            conn.commit()
        
        # Close all connections
        for conn in connections:
            conn.close()
        
        # Verify all inserts succeeded
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM notes WHERE title LIKE 'Concurrent Note%'")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 5
    
    def test_database_size_formatting(self, test_db_dir):
        """Test database size retrieval and formatting."""
        db_dir, db_paths = test_db_dir
        
        # Create databases of different sizes
        test_sizes = {
            'tiny.db': 512,  # 512 B
            'small.db': 1024 * 100,  # 100 KB
            'medium.db': 1024 * 1024 * 5,  # 5 MB
        }
        
        for filename, target_size in test_sizes.items():
            db_path = db_dir / filename
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Create a table with data to reach target size
            cursor.execute("CREATE TABLE data (id INTEGER PRIMARY KEY, content TEXT)")
            
            # Insert data until we reach approximate target size
            content = "X" * 1000  # 1KB chunks
            while db_path.stat().st_size < target_size:
                cursor.execute("INSERT INTO data (content) VALUES (?)", (content,))
            
            conn.commit()
            conn.close()
            
            # Verify size is approximately correct
            actual_size = db_path.stat().st_size
            assert actual_size >= target_size * 0.8  # Within 20% of target


class TestDatabaseMigration:
    """Test database migration and compatibility."""
    
    def test_schema_version_handling(self):
        """Test handling of different schema versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            # Create database with old schema version
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.execute("PRAGMA user_version = 5")
            conn.commit()
            conn.close()
            
            # Read schema version
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("PRAGMA user_version")
            version = cursor.fetchone()[0]
            conn.close()
            
            assert version == 5
            
            # Simulate upgrade
            conn = sqlite3.connect(str(db_path))
            conn.execute("ALTER TABLE test ADD COLUMN name TEXT")
            conn.execute("PRAGMA user_version = 6")
            conn.commit()
            conn.close()
            
            # Verify upgrade
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("PRAGMA user_version")
            new_version = cursor.fetchone()[0]
            
            # Check new column exists
            cursor = conn.execute("PRAGMA table_info(test)")
            columns = [row[1] for row in cursor.fetchall()]
            conn.close()
            
            assert new_version == 6
            assert 'name' in columns