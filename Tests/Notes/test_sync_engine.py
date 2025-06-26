"""
test_sync_engine.py
Test suite for the notes synchronization engine
"""
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pytest

from tldw_chatbook.Notes.sync_engine import (
    NotesSyncEngine, SyncDirection, ConflictResolution,
    SyncProgress, SyncFileInfo
)
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def test_db(temp_dir):
    """Create a test database."""
    db_path = temp_dir / "test_notes.db"
    db = CharactersRAGDB(str(db_path), "test_client")
    yield db
    db.close_connection()


@pytest.fixture
def notes_service(test_db, temp_dir):
    """Create a notes service instance."""
    service = NotesInteropService(
        base_db_directory=temp_dir,
        api_client_id="test_api",
        global_db_to_use=test_db
    )
    yield service
    service.close_all_user_connections()


@pytest.fixture
def sync_engine(notes_service, test_db):
    """Create a sync engine instance."""
    return NotesSyncEngine(notes_service, test_db)


class TestSyncEngine:
    """Test cases for the sync engine."""
    
    def test_calculate_hash(self, sync_engine):
        """Test content hashing."""
        content = "Hello, World!"
        hash1 = sync_engine._calculate_hash(content)
        hash2 = sync_engine._calculate_hash(content)
        hash3 = sync_engine._calculate_hash("Different content")
        
        assert hash1 == hash2  # Same content produces same hash
        assert hash1 != hash3  # Different content produces different hash
        assert len(hash1) == 64  # SHA256 produces 64 character hex string
    
    def test_get_file_info(self, sync_engine, temp_dir):
        """Test getting file information."""
        # Create a test file
        test_file = temp_dir / "test.md"
        test_content = "# Test Note\n\nThis is a test."
        test_file.write_text(test_content)
        
        file_info = sync_engine._get_file_info(test_file, temp_dir)
        
        assert file_info is not None
        assert file_info.absolute_path == test_file
        assert file_info.relative_path == Path("test.md")
        assert file_info.content == test_content
        assert file_info.extension == ".md"
        assert file_info.content_hash == sync_engine._calculate_hash(test_content)
    
    def test_scan_directory(self, sync_engine, temp_dir):
        """Test directory scanning."""
        # Create test files
        (temp_dir / "note1.md").write_text("Note 1")
        (temp_dir / "note2.txt").write_text("Note 2")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "note3.md").write_text("Note 3")
        (temp_dir / "ignored.pdf").write_text("Ignored")
        
        # Scan for markdown files only
        files = sync_engine._scan_directory(temp_dir, [".md"])
        
        assert len(files) == 2
        assert Path("note1.md") in files
        assert Path("subdir/note3.md") in files
        assert Path("note2.txt") not in files
        
        # Scan for both markdown and text files
        files = sync_engine._scan_directory(temp_dir, [".md", ".txt"])
        
        assert len(files) == 3
        assert Path("note2.txt") in files
    
    @pytest.mark.asyncio
    async def test_sync_disk_to_db(self, sync_engine, notes_service, temp_dir):
        """Test syncing from disk to database."""
        # Create test files
        sync_dir = temp_dir / "sync_test"
        sync_dir.mkdir()
        
        file1 = sync_dir / "note1.md"
        file1.write_text("# First Note\n\nContent of first note.")
        
        file2 = sync_dir / "folder" / "note2.md"
        file2.parent.mkdir()
        file2.write_text("# Second Note\n\nContent of second note.")
        
        # Run sync
        session_id, progress = await sync_engine.sync(
            root_path=sync_dir,
            user_id="test_user",
            direction=SyncDirection.DISK_TO_DB,
            conflict_resolution=ConflictResolution.ASK
        )
        
        # Verify results
        assert len(progress.created_notes) == 2
        assert len(progress.errors) == 0
        assert progress.processed_files == 2
        
        # Check that notes were created in database
        notes = notes_service.list_notes("test_user")
        assert len(notes) == 2
        
        # Verify note content
        note_titles = [note['title'] for note in notes]
        assert "note1" in note_titles
        assert "note2" in note_titles
    
    @pytest.mark.asyncio
    async def test_sync_db_to_disk(self, sync_engine, notes_service, temp_dir):
        """Test syncing from database to disk."""
        sync_dir = temp_dir / "sync_test"
        sync_dir.mkdir()
        
        # Create notes in database
        note1_id = notes_service.add_note(
            user_id="test_user",
            title="Database Note 1",
            content="Content from database 1"
        )
        
        note2_id = notes_service.add_note(
            user_id="test_user",
            title="Database Note 2",
            content="Content from database 2"
        )
        
        # Update sync metadata for notes
        note1 = notes_service.get_note_by_id("test_user", note1_id)
        notes_service.update_note_sync_metadata(
            user_id="test_user",
            note_id=note1_id,
            sync_metadata={
                'file_path_on_disk': str(sync_dir / "db_note1.md"),
                'relative_file_path_on_disk': "db_note1.md",
                'sync_root_folder': str(sync_dir),
                'is_externally_synced': 1,
                'file_extension': '.md'
            },
            expected_version=note1['version']
        )
        
        note2 = notes_service.get_note_by_id("test_user", note2_id)
        notes_service.update_note_sync_metadata(
            user_id="test_user",
            note_id=note2_id,
            sync_metadata={
                'file_path_on_disk': str(sync_dir / "subdir" / "db_note2.md"),
                'relative_file_path_on_disk': "subdir/db_note2.md",
                'sync_root_folder': str(sync_dir),
                'is_externally_synced': 1,
                'file_extension': '.md'
            },
            expected_version=note2['version']
        )
        
        # Run sync
        session_id, progress = await sync_engine.sync(
            root_path=sync_dir,
            user_id="test_user",
            direction=SyncDirection.DB_TO_DISK,
            conflict_resolution=ConflictResolution.ASK
        )
        
        # Verify results
        assert len(progress.created_files) == 2
        assert len(progress.errors) == 0
        
        # Check that files were created
        assert (sync_dir / "db_note1.md").exists()
        assert (sync_dir / "subdir" / "db_note2.md").exists()
        
        # Verify file content
        content1 = (sync_dir / "db_note1.md").read_text()
        assert content1 == "Content from database 1"
    
    @pytest.mark.asyncio
    async def test_conflict_detection(self, sync_engine, notes_service, temp_dir):
        """Test conflict detection in bidirectional sync."""
        sync_dir = temp_dir / "sync_test"
        sync_dir.mkdir()
        
        # Create a note and sync it
        note_id = notes_service.add_note(
            user_id="test_user",
            title="Conflict Test",
            content="Original content"
        )
        
        file_path = sync_dir / "conflict_test.md"
        note = notes_service.get_note_by_id("test_user", note_id)
        notes_service.update_note_sync_metadata(
            user_id="test_user",
            note_id=note_id,
            sync_metadata={
                'file_path_on_disk': str(file_path),
                'relative_file_path_on_disk': "conflict_test.md",
                'sync_root_folder': str(sync_dir),
                'is_externally_synced': 1,
                'file_extension': '.md'
            },
            expected_version=note['version']
        )
        
        # Create the file with initial content
        file_path.write_text("Original content")
        
        # Update sync metadata to establish baseline
        file_info = sync_engine._get_file_info(file_path, sync_dir)
        notes_service.update_note_sync_metadata(
            user_id="test_user",
            note_id=note_id,
            sync_metadata={
                'last_synced_disk_file_hash': file_info.content_hash,
                'last_synced_disk_file_mtime': file_info.mtime
            },
            expected_version=1
        )
        
        # Now modify both the file and the database note
        file_path.write_text("Modified on disk")
        notes_service.update_note(
            user_id="test_user",
            note_id=note_id,
            update_data={'content': "Modified in database"},
            expected_version=2
        )
        
        # Run bidirectional sync
        session_id, progress = await sync_engine.sync(
            root_path=sync_dir,
            user_id="test_user",
            direction=SyncDirection.BIDIRECTIONAL,
            conflict_resolution=ConflictResolution.ASK
        )
        
        # Should detect a conflict
        assert len(progress.conflicts) == 1
        conflict = progress.conflicts[0]
        assert conflict.conflict_type == 'both_changed'
        assert conflict.db_content == "Modified in database"
        assert conflict.disk_content == "Modified on disk"
    
    @pytest.mark.asyncio
    async def test_sync_with_progress_callback(self, sync_engine, temp_dir):
        """Test sync with progress callback."""
        sync_dir = temp_dir / "sync_test"
        sync_dir.mkdir()
        
        # Create multiple test files
        for i in range(5):
            (sync_dir / f"note{i}.md").write_text(f"Note {i} content")
        
        progress_updates = []
        
        def progress_callback(progress: SyncProgress):
            progress_updates.append({
                'processed': progress.processed_files,
                'total': progress.total_files
            })
        
        sync_engine.progress_callback = progress_callback
        
        # Run sync
        session_id, progress = await sync_engine.sync(
            root_path=sync_dir,
            user_id="test_user",
            direction=SyncDirection.DISK_TO_DB
        )
        
        # Verify progress updates were received
        assert len(progress_updates) > 0
        assert progress_updates[-1]['processed'] == 5
    
    def test_sync_session_creation(self, sync_engine, test_db, temp_dir):
        """Test sync session creation and tracking."""
        session_id = sync_engine._create_sync_session(
            sync_root=temp_dir,
            direction=SyncDirection.BIDIRECTIONAL,
            conflict_resolution=ConflictResolution.NEWER_WINS,
            user_id="test_user"
        )
        
        assert session_id is not None
        assert session_id in sync_engine._active_sessions
        
        # Verify session was created in database
        with test_db.transaction() as conn:
            cursor = conn.execute(
                "SELECT sync_direction, conflict_resolution, status FROM sync_sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            
        assert row is not None
        assert row[0] == "bidirectional"
        assert row[1] == "newer_wins"
        assert row[2] == "running"