# tests/Notes/test_notes_integration.py
# Integration tests for Notes functionality using real database

import pytest
import pytest_asyncio
import tempfile
import os
from datetime import datetime, timezone
import uuid
from pathlib import Path

# Local imports
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesLibrary
from tldw_chatbook.Notes.sync_engine import SyncEngine

# Test marker for integration tests
pytestmark = pytest.mark.integration

#######################################################################################################################
#
# Fixtures

@pytest.fixture
def test_db():
    """Create a real CharactersRAGDB instance for testing"""
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    tmp.close()
    
    db = CharactersRAGDB(tmp.name, "test_client")
    yield db
    
    db.close()
    os.unlink(tmp.name)


@pytest.fixture
def notes_library(test_db, tmp_path):
    """Create a NotesLibrary instance with real database"""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    
    library = NotesLibrary(
        db=test_db,
        notes_directory=str(notes_dir),
        sync_enabled=False  # Disable sync for basic tests
    )
    return library


@pytest.fixture
def sync_enabled_library(test_db, tmp_path):
    """Create a NotesLibrary instance with sync enabled"""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    
    library = NotesLibrary(
        db=test_db,
        notes_directory=str(notes_dir),
        sync_enabled=True
    )
    return library


# Helper functions

def create_test_note(library: NotesLibrary, title: str = "Test Note", content: str = "Test content") -> str:
    """Helper to create a test note"""
    note_id = library.create_note(title=title, content=content)
    return note_id


def create_test_keyword(db: CharactersRAGDB, keyword: str = "test_keyword") -> tuple:
    """Helper to create a test keyword"""
    return db.add_keyword(keyword)


#######################################################################################################################
#
# Test Classes

class TestNotesBasicOperations:
    """Test basic CRUD operations for notes with real database"""
    
    def test_create_note(self, notes_library):
        """Test creating a note"""
        note_id = notes_library.create_note(
            title="My Test Note",
            content="This is test content"
        )
        
        assert note_id is not None
        assert isinstance(note_id, str)
        
        # Verify note exists in database
        note = notes_library.db.get_note_by_id(note_id)
        assert note is not None
        assert note['title'] == "My Test Note"
        assert note['content'] == "This is test content"
    
    def test_get_note(self, notes_library):
        """Test retrieving a note"""
        # Create a note
        note_id = create_test_note(notes_library, "Get Test", "Content to retrieve")
        
        # Get the note
        note = notes_library.get_note(note_id)
        assert note is not None
        assert note['id'] == note_id
        assert note['title'] == "Get Test"
        assert note['content'] == "Content to retrieve"
    
    def test_update_note(self, notes_library):
        """Test updating a note"""
        # Create a note
        note_id = create_test_note(notes_library)
        
        # Update it
        success = notes_library.update_note(
            note_id,
            title="Updated Title",
            content="Updated content"
        )
        assert success is True
        
        # Verify update
        note = notes_library.get_note(note_id)
        assert note['title'] == "Updated Title"
        assert note['content'] == "Updated content"
        assert note['version'] == 2  # Version should increment
    
    def test_delete_note(self, notes_library):
        """Test soft deleting a note"""
        # Create a note
        note_id = create_test_note(notes_library)
        
        # Delete it
        success = notes_library.delete_note(note_id)
        assert success is True
        
        # Note should be soft deleted
        note = notes_library.db.get_note_by_id(note_id)
        assert note is not None
        assert note['deleted'] is True
    
    def test_list_notes(self, notes_library):
        """Test listing notes"""
        # Create multiple notes
        note_ids = []
        for i in range(5):
            note_id = create_test_note(
                notes_library,
                f"Note {i}",
                f"Content {i}"
            )
            note_ids.append(note_id)
        
        # List all notes
        notes = notes_library.list_notes()
        assert len(notes) == 5
        
        # Check ordering (should be by last_modified desc)
        for i, note in enumerate(notes):
            assert note['title'] == f"Note {4-i}"  # Reversed order
    
    def test_search_notes(self, notes_library):
        """Test searching notes"""
        # Create notes with different content
        create_test_note(notes_library, "Python Guide", "Learn Python programming")
        create_test_note(notes_library, "Java Guide", "Learn Java programming")
        create_test_note(notes_library, "Recipe", "How to make pizza")
        
        # Search for programming
        results = notes_library.search_notes("programming")
        assert len(results) == 2
        
        # Search for pizza
        results = notes_library.search_notes("pizza")
        assert len(results) == 1
        assert results[0]['title'] == "Recipe"


class TestNotesKeywordIntegration:
    """Test notes and keywords integration"""
    
    def test_link_note_to_keyword(self, notes_library):
        """Test linking a note to keywords"""
        # Create note and keywords
        note_id = create_test_note(notes_library)
        kw1_id, kw1_uuid = create_test_keyword(notes_library.db, "python")
        kw2_id, kw2_uuid = create_test_keyword(notes_library.db, "programming")
        
        # Link note to keywords
        notes_library.db.link_note_to_keyword(note_id, kw1_id)
        notes_library.db.link_note_to_keyword(note_id, kw2_id)
        
        # Get keywords for note
        keywords = notes_library.db.get_keywords_for_note(note_id)
        assert len(keywords) == 2
        keyword_names = [kw['keyword'] for kw in keywords]
        assert "python" in keyword_names
        assert "programming" in keyword_names
    
    def test_get_notes_for_keyword(self, notes_library):
        """Test getting notes for a keyword"""
        # Create keyword
        kw_id, kw_uuid = create_test_keyword(notes_library.db, "important")
        
        # Create multiple notes and link some to keyword
        note1_id = create_test_note(notes_library, "Important Note 1")
        note2_id = create_test_note(notes_library, "Important Note 2")
        note3_id = create_test_note(notes_library, "Regular Note")
        
        notes_library.db.link_note_to_keyword(note1_id, kw_id)
        notes_library.db.link_note_to_keyword(note2_id, kw_id)
        
        # Get notes for keyword
        notes = notes_library.db.get_notes_for_keyword(kw_id)
        assert len(notes) == 2
        note_titles = [note['title'] for note in notes]
        assert "Important Note 1" in note_titles
        assert "Important Note 2" in note_titles
        assert "Regular Note" not in note_titles


class TestNotesFileSync:
    """Test notes file synchronization"""
    
    def test_create_note_creates_file(self, sync_enabled_library):
        """Test that creating a note creates a corresponding file"""
        note_id = create_test_note(sync_enabled_library, "File Test", "File content")
        
        # Check that file was created
        notes_dir = Path(sync_enabled_library.notes_directory)
        note_files = list(notes_dir.glob("*.md"))
        assert len(note_files) == 1
        
        # Verify file content
        with open(note_files[0], 'r') as f:
            content = f.read()
        assert "File Test" in content
        assert "File content" in content
    
    def test_update_note_updates_file(self, sync_enabled_library):
        """Test that updating a note updates the file"""
        note_id = create_test_note(sync_enabled_library)
        
        # Update note
        sync_enabled_library.update_note(note_id, content="Updated file content")
        
        # Check file was updated
        notes_dir = Path(sync_enabled_library.notes_directory)
        note_files = list(notes_dir.glob("*.md"))
        assert len(note_files) == 1
        
        with open(note_files[0], 'r') as f:
            content = f.read()
        assert "Updated file content" in content
    
    def test_file_change_syncs_to_db(self, sync_enabled_library):
        """Test that changing a file syncs to database"""
        # Create a note
        note_id = create_test_note(sync_enabled_library, "Sync Test", "Original content")
        
        # Find the file
        notes_dir = Path(sync_enabled_library.notes_directory)
        note_files = list(notes_dir.glob("*.md"))
        assert len(note_files) == 1
        note_file = note_files[0]
        
        # Modify the file directly
        with open(note_file, 'w') as f:
            f.write("---\n")
            f.write("title: Sync Test\n")
            f.write("---\n\n")
            f.write("Modified content from file")
        
        # Trigger sync
        if hasattr(sync_enabled_library, 'sync_engine') and sync_enabled_library.sync_engine:
            sync_enabled_library.sync_engine.sync()
        
        # Check database was updated
        note = sync_enabled_library.get_note(note_id)
        assert "Modified content from file" in note['content']
    
    def test_delete_note_removes_file(self, sync_enabled_library):
        """Test that deleting a note removes the file"""
        note_id = create_test_note(sync_enabled_library)
        
        # Verify file exists
        notes_dir = Path(sync_enabled_library.notes_directory)
        note_files = list(notes_dir.glob("*.md"))
        assert len(note_files) == 1
        
        # Delete note
        sync_enabled_library.delete_note(note_id)
        
        # File should be removed
        note_files = list(notes_dir.glob("*.md"))
        assert len(note_files) == 0


class TestNotesVersioning:
    """Test notes versioning and conflict handling"""
    
    def test_concurrent_updates(self, notes_library):
        """Test handling of concurrent updates"""
        # Create a note
        note_id = create_test_note(notes_library)
        
        # Get note to have version
        note = notes_library.get_note(note_id)
        version = note['version']
        
        # First update should succeed
        success = notes_library.db.update_note(
            note_id=note_id,
            title="Update 1",
            expected_version=version
        )
        assert success is True
        
        # Second update with same version should fail (conflict)
        from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError
        with pytest.raises(ConflictError):
            notes_library.db.update_note(
                note_id=note_id,
                title="Update 2",
                expected_version=version  # Using old version
            )
    
    def test_version_increment(self, notes_library):
        """Test that version increments on each update"""
        note_id = create_test_note(notes_library)
        
        # Initial version should be 1
        note = notes_library.get_note(note_id)
        assert note['version'] == 1
        
        # Update and check version
        for i in range(2, 5):
            notes_library.update_note(note_id, content=f"Version {i}")
            note = notes_library.get_note(note_id)
            assert note['version'] == i


class TestNotesPerformance:
    """Test notes performance with larger datasets"""
    
    def test_bulk_create_performance(self, notes_library):
        """Test creating many notes"""
        import time
        
        start_time = time.time()
        note_ids = []
        
        # Create 100 notes
        for i in range(100):
            note_id = create_test_note(
                notes_library,
                f"Bulk Note {i}",
                f"Content for note {i}"
            )
            note_ids.append(note_id)
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed < 10.0  # 10 seconds for 100 notes
        
        # Verify all created
        notes = notes_library.list_notes(limit=200)
        assert len(notes) == 100
    
    def test_search_performance(self, notes_library):
        """Test search performance with many notes"""
        # Create notes with varied content
        for i in range(50):
            create_test_note(
                notes_library,
                f"Note {i}",
                f"Python programming tutorial part {i}"
            )
        
        for i in range(50):
            create_test_note(
                notes_library,
                f"Recipe {i}",
                f"Cooking recipe number {i}"
            )
        
        import time
        start_time = time.time()
        
        # Search for programming
        results = notes_library.search_notes("programming")
        
        elapsed = time.time() - start_time
        
        # Should be fast
        assert elapsed < 1.0  # Less than 1 second
        assert len(results) == 50


class TestNotesErrorHandling:
    """Test error handling in notes operations"""
    
    def test_get_nonexistent_note(self, notes_library):
        """Test getting a note that doesn't exist"""
        fake_id = str(uuid.uuid4())
        note = notes_library.get_note(fake_id)
        assert note is None
    
    def test_update_nonexistent_note(self, notes_library):
        """Test updating a note that doesn't exist"""
        fake_id = str(uuid.uuid4())
        success = notes_library.update_note(fake_id, title="Won't work")
        assert success is False
    
    def test_invalid_note_data(self, notes_library):
        """Test creating note with invalid data"""
        from tldw_chatbook.DB.ChaChaNotes_DB import InputError
        
        # Empty title should raise error
        with pytest.raises(InputError):
            notes_library.db.add_note(title="", content="Content")
    
    def test_database_transaction_rollback(self, test_db):
        """Test that failed operations don't leave partial data"""
        # Start a transaction
        with test_db.transaction() as cursor:
            # Create a note
            note_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO notes (id, title, content, client_id) VALUES (?, ?, ?, ?)",
                (note_id, "Test", "Content", test_db.client_id)
            )
            
            # Force an error to trigger rollback
            with pytest.raises(Exception):
                cursor.execute("INVALID SQL")
        
        # Note should not exist due to rollback
        note = test_db.get_note_by_id(note_id)
        assert note is None