# test_prompts_db_pytest.py
# Pytest-based tests for Prompts_DB

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone

from tldw_chatbook.DB.Prompts_DB import (
    PromptsDatabase,
    DatabaseError,
    SchemaError,
    InputError,
    ConflictError,
    add_or_update_prompt,
    load_prompt_details_for_ui,
    export_prompt_keywords_to_csv,
    view_prompt_keywords_markdown,
    export_prompts_formatted,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_prompts.db"
    yield str(db_path)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def in_memory_db():
    """Create an in-memory database for testing."""
    db = PromptsDatabase(':memory:', client_id='test_client')
    yield db
    db.close_connection()


@pytest.fixture
def file_db(temp_db_path):
    """Create a file-based database for testing."""
    db = PromptsDatabase(temp_db_path, client_id='test_client')
    yield db
    db.close_connection()


class TestPromptsDBInitialization:
    """Test database initialization and schema creation."""
    
    def test_memory_db_initialization(self):
        """Test in-memory database initialization."""
        db = PromptsDatabase(':memory:', client_id='test_client')
        assert db.is_memory_db is True
        assert db.db_path_str == ':memory:'
        assert db.client_id == 'test_client'
        db.close_connection()
    
    def test_file_db_initialization(self, temp_db_path):
        """Test file-based database initialization."""
        db = PromptsDatabase(temp_db_path, client_id='test_client')
        assert db.db_path_str == temp_db_path or str(db.db_path) == temp_db_path
        assert Path(temp_db_path).exists()
        db.close_connection()
    
    def test_schema_creation(self, in_memory_db):
        """Test that all required tables are created."""
        conn = in_memory_db._get_thread_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['Prompts', 'Keywords', 'PromptKeywords', 'sync_log']
        for table in expected_tables:
            assert table in tables
    
    def test_fts_tables_creation(self, in_memory_db):
        """Test that FTS5 virtual tables are created."""
        conn = in_memory_db._get_thread_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts'"
        )
        fts_tables = [row[0] for row in cursor.fetchall()]
        assert 'Prompts_fts' in fts_tables


class TestPromptOperations:
    """Test CRUD operations for prompts."""
    
    def test_add_prompt_basic(self, in_memory_db):
        """Test basic prompt creation."""
        name = "Test Prompt"
        prompt_id = in_memory_db.add_prompt(
            name=name,
            author="Test Author",
            system="Test system prompt",
            user="Test user prompt"
        )
        
        assert prompt_id is not None
        assert isinstance(prompt_id, int)
        
        # Verify prompt was created
        prompt = in_memory_db.get_prompt_details(prompt_id)
        assert prompt is not None
        assert prompt['name'] == name
    
    def test_add_prompt_with_keywords(self, in_memory_db):
        """Test prompt creation with keywords."""
        prompt_id = in_memory_db.add_prompt(
            name="Test Prompt with Keywords",
            author="Test Author",
            system="System prompt",
            user="User prompt",
            keywords=["test", "keywords", "example"]
        )
        
        prompt = in_memory_db.get_prompt_details(prompt_id)
        assert len(prompt['keywords']) == 3
        assert "test" in prompt['keywords']
    
    def test_update_prompt(self, in_memory_db):
        """Test prompt update."""
        # Create prompt
        prompt_id = in_memory_db.add_prompt(
            name="Original Name",
            author="Original Author"
        )
        
        # Update it
        in_memory_db.update_prompt(
            prompt_id,
            name="Updated Name",
            system="Updated system prompt"
        )
        
        # Verify update
        prompt = in_memory_db.get_prompt_details(prompt_id)
        assert prompt['name'] == "Updated Name"
        assert prompt['system'] == "Updated system prompt"
        assert prompt['author'] == "Original Author"  # Should not change
    
    def test_delete_prompt(self, in_memory_db):
        """Test prompt deletion (soft delete)."""
        prompt_id = in_memory_db.add_prompt(name="To Delete")
        
        # Delete it
        in_memory_db.delete_prompt(prompt_id)
        
        # Should not be in active prompts
        prompts = in_memory_db.get_all_prompts()
        assert not any(p['id'] == prompt_id for p in prompts)
    
    def test_duplicate_prompt_name(self, in_memory_db):
        """Test that duplicate prompt names are rejected."""
        name = "Unique Prompt"
        in_memory_db.add_prompt(name=name)
        
        with pytest.raises(ConflictError):
            in_memory_db.add_prompt(name=name)


class TestKeywordOperations:
    """Test keyword management."""
    
    def test_add_keyword(self, in_memory_db):
        """Test adding keywords."""
        keyword_id = in_memory_db.add_keyword("test-keyword")
        assert keyword_id is not None
        
        # Adding same keyword should return same ID
        keyword_id2 = in_memory_db.add_keyword("test-keyword")
        assert keyword_id == keyword_id2
    
    def test_get_all_keywords(self, in_memory_db):
        """Test retrieving all keywords."""
        # Add some keywords
        keywords = ["python", "testing", "database"]
        for kw in keywords:
            in_memory_db.add_keyword(kw)
        
        all_keywords = in_memory_db.get_all_keywords()
        keyword_names = [kw['name'] for kw in all_keywords]
        
        for kw in keywords:
            assert kw in keyword_names


class TestSearchFunctionality:
    """Test search and filtering operations."""
    
    def test_search_prompts_by_keyword(self, in_memory_db):
        """Test searching prompts by keyword."""
        # Create prompts with different keywords
        in_memory_db.add_prompt(
            name="Python Tutorial",
            keywords=["python", "tutorial"]
        )
        in_memory_db.add_prompt(
            name="SQL Guide",
            keywords=["sql", "database"]
        )
        in_memory_db.add_prompt(
            name="Python Database",
            keywords=["python", "database"]
        )
        
        # Search for python prompts
        results = in_memory_db.search_prompts_by_keyword("python")
        assert len(results) == 2
        
        # Search for database prompts
        results = in_memory_db.search_prompts_by_keyword("database")
        assert len(results) == 2
    
    def test_search_prompts_by_text(self, in_memory_db):
        """Test full-text search."""
        in_memory_db.add_prompt(
            name="Code Review Assistant",
            system="Help review code for best practices"
        )
        in_memory_db.add_prompt(
            name="Writing Helper",
            system="Assist with writing and editing"
        )
        
        # Search for "review"
        results = in_memory_db.search_prompts_by_text("review")
        assert len(results) == 1
        assert results[0]['name'] == "Code Review Assistant"


class TestStandaloneFunctions:
    """Test standalone utility functions."""
    
    def test_add_or_update_prompt(self, temp_db_path):
        """Test the add_or_update_prompt standalone function."""
        # First add
        prompt_id = add_or_update_prompt(
            db_path=temp_db_path,
            prompt_name="Test Standalone",
            keywords=["test"],
            author="Tester"
        )
        assert prompt_id is not None
        
        # Update (same name)
        prompt_id2 = add_or_update_prompt(
            db_path=temp_db_path,
            prompt_name="Test Standalone",
            system="Updated system"
        )
        assert prompt_id == prompt_id2
    
    def test_load_prompt_details(self, temp_db_path):
        """Test loading prompt details for UI."""
        # Add a prompt
        prompt_id = add_or_update_prompt(
            db_path=temp_db_path,
            prompt_name="UI Test",
            keywords=["ui", "test"]
        )
        
        # Load details
        details = load_prompt_details_for_ui(temp_db_path, prompt_id)
        assert details['name'] == "UI Test"
        assert len(details['keywords']) == 2
    
    def test_export_keywords_csv(self, temp_db_path, tmp_path):
        """Test exporting keywords to CSV."""
        # Add prompts with keywords
        add_or_update_prompt(
            db_path=temp_db_path,
            prompt_name="Prompt 1",
            keywords=["python", "test"]
        )
        add_or_update_prompt(
            db_path=temp_db_path,
            prompt_name="Prompt 2",
            keywords=["python", "database"]
        )
        
        # Export
        csv_file = tmp_path / "keywords.csv"
        export_prompt_keywords_to_csv(temp_db_path, str(csv_file))
        
        assert csv_file.exists()
        content = csv_file.read_text()
        assert "python" in content
        assert "2" in content  # Usage count


class TestErrorHandling:
    """Test error conditions and validation."""
    
    def test_invalid_prompt_name(self, in_memory_db):
        """Test validation of prompt names."""
        with pytest.raises(InputError):
            in_memory_db.add_prompt(name="")  # Empty name
        
        with pytest.raises(InputError):
            in_memory_db.add_prompt(name="   ")  # Whitespace only
    
    def test_nonexistent_prompt(self, in_memory_db):
        """Test operations on non-existent prompts."""
        result = in_memory_db.get_prompt_details(9999)
        assert result is None
        
        # Update non-existent prompt should not fail
        in_memory_db.update_prompt(9999, name="Won't work")
        
        # Delete non-existent prompt should not fail
        in_memory_db.delete_prompt(9999)
    
    def test_concurrent_access(self, file_db):
        """Test thread-safe access."""
        import threading
        results = []
        
        def add_prompt(name):
            try:
                prompt_id = file_db.add_prompt(name=name)
                results.append(('success', prompt_id))
            except Exception as e:
                results.append(('error', str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=add_prompt, args=(f"Thread {i}",))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check results
        successes = [r for r in results if r[0] == 'success']
        assert len(successes) == 5  # All should succeed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])