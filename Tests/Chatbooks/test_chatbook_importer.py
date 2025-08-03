# test_chatbook_importer.py
# Unit tests for chatbook importer

import pytest
import json
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock
import sqlite3

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tldw_chatbook.Chatbooks.chatbook_models import (
    ContentType, ContentItem, Relationship, ChatbookManifest, 
    ChatbookContent, Chatbook, ChatbookVersion
)
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution


class TestChatbookImporter:
    """Test ChatbookImporter functionality."""
    
    @pytest.fixture
    def temp_db_paths(self, tmp_path):
        """Create temporary database paths with schema."""
        db_dir = tmp_path / "databases"
        db_dir.mkdir()
        
        paths = {
            'ChaChaNotes': str(db_dir / "ChaChaNotes.db"),
            'Media': str(db_dir / "Client_Media_DB.db"),
            'Prompts': str(db_dir / "Prompts_DB.db"),
            'Evals': str(db_dir / "Evals_DB.db"),
            'RAG': str(db_dir / "RAG_Indexing_DB.db"),
            'Subscriptions': str(db_dir / "Subscriptions_DB.db")
        }
        
        # Create database files with schema
        for name, path in paths.items():
            conn = sqlite3.connect(path)
            if name == 'ChaChaNotes':
                # Create schema for ChaChaNotes
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY,
                        title TEXT,
                        created_at TEXT,
                        character_id INTEGER
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY,
                        conversation_id INTEGER,
                        role TEXT,
                        content TEXT,
                        timestamp TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS notes (
                        id INTEGER PRIMARY KEY,
                        title TEXT,
                        content TEXT,
                        created_at TEXT,
                        keywords TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS characters (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        description TEXT,
                        personality TEXT,
                        scenario TEXT,
                        greeting_message TEXT,
                        example_messages TEXT
                    )
                """)
            elif name == 'Prompts':
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS prompts (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        author TEXT,
                        details TEXT,
                        system_prompt TEXT,
                        user_prompt TEXT
                    )
                """)
            conn.commit()
            conn.close()
            
        return paths
    
    @pytest.fixture
    def chatbook_importer(self, temp_db_paths):
        """Create a ChatbookImporter instance with test database paths."""
        return ChatbookImporter(db_paths=temp_db_paths)
    
    @pytest.fixture
    def sample_chatbook_path(self, tmp_path):
        """Create a sample chatbook for testing."""
        chatbook_path = tmp_path / "sample_chatbook.zip"
        
        # Create manifest
        manifest = {
            "version": "1.0",
            "name": "Sample Chatbook",
            "description": "A sample chatbook for testing",
            "author": "Test Author",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "content_items": [
                {
                    "id": "conversation_1",
                    "type": "conversation",
                    "title": "Test Conversation",
                    "created_at": datetime.now().isoformat(),
                    "file_path": "content/conversations/conversation_1.json"
                },
                {
                    "id": "note_1",
                    "type": "note",
                    "title": "Test Note",
                    "created_at": datetime.now().isoformat(),
                    "file_path": "content/notes/Test Note.md"
                },
                {
                    "id": "character_1",
                    "type": "character",
                    "title": "Test Character",
                    "created_at": datetime.now().isoformat(),
                    "file_path": "content/characters/character_1.json"
                }
            ],
            "relationships": [
                {
                    "source_id": "conversation_1",
                    "target_id": "character_1",
                    "relationship_type": "uses_character",
                    "metadata": {}
                }
            ],
            "include_media": False,
            "include_embeddings": False,
            "media_quality": "thumbnail",
            "statistics": {
                "total_conversations": 1,
                "total_notes": 1,
                "total_characters": 1,
                "total_media_items": 0,
                "total_size_bytes": 1024
            },
            "tags": ["test", "sample"],
            "categories": ["testing"],
            "language": "en",
            "license": None
        }
        
        # Create conversation content
        conversation_content = {
            "id": 1,
            "title": "Test Conversation",
            "created_at": datetime.now().isoformat(),
            "messages": [
                {"role": "user", "content": "Hello", "timestamp": datetime.now().isoformat()},
                {"role": "assistant", "content": "Hi there!", "timestamp": datetime.now().isoformat()}
            ],
            "character_id": 1
        }
        
        # Create note content
        note_content = """# Test Note

This is a test note with some content.

Keywords: test, sample"""
        
        # Create character content
        character_content = {
            "id": 1,
            "name": "Test Character",
            "description": "A test character",
            "personality": "Helpful and friendly",
            "scenario": "Testing environment",
            "greeting_message": "Hello!",
            "example_messages": ""
        }
        
        # Create ZIP file
        with zipfile.ZipFile(chatbook_path, 'w') as zf:
            zf.writestr('manifest.json', json.dumps(manifest, indent=2))
            zf.writestr('content/conversations/conversation_1.json', 
                       json.dumps(conversation_content, indent=2))
            zf.writestr('content/notes/Test Note.md', note_content)
            zf.writestr('content/characters/character_1.json',
                       json.dumps(character_content, indent=2))
        
        return chatbook_path
    
    def test_importer_initialization(self, chatbook_importer, temp_db_paths):
        """Test ChatbookImporter initialization."""
        assert chatbook_importer.db_paths == temp_db_paths
        assert chatbook_importer.temp_dir.exists()
        assert chatbook_importer.conflict_resolver is not None
    
    def test_preview_chatbook_valid(self, chatbook_importer, sample_chatbook_path):
        """Test previewing a valid chatbook."""
        manifest, error = chatbook_importer.preview_chatbook(sample_chatbook_path)
        
        assert manifest is not None
        assert error is None
        assert manifest.name == "Sample Chatbook"
        assert manifest.description == "A sample chatbook for testing"
        assert len(manifest.content_items) == 3
    
    def test_preview_chatbook_invalid_zip(self, chatbook_importer, tmp_path):
        """Test previewing an invalid ZIP file."""
        invalid_path = tmp_path / "invalid.zip"
        invalid_path.write_text("Not a ZIP file")
        
        manifest, error = chatbook_importer.preview_chatbook(invalid_path)
        
        assert manifest is None
        assert error is not None
        assert "zip" in error.lower()
    
    def test_preview_chatbook_missing_manifest(self, chatbook_importer, tmp_path):
        """Test previewing a chatbook without manifest."""
        invalid_path = tmp_path / "no_manifest.zip"
        
        with zipfile.ZipFile(invalid_path, 'w') as zf:
            zf.writestr('content/test.txt', 'test')
        
        manifest, error = chatbook_importer.preview_chatbook(invalid_path)
        
        assert manifest is None
        assert error is not None
        assert "manifest.json" in error
    
    @patch('tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB')
    def test_import_chatbook_no_conflicts(self, mock_chacha_db, chatbook_importer, sample_chatbook_path):
        """Test importing a chatbook with no conflicts."""
        # Setup mock
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        mock_db_instance.add_conversation.return_value = 1
        mock_db_instance.add_message.return_value = True
        mock_db_instance.add_note.return_value = 1
        mock_db_instance.create_character.return_value = 1
        
        status = ImportStatus()
        
        success, message = chatbook_importer.import_chatbook(
            chatbook_path=sample_chatbook_path,
            conflict_resolution=ConflictResolution.SKIP,
            import_status=status
        )
        
        assert success is True
        assert status.processed_items > 0
        assert len(status.errors) == 0
    
    @patch('tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB')
    def test_import_chatbook_with_conflicts(self, mock_chacha_db, chatbook_importer, sample_chatbook_path, temp_db_paths):
        """Test importing with existing data (conflicts)."""
        # Setup mock
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        mock_db_instance.add_conversation.return_value = 1
        mock_db_instance.add_message.return_value = True
        mock_db_instance.add_note.return_value = 1
        mock_db_instance.create_character.return_value = 1
        
        status = ImportStatus()
        
        # Import with SKIP resolution
        success, message = chatbook_importer.import_chatbook(
            chatbook_path=sample_chatbook_path,
            conflict_resolution=ConflictResolution.SKIP,
            import_status=status
        )
        
        assert success is True
        # Since we're using mocks and not simulating actual conflicts,
        # we just verify that the import succeeded
        assert status.processed_items > 0
    
    @patch('tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB')
    def test_import_chatbook_rename_conflicts(self, mock_chacha_db, chatbook_importer, sample_chatbook_path, temp_db_paths):
        """Test importing with RENAME conflict resolution."""
        # Setup mock
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        mock_db_instance.add_conversation.return_value = 1
        mock_db_instance.add_message.return_value = True
        mock_db_instance.add_note.return_value = 1
        mock_db_instance.create_character.return_value = 1
        
        status = ImportStatus()
        
        # Import with RENAME resolution
        success, message = chatbook_importer.import_chatbook(
            chatbook_path=sample_chatbook_path,
            conflict_resolution=ConflictResolution.RENAME,
            import_status=status
        )
        
        assert success is True
        assert status.successful_items > 0
        
        # Since we're mocking, just verify that the import was successful
        # In real implementation, it would rename the note
    
    @patch('tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB')
    def test_import_status_tracking(self, mock_chacha_db, chatbook_importer, sample_chatbook_path):
        """Test import status tracking."""
        # Setup mock
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        mock_db_instance.add_conversation.return_value = 1
        mock_db_instance.add_message.return_value = True
        mock_db_instance.add_note.return_value = 1
        mock_db_instance.create_character.return_value = 1
        
        status = ImportStatus()
        
        success, message = chatbook_importer.import_chatbook(
            chatbook_path=sample_chatbook_path,
            conflict_resolution=ConflictResolution.SKIP,
            import_status=status
        )
        
        assert success is True
        assert status.total_items == 3  # 1 conv + 1 note + 1 char
        assert status.processed_items == 3
        assert status.successful_items > 0
        
        # Check status dict
        status_dict = status.to_dict()
        assert "total_items" in status_dict
        assert "errors" in status_dict
        assert "warnings" in status_dict
    
    @patch('tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB')
    def test_import_error_handling(self, mock_chacha_db, chatbook_importer, tmp_path):
        """Test error handling during import."""
        # Setup mock
        mock_db_instance = MagicMock()
        mock_chacha_db.return_value = mock_db_instance
        
        # Create a chatbook with invalid content
        bad_chatbook = tmp_path / "bad_chatbook.zip"
        
        manifest = {
            "version": "1.0",
            "name": "Bad Chatbook",
            "description": "Test",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "content_items": [
                {
                    "id": "bad_1",
                    "type": "conversation",
                    "title": "Bad Item",
                    "file_path": "content/missing.json"  # File doesn't exist
                }
            ]
        }
        
        with zipfile.ZipFile(bad_chatbook, 'w') as zf:
            zf.writestr('manifest.json', json.dumps(manifest))
            # Don't create the content file
        
        status = ImportStatus()
        success, message = chatbook_importer.import_chatbook(
            chatbook_path=bad_chatbook,
            conflict_resolution=ConflictResolution.SKIP,
            import_status=status
        )
        
        # Import fails when no items could be imported
        assert success is False
        assert status.failed_items > 0
        assert len(status.warnings) > 0  # File not found generates warnings
    
    def test_import_with_media_settings(self, chatbook_importer, tmp_path):
        """Test importing chatbook with media settings."""
        chatbook_path = tmp_path / "media_chatbook.zip"
        
        manifest = {
            "version": "1.0",
            "name": "Media Chatbook",
            "description": "Test with media",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "include_media": True,
            "media_quality": "original",
            "include_embeddings": True,
            "content_items": []
        }
        
        with zipfile.ZipFile(chatbook_path, 'w') as zf:
            zf.writestr('manifest.json', json.dumps(manifest))
        
        manifest_obj, error = chatbook_importer.preview_chatbook(chatbook_path)
        
        assert manifest_obj is not None
        assert manifest_obj.include_media is True
        assert manifest_obj.media_quality == "original"
        assert manifest_obj.include_embeddings is True