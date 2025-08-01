# test_chatbook_importer.py
# Unit tests for chatbook importer

import pytest
import tempfile
import shutil
import json
import zipfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, call

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.chatbook_models import ContentType, ChatbookVersion, ChatbookManifest
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution


class TestImportStatus:
    """Test ImportStatus class."""
    
    def test_import_status_initialization(self):
        """Test ImportStatus initialization."""
        status = ImportStatus()
        
        assert status.total_items == 0
        assert status.processed_items == 0
        assert status.successful_items == 0
        assert status.failed_items == 0
        assert status.skipped_items == 0
        assert status.errors == []
        assert status.warnings == []
    
    def test_import_status_methods(self):
        """Test ImportStatus methods."""
        status = ImportStatus()
        
        status.add_error("Test error")
        status.add_warning("Test warning")
        
        assert len(status.errors) == 1
        assert status.errors[0] == "Test error"
        assert len(status.warnings) == 1
        assert status.warnings[0] == "Test warning"
    
    def test_import_status_to_dict(self):
        """Test ImportStatus to_dict method."""
        status = ImportStatus()
        status.total_items = 10
        status.processed_items = 8
        status.successful_items = 6
        status.failed_items = 1
        status.skipped_items = 1
        status.add_error("Import failed")
        status.add_warning("Duplicate found")
        
        data = status.to_dict()
        
        assert data["total_items"] == 10
        assert data["processed_items"] == 8
        assert data["successful_items"] == 6
        assert data["failed_items"] == 1
        assert data["skipped_items"] == 1
        assert data["errors"] == ["Import failed"]
        assert data["warnings"] == ["Duplicate found"]


class TestChatbookImporter:
    """Test ChatbookImporter functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_dbs(self):
        """Create mock database instances."""
        # Mock ChaChaNotes DB
        mock_chachanotes = Mock()
        mock_chachanotes.add_conversation.return_value = 100  # New conversation ID
        mock_chachanotes.add_message.return_value = True
        mock_chachanotes.add_note.return_value = 200  # New note ID
        mock_chachanotes.create_character.return_value = 300  # New character ID
        
        # Mock Prompts DB
        mock_prompts = Mock()
        mock_prompts.add_prompt.return_value = (400, "added", "Success")  # New prompt ID
        
        return {
            'chachanotes': mock_chachanotes,
            'prompts': mock_prompts
        }
    
    @pytest.fixture
    def importer(self, temp_dir):
        """Create a ChatbookImporter instance."""
        db_paths = {
            'chachanotes': str(temp_dir / 'chachanotes.db'),
            'prompts': str(temp_dir / 'prompts.db'),
            'media': str(temp_dir / 'media.db')
        }
        return ChatbookImporter(db_paths)
    
    @pytest.fixture
    def sample_chatbook(self, temp_dir):
        """Create a sample chatbook ZIP file."""
        # Create temporary directory structure
        chatbook_dir = temp_dir / "chatbook"
        chatbook_dir.mkdir()
        
        # Create manifest
        manifest = {
            "version": "1.0",
            "name": "Test Chatbook",
            "description": "A test chatbook for unit tests",
            "author": "Test Author",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "content_items": [
                {
                    "id": "conv_1",
                    "type": "conversation",
                    "title": "Test Conversation",
                    "file_path": "content/conversations/conversation_conv_1.json"
                },
                {
                    "id": "note_1",
                    "type": "note",
                    "title": "Test Note",
                    "file_path": "content/notes/Test Note.md"
                }
            ],
            "relationships": [],
            "include_media": False,
            "include_embeddings": False,
            "media_quality": "thumbnail",
            "statistics": {
                "total_conversations": 1,
                "total_notes": 1,
                "total_characters": 0,
                "total_media_items": 0,
                "total_size_bytes": 0
            },
            "tags": ["test"],
            "categories": ["testing"],
            "language": "en"
        }
        
        with open(chatbook_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create content directory
        content_dir = chatbook_dir / "content"
        content_dir.mkdir()
        
        # Create conversation
        conv_dir = content_dir / "conversations"
        conv_dir.mkdir()
        conv_data = {
            "id": "conv_1",
            "name": "Test Conversation",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "character_id": None,
            "messages": [
                {"id": 1, "role": "user", "content": "Hello", "timestamp": datetime.now().isoformat()},
                {"id": 2, "role": "assistant", "content": "Hi!", "timestamp": datetime.now().isoformat()}
            ]
        }
        with open(conv_dir / "conversation_conv_1.json", 'w') as f:
            json.dump(conv_data, f, indent=2)
        
        # Create note
        notes_dir = content_dir / "notes"
        notes_dir.mkdir()
        note_content = """---
id: note_1
title: Test Note
created_at: 2024-01-01T12:00:00
---

# Test Note

This is test content."""
        with open(notes_dir / "Test Note.md", 'w') as f:
            f.write(note_content)
        
        # Create README
        with open(chatbook_dir / "README.md", 'w') as f:
            f.write("# Test Chatbook\n\nTest description")
        
        # Create ZIP file
        zip_path = temp_dir / "test_chatbook.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in chatbook_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(chatbook_dir)
                    zf.write(file_path, arcname)
        
        return zip_path
    
    def test_importer_initialization(self, temp_dir):
        """Test ChatbookImporter initialization."""
        db_paths = {
            'chachanotes': str(temp_dir / 'chachanotes.db'),
            'prompts': str(temp_dir / 'prompts.db')
        }
        
        importer = ChatbookImporter(db_paths)
        
        assert importer.db_paths == db_paths
        assert importer.temp_dir.exists()
        assert importer.conflict_resolver is not None
    
    def test_preview_chatbook_success(self, importer, sample_chatbook):
        """Test successful chatbook preview."""
        manifest, error = importer.preview_chatbook(sample_chatbook)
        
        assert error is None
        assert manifest is not None
        assert manifest.name == "Test Chatbook"
        assert manifest.description == "A test chatbook for unit tests"
        assert len(manifest.content_items) == 2
        assert manifest.total_conversations == 1
        assert manifest.total_notes == 1
    
    def test_preview_chatbook_invalid_format(self, importer, temp_dir):
        """Test preview with invalid file format."""
        invalid_file = temp_dir / "not_a_zip.txt"
        invalid_file.write_text("Not a ZIP file")
        
        manifest, error = importer.preview_chatbook(invalid_file)
        
        assert manifest is None
        assert error is not None
        assert "Unsupported chatbook format" in error
    
    def test_preview_chatbook_missing_manifest(self, importer, temp_dir):
        """Test preview with missing manifest."""
        # Create ZIP without manifest
        zip_path = temp_dir / "no_manifest.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("README.md", "# Empty")
        
        manifest, error = importer.preview_chatbook(zip_path)
        
        assert manifest is None
        assert error is not None
        assert "manifest.json not found" in error
    
    @patch('tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB')
    def test_import_conversations(self, mock_db_class, importer, sample_chatbook, mock_dbs):
        """Test importing conversations."""
        mock_db_class.return_value = mock_dbs['chachanotes']
        
        # Import chatbook
        success, status = importer.import_chatbook(
            chatbook_path=sample_chatbook,
            content_selections={ContentType.CONVERSATION: ["conv_1"]},
            conflict_resolution=ConflictResolution.RENAME,
            prefix_imported=True
        )
        
        assert success is True
        assert status.successful_items == 1
        assert status.failed_items == 0
        
        # Verify conversation was created
        mock_dbs['chachanotes'].add_conversation.assert_called_once()
        call_args = mock_dbs['chachanotes'].add_conversation.call_args
        assert call_args[1]["conversation_name"] == "[Imported] Test Conversation"
        
        # Verify messages were added
        assert mock_dbs['chachanotes'].add_message.call_count == 2
    
    @patch('tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB')
    def test_import_notes(self, mock_db_class, importer, sample_chatbook, mock_dbs):
        """Test importing notes."""
        mock_db_class.return_value = mock_dbs['chachanotes']
        
        # Import chatbook
        success, status = importer.import_chatbook(
            chatbook_path=sample_chatbook,
            content_selections={ContentType.NOTE: ["note_1"]},
            conflict_resolution=ConflictResolution.RENAME,
            prefix_imported=True
        )
        
        assert success is True
        assert status.successful_items == 1
        assert status.failed_items == 0
        
        # Verify note was created
        mock_dbs['chachanotes'].add_note.assert_called_once()
        call_args = mock_dbs['chachanotes'].add_note.call_args
        assert call_args[1]["title"] == "[Imported] Test Note"
        assert "This is test content" in call_args[1]["content"]
    
    def test_import_with_conflict_skip(self, importer, sample_chatbook):
        """Test import with skip conflict resolution."""
        # Mock existing conversation
        with patch('tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB') as mock_db_class:
            mock_db = Mock()
            # Simulate existing conversation by having resolver return SKIP
            mock_db_class.return_value = mock_db
            
            with patch.object(importer.conflict_resolver, 'resolve_conversation_conflict', 
                            return_value=ConflictResolution.SKIP):
                success, status = importer.import_chatbook(
                    chatbook_path=sample_chatbook,
                    content_selections={ContentType.CONVERSATION: ["conv_1"]},
                    conflict_resolution=ConflictResolution.SKIP
                )
            
            assert status.skipped_items == 1
            assert status.successful_items == 0
            # Verify no conversation was created
            mock_db.add_conversation.assert_not_called()
    
    def test_import_all_content(self, importer, sample_chatbook):
        """Test importing all content when no selections specified."""
        with patch('tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB') as mock_db_class:
            mock_db = Mock()
            mock_db.add_conversation.return_value = 100
            mock_db.add_message.return_value = True
            mock_db.add_note.return_value = 200
            mock_db_class.return_value = mock_db
            
            # Import without specifying content_selections
            success, status = importer.import_chatbook(
                chatbook_path=sample_chatbook,
                content_selections=None,  # Import everything
                conflict_resolution=ConflictResolution.RENAME
            )
            
            assert success is True
            assert status.total_items == 2  # 1 conversation + 1 note
            assert status.successful_items == 2
    
    def test_import_version_warning(self, importer, temp_dir):
        """Test import with incompatible version warning."""
        # Create chatbook with future version
        chatbook_dir = temp_dir / "chatbook"
        chatbook_dir.mkdir()
        
        manifest = {
            "version": "2.0",  # Future version
            "name": "Future Chatbook",
            "description": "From the future",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "content_items": [],
            "relationships": []
        }
        
        with open(chatbook_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        # Create ZIP
        zip_path = temp_dir / "future_chatbook.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(chatbook_dir / "manifest.json", "manifest.json")
        
        # Import
        success, status = importer.import_chatbook(zip_path)
        
        # Should still succeed but with warning
        assert len(status.warnings) > 0
        assert "may not be fully compatible" in status.warnings[0]
    
    def test_import_error_handling(self, importer, sample_chatbook):
        """Test error handling during import."""
        with patch('tldw_chatbook.Chatbooks.chatbook_importer.CharactersRAGDB') as mock_db_class:
            # Mock database to raise exception
            mock_db = Mock()
            mock_db.add_conversation.side_effect = Exception("Database error")
            mock_db_class.return_value = mock_db
            
            success, status = importer.import_chatbook(
                chatbook_path=sample_chatbook,
                content_selections={ContentType.CONVERSATION: ["conv_1"]}
            )
            
            assert success is False
            assert status.failed_items == 1
            assert len(status.errors) > 0
            assert "Database error" in status.errors[0]
    
    def test_generate_unique_names(self, importer):
        """Test unique name generation methods."""
        mock_db = Mock()
        
        # Test conversation name generation
        # First call returns True (exists), second returns False
        mock_db.get_conversation_by_name = Mock(side_effect=[True, False])
        name = importer._generate_unique_name("Test Conv", mock_db)
        assert name == "Test Conv (1)"
        
        # Test note title generation
        mock_db.get_note_by_title = Mock(side_effect=[True, True, False])
        title = importer._generate_unique_note_title("Test Note", mock_db)
        assert title == "Test Note (2)"
        
        # Test character name generation
        mock_db.get_character_by_name = Mock(side_effect=[True, False])
        char_name = importer._generate_unique_character_name("Test Char", mock_db)
        assert char_name == "Test Char (1)"
    
    @patch('tldw_chatbook.Chatbooks.chatbook_importer.PromptsDatabase')
    def test_import_prompts(self, mock_db_class, importer, temp_dir):
        """Test importing prompts."""
        # Create chatbook with prompt
        chatbook_dir = temp_dir / "chatbook"
        chatbook_dir.mkdir()
        content_dir = chatbook_dir / "content" / "prompts"
        content_dir.mkdir(parents=True)
        
        prompt_data = {
            "id": "prompt_1",
            "name": "Test Prompt",
            "description": "A test prompt",
            "content": "You are a helpful assistant."
        }
        with open(content_dir / "prompt_prompt_1.json", 'w') as f:
            json.dump(prompt_data, f)
        
        manifest = {
            "version": "1.0",
            "name": "Prompt Chatbook",
            "description": "Contains prompts",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "content_items": [{
                "id": "prompt_1",
                "type": "prompt",
                "title": "Test Prompt",
                "file_path": "content/prompts/prompt_prompt_1.json"
            }],
            "relationships": []
        }
        with open(chatbook_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        # Create ZIP
        zip_path = temp_dir / "prompt_chatbook.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for file_path in chatbook_dir.rglob('*'):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(chatbook_dir))
        
        # Setup mock
        mock_db = Mock()
        mock_db.add_prompt.return_value = (1, "added", "Success")
        mock_db_class.return_value = mock_db
        
        # Import
        success, status = importer.import_chatbook(
            chatbook_path=zip_path,
            content_selections={ContentType.PROMPT: ["prompt_1"]},
            prefix_imported=True
        )
        
        assert success is True
        assert status.successful_items == 1
        
        # Verify prompt was created
        mock_db.add_prompt.assert_called_once()
        call_args = mock_db.add_prompt.call_args
        assert call_args[1]["name"] == "[Imported] Test Prompt"
        assert call_args[1]["system_prompt"] == "You are a helpful assistant."