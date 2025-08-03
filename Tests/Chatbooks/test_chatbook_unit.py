# test_chatbook_unit.py
# Description: Comprehensive unit tests for chatbook functionality
#
"""
Chatbook Unit Tests
-------------------

Detailed unit tests for chatbook components with mocked dependencies.
"""

import pytest
import json
import zipfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, call
import sqlite3

from tldw_chatbook.Chatbooks import (
    ChatbookCreator, ChatbookImporter, ChatbookError, 
    ChatbookErrorType, ChatbookErrorHandler
)
from tldw_chatbook.Chatbooks.chatbook_models import (
    ContentType, ContentItem, ChatbookManifest, ChatbookVersion,
    Relationship, ChatbookContent
)
from tldw_chatbook.Chatbooks.conflict_resolver import (
    ConflictResolver, ConflictResolution
)
from tldw_chatbook.Chatbooks.error_handler import safe_chatbook_operation
from Tests.Chatbooks.factories import (
    CharacterFactory, ConversationFactory, NoteFactory,
    MediaFactory, PromptFactory, ManifestFactory
)


class TestChatbookCreator:
    """Test ChatbookCreator functionality."""
    
    def test_init(self, mock_db_paths):
        """Test creator initialization."""
        creator = ChatbookCreator(mock_db_paths)
        
        assert creator.db_paths == mock_db_paths
        assert creator.temp_dir.exists()
        assert len(creator.missing_dependencies) == 0
        assert len(creator.auto_included_characters) == 0
    
    def test_create_empty_chatbook(self, chatbook_creator, temp_dir):
        """Test creating an empty chatbook."""
        output_path = temp_dir / "empty.zip"
        
        success, message, info = chatbook_creator.create_chatbook(
            name="Empty Chatbook",
            description="An empty test chatbook",
            content_selections={},
            output_path=output_path,
            author="Unit Test"
        )
        
        assert success is True
        assert "successfully" in message.lower()
        assert output_path.exists()
        assert info["missing_dependencies"] == []
        assert info["auto_included"] == []
        
        # Verify ZIP contents
        with zipfile.ZipFile(output_path, 'r') as zf:
            files = zf.namelist()
            assert "manifest.json" in files
            assert "README.md" in files
    
    def test_create_chatbook_with_conversations(self, chatbook_creator, temp_dir, mock_db_paths):
        """Test creating chatbook with conversations."""
        # Mock database interactions
        with patch.object(chatbook_creator, '_collect_conversations') as mock_collect:
            output_path = temp_dir / "conversations.zip"
            
            success, message, info = chatbook_creator.create_chatbook(
                name="Conversation Chatbook",
                description="Test with conversations",
                content_selections={
                    ContentType.CONVERSATION: ["conv1", "conv2"]
                },
                output_path=output_path
            )
            
            assert success is True
            assert mock_collect.called
            assert mock_collect.call_args[0][0] == ["conv1", "conv2"]
    
    def test_create_chatbook_with_all_content_types(self, chatbook_creator, temp_dir):
        """Test creating chatbook with all content types."""
        with patch.object(chatbook_creator, '_collect_conversations'), \
             patch.object(chatbook_creator, '_collect_notes'), \
             patch.object(chatbook_creator, '_collect_characters'), \
             patch.object(chatbook_creator, '_collect_media'), \
             patch.object(chatbook_creator, '_collect_prompts'):
            
            output_path = temp_dir / "all_content.zip"
            
            success, message, info = chatbook_creator.create_chatbook(
                name="Complete Chatbook",
                description="Test with all content types",
                content_selections={
                    ContentType.CONVERSATION: ["conv1"],
                    ContentType.NOTE: ["note1"],
                    ContentType.CHARACTER: ["char1"],
                    ContentType.MEDIA: ["media1"],
                    ContentType.PROMPT: ["prompt1"]
                },
                output_path=output_path,
                include_media=True,
                include_embeddings=True
            )
            
            assert success is True
    
    def test_create_chatbook_with_tags_and_categories(self, chatbook_creator, temp_dir):
        """Test creating chatbook with metadata."""
        output_path = temp_dir / "tagged.zip"
        
        success, message, info = chatbook_creator.create_chatbook(
            name="Tagged Chatbook",
            description="Test with tags",
            content_selections={},
            output_path=output_path,
            tags=["test", "unit", "chatbook"],
            categories=["testing", "development"]
        )
        
        assert success is True
        
        # Verify manifest contains tags
        with zipfile.ZipFile(output_path, 'r') as zf:
            manifest_data = json.loads(zf.read("manifest.json"))
            assert manifest_data["tags"] == ["test", "unit", "chatbook"]
            assert manifest_data["categories"] == ["testing", "development"]
    
    def test_create_chatbook_invalid_path(self, chatbook_creator):
        """Test creating chatbook with invalid output path."""
        invalid_path = Path("/invalid/path/that/does/not/exist/chatbook.zip")
        
        success, message, info = chatbook_creator.create_chatbook(
            name="Invalid Path",
            description="Test",
            content_selections={},
            output_path=invalid_path
        )
        
        assert success is False
        assert "error" in message.lower()
    
    def test_collect_conversations_with_character_dependency(self, chatbook_creator, temp_dir):
        """Test collecting conversations with character dependencies."""
        # Mock database
        mock_db = Mock()
        mock_db.get_conversation_by_id.return_value = {
            'id': 'conv1',
            'title': 'Test Conv',
            'created_at': datetime.now(),
            'character_id': 123
        }
        mock_db.get_messages_for_conversation.return_value = []
        
        with patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB', return_value=mock_db):
            work_dir = temp_dir / "work"
            work_dir.mkdir()
            manifest = ManifestFactory.create()
            content = ChatbookContent()
            
            chatbook_creator._collect_conversations(
                ["conv1"],
                work_dir,
                manifest,
                content,
                auto_include_dependencies=True
            )
            
            # Should track character dependency
            assert 123 in chatbook_creator.missing_dependencies or \
                   123 in chatbook_creator.auto_included_characters
    
    def test_generate_unique_name_methods(self, chatbook_importer):
        """Test unique name generation for conflicts."""
        # These methods are now implemented to use actual DB checks
        # but we can test the fallback behavior
        mock_db = Mock()
        # Mock returns list of conversations (non-empty list is truthy, empty list is falsy)
        mock_db.get_conversation_by_name.side_effect = [
            [{"id": 1, "name": "Test"}],  # First call returns existing conversation
            [{"id": 2, "name": "Test (1)"}],  # Second call returns existing conversation
            []  # Third call returns empty list (no conversation found)
        ]
        
        unique_name = chatbook_importer._generate_unique_name("Test", mock_db)
        assert unique_name == "Test (3)"
        assert mock_db.get_conversation_by_name.call_count == 3


class TestChatbookImporter:
    """Test ChatbookImporter functionality."""
    
    def test_init(self, mock_db_paths):
        """Test importer initialization."""
        importer = ChatbookImporter(mock_db_paths)
        
        assert importer.db_paths == mock_db_paths
        assert importer.temp_dir.exists()
        assert isinstance(importer.conflict_resolver, ConflictResolver)
    
    def test_preview_valid_chatbook(self, chatbook_importer, sample_chatbook_zip):
        """Test previewing a valid chatbook."""
        manifest, error = chatbook_importer.preview_chatbook(sample_chatbook_zip)
        
        assert manifest is not None
        assert error is None
        assert manifest.name == "Test Chatbook"
        assert manifest.version == ChatbookVersion.V1
    
    def test_preview_invalid_file(self, chatbook_importer, temp_dir):
        """Test previewing an invalid file."""
        invalid_file = temp_dir / "not_a_zip.txt"
        invalid_file.write_text("This is not a ZIP file")
        
        manifest, error = chatbook_importer.preview_chatbook(invalid_file)
        
        assert manifest is None
        assert error is not None
        assert "unsupported" in error.lower()
    
    def test_preview_missing_manifest(self, chatbook_importer, temp_dir):
        """Test previewing ZIP without manifest."""
        bad_zip = temp_dir / "no_manifest.zip"
        with zipfile.ZipFile(bad_zip, 'w') as zf:
            zf.writestr("README.md", "# No Manifest Here")
        
        manifest, error = chatbook_importer.preview_chatbook(bad_zip)
        
        assert manifest is None
        assert error is not None
        assert "manifest" in error.lower()
    
    def test_import_chatbook_all_content(self, chatbook_importer, sample_chatbook_zip):
        """Test importing all content from a chatbook."""
        with patch.object(chatbook_importer, '_import_conversations') as mock_conv, \
             patch.object(chatbook_importer, '_import_notes') as mock_notes, \
             patch.object(chatbook_importer, '_import_characters') as mock_chars:
            
            success, message = chatbook_importer.import_chatbook(
                chatbook_path=sample_chatbook_zip,
                conflict_resolution=ConflictResolution.SKIP
            )
            
            assert mock_conv.called
            assert mock_notes.called
            assert mock_chars.called
    
    def test_import_with_conflict_resolution(self, chatbook_importer, sample_chatbook_zip):
        """Test import with different conflict resolution strategies."""
        # Test each resolution strategy
        for strategy in ConflictResolution:
            with patch.object(chatbook_importer, '_import_conversations'):
                success, message = chatbook_importer.import_chatbook(
                    chatbook_path=sample_chatbook_zip,
                    conflict_resolution=strategy
                )
    
    def test_import_selective_content(self, chatbook_importer, sample_chatbook_zip):
        """Test importing only selected content types."""
        with patch.object(chatbook_importer, '_import_conversations') as mock_conv, \
             patch.object(chatbook_importer, '_import_notes') as mock_notes:
            
            # Only import conversations
            success, message = chatbook_importer.import_chatbook(
                chatbook_path=sample_chatbook_zip,
                content_selections={
                    ContentType.CONVERSATION: ["conv1"]
                }
            )
            
            assert mock_conv.called
            assert not mock_notes.called


class TestChatbookModels:
    """Test chatbook data models."""
    
    def test_content_item_creation(self):
        """Test ContentItem creation and methods."""
        item = ContentItem(
            id="test1",
            type=ContentType.NOTE,
            title="Test Note",
            description="A test note",
            tags=["test", "unit"]
        )
        
        assert item.id == "test1"
        assert item.type == ContentType.NOTE
        assert "test" in item.tags
    
    def test_content_item_serialization(self):
        """Test ContentItem serialization."""
        item = ContentItem(
            id="test1",
            type=ContentType.CHARACTER,
            title="Test Character",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        data = item.to_dict()
        assert data["id"] == "test1"
        assert data["type"] == "character"
        assert "created_at" in data
        
        # Test deserialization
        restored = ContentItem.from_dict(data)
        assert restored.id == item.id
        assert restored.type == item.type
    
    def test_manifest_statistics(self):
        """Test manifest statistics calculation."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Stats Test",
            description="Testing statistics"
        )
        
        # Add various content items
        for i in range(3):
            manifest.content_items.append(
                ContentItem(f"conv{i}", ContentType.CONVERSATION, f"Conv {i}")
            )
        
        for i in range(2):
            manifest.content_items.append(
                ContentItem(f"note{i}", ContentType.NOTE, f"Note {i}")
            )
        
        manifest.content_items.append(
            ContentItem("char1", ContentType.CHARACTER, "Character 1")
        )
        
        # Update stats manually (normally done by creator)
        manifest.total_conversations = 3
        manifest.total_notes = 2
        manifest.total_characters = 1
        
        data = manifest.to_dict()
        assert data["statistics"]["total_conversations"] == 3
        assert data["statistics"]["total_notes"] == 2
        assert data["statistics"]["total_characters"] == 1
    
    def test_relationship_creation(self):
        """Test Relationship model."""
        rel = Relationship(
            source_id="conv1",
            target_id="char1",
            relationship_type="uses_character",
            metadata={"strength": "primary"}
        )
        
        data = rel.to_dict()
        assert data["source_id"] == "conv1"
        assert data["target_id"] == "char1"
        assert data["relationship_type"] == "uses_character"
        assert data["metadata"]["strength"] == "primary"


class TestConflictResolver:
    """Test conflict resolution functionality."""
    
    def test_conversation_conflict_skip(self):
        """Test skipping conversation conflicts."""
        resolver = ConflictResolver()
        
        existing = {"conversation_name": "Test Conv", "message_count": 10}
        incoming = {"name": "Test Conv", "messages": [1, 2, 3]}
        
        resolution = resolver.resolve_conversation_conflict(
            existing, incoming, ConflictResolution.SKIP
        )
        
        assert resolution == ConflictResolution.SKIP
    
    def test_note_conflict_rename(self):
        """Test renaming note conflicts."""
        resolver = ConflictResolver()
        
        existing = {"title": "My Note", "content": "Original"}
        incoming = {"title": "My Note", "content": "New"}
        
        resolution = resolver.resolve_note_conflict(
            existing, incoming, ConflictResolution.RENAME
        )
        
        assert resolution == ConflictResolution.RENAME
    
    def test_character_conflict_replace(self):
        """Test replacing character conflicts."""
        resolver = ConflictResolver()
        
        existing = {"name": "Alice", "description": "Old"}
        incoming = {"name": "Alice", "description": "New"}
        
        resolution = resolver.resolve_character_conflict(
            existing, incoming, ConflictResolution.REPLACE
        )
        
        assert resolution == ConflictResolution.REPLACE
    
    def test_merge_notes(self):
        """Test merging note content."""
        resolver = ConflictResolver()
        
        existing = {
            "title": "Research",
            "content": "Original research notes",
            "keywords": "research,science"
        }
        
        incoming = {
            "title": "Research",
            "content": "Additional findings",
            "tags": ["research", "new"]
        }
        
        merged = resolver.merge_notes(existing, incoming)
        
        assert "Original research notes" in merged["content"]
        assert "Additional findings" in merged["content"]
        assert "Imported from chatbook" in merged["content"]
    
    def test_conflict_with_callback(self):
        """Test conflict resolution with user callback."""
        def mock_callback(conflict_info):
            # Simulate user choosing to rename
            return ConflictResolution.RENAME
        
        resolver = ConflictResolver(ask_callback=mock_callback)
        
        resolution = resolver.resolve_conversation_conflict(
            {"conversation_name": "Test"},
            {"name": "Test"},
            ConflictResolution.ASK
        )
        
        assert resolution == ConflictResolution.RENAME


class TestErrorHandling:
    """Test error handling functionality."""
    
    def test_chatbook_error_creation(self):
        """Test ChatbookError creation and methods."""
        error = ChatbookError(
            error_type=ChatbookErrorType.FILE_NOT_FOUND,
            message="Chatbook not found",
            details="Looking for test.zip in /tmp",
            recovery_suggestions=["Check file path", "Verify file exists"]
        )
        
        assert error.error_type == ChatbookErrorType.FILE_NOT_FOUND
        assert error.get_user_message() == "Chatbook not found"
        
        full_msg = error.get_full_message()
        assert "Details:" in full_msg
        assert "Check file path" in full_msg
    
    def test_error_handler_classification(self):
        """Test error type classification."""
        # File not found
        error = ChatbookErrorHandler.handle_error(
            FileNotFoundError("test.zip"),
            "loading chatbook"
        )
        assert error.error_type == ChatbookErrorType.FILE_NOT_FOUND
        
        # Permission error
        error = ChatbookErrorHandler.handle_error(
            PermissionError("Access denied"),
            "exporting"
        )
        assert error.error_type == ChatbookErrorType.PERMISSION_ERROR
        
        # Database error
        error = ChatbookErrorHandler.handle_error(
            sqlite3.DatabaseError("Database locked"),
            "querying"
        )
        assert error.error_type == ChatbookErrorType.DATABASE_ERROR
    
    def test_safe_operation_decorator(self):
        """Test safe_chatbook_operation decorator."""
        @safe_chatbook_operation("test operation")
        def risky_operation():
            raise ValueError("Something went wrong")
        
        with pytest.raises(ChatbookError) as exc_info:
            risky_operation()
        
        assert "test operation" in str(exc_info.value)
    
    def test_error_handler_with_context(self):
        """Test error handling with context."""
        error = ChatbookErrorHandler.handle_error(
            FileNotFoundError("missing.zip"),
            "importing chatbook",
            context={"path": "/tmp/missing.zip", "size": "unknown"}
        )
        
        full_msg = error.get_full_message()
        assert "path: /tmp/missing.zip" in full_msg
        assert "size: unknown" in full_msg


class TestCharacterDependencyTracking:
    """Test character dependency tracking."""
    
    def test_track_missing_dependencies(self, chatbook_creator):
        """Test tracking missing character dependencies."""
        chatbook_creator._selected_characters = set()
        
        # Simulate adding a dependency that's not selected
        with patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB'):
            chatbook_creator._add_character_dependency(
                123,
                ManifestFactory.create(),
                ChatbookContent(),
                Path("/tmp"),
                auto_include=False
            )
        
        assert 123 in chatbook_creator.missing_dependencies
        assert 123 not in chatbook_creator.auto_included_characters
    
    def test_auto_include_dependencies(self, chatbook_creator, temp_dir):
        """Test auto-including character dependencies."""
        mock_db = Mock()
        mock_db.get_character_card_by_id.return_value = {
            'id': 123,
            'name': 'Dependent Character',
            'description': 'Auto-included'
        }
        
        with patch('tldw_chatbook.Chatbooks.chatbook_creator.CharactersRAGDB', return_value=mock_db):
            manifest = ManifestFactory.create()
            content = ChatbookContent()
            work_dir = temp_dir / "work"
            work_dir.mkdir()
            
            chatbook_creator._add_character_dependency(
                123,
                manifest,
                content,
                work_dir,
                auto_include=True
            )
            
            assert 123 in chatbook_creator.auto_included_characters
            assert 123 not in chatbook_creator.missing_dependencies
            assert len(content.characters) == 1
            assert any(item.id == "123" for item in manifest.content_items)