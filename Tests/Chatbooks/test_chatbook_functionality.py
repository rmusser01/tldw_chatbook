# test_chatbook_functionality.py
# Description: Tests for chatbook functionality
#
"""
Chatbook Functionality Tests
----------------------------

Tests for the chatbook creation, import, and management features.
"""

import pytest
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from tldw_chatbook.Chatbooks import (
    ChatbookCreator,
    ChatbookImporter,
    ChatbookManifest,
    ChatbookError,
    ChatbookErrorType
)
from tldw_chatbook.Chatbooks.chatbook_models import ContentType, ChatbookVersion
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution


class TestChatbookCreation:
    """Test chatbook creation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_paths = {
            "ChaChaNotes": ":memory:",
            "Prompts": ":memory:",
            "Media": ":memory:"
        }
        self.creator = ChatbookCreator(self.db_paths)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_empty_chatbook(self):
        """Test creating an empty chatbook."""
        output_path = Path(self.temp_dir) / "test_chatbook.zip"
        
        success, message, info = self.creator.create_chatbook(
            name="Test Chatbook",
            description="A test chatbook",
            content_selections={},
            output_path=output_path,
            author="Test Author"
        )
        
        assert success is True
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_create_chatbook_with_invalid_path(self):
        """Test creating chatbook with invalid output path."""
        invalid_path = Path("/invalid/path/chatbook.zip")
        
        success, message, info = self.creator.create_chatbook(
            name="Test Chatbook",
            description="A test chatbook",
            content_selections={},
            output_path=invalid_path
        )
        
        assert success is False
        assert "error" in message.lower()


class TestChatbookImport:
    """Test chatbook import functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_paths = {
            "ChaChaNotes": ":memory:",
            "Prompts": ":memory:",
            "Media": ":memory:"
        }
        self.importer = ChatbookImporter(self.db_paths)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_preview_invalid_file(self):
        """Test previewing an invalid chatbook file."""
        invalid_file = Path(self.temp_dir) / "invalid.txt"
        invalid_file.write_text("This is not a chatbook")
        
        manifest, error = self.importer.preview_chatbook(invalid_file)
        
        assert manifest is None
        assert error is not None
        assert "unsupported" in error.lower()


class TestChatbookModels:
    """Test chatbook data models."""
    
    def test_manifest_creation(self):
        """Test creating a chatbook manifest."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test Manifest",
            description="Test description",
            author="Test Author"
        )
        
        assert manifest.name == "Test Manifest"
        assert manifest.version == ChatbookVersion.V1
        assert isinstance(manifest.created_at, datetime)
    
    def test_manifest_serialization(self):
        """Test manifest serialization to dict."""
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Test Manifest",
            description="Test description"
        )
        
        data = manifest.to_dict()
        
        assert data["name"] == "Test Manifest"
        assert data["version"] == "1.0"
        assert "created_at" in data
        assert "content_items" in data
    
    def test_manifest_deserialization(self):
        """Test manifest deserialization from dict."""
        data = {
            "version": "1.0",
            "name": "Test Manifest",
            "description": "Test description",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        manifest = ChatbookManifest.from_dict(data)
        
        assert manifest.name == "Test Manifest"
        assert manifest.version == ChatbookVersion.V1


class TestErrorHandling:
    """Test error handling functionality."""
    
    def test_chatbook_error_creation(self):
        """Test creating a chatbook error."""
        error = ChatbookError(
            error_type=ChatbookErrorType.FILE_NOT_FOUND,
            message="Test file not found",
            details="Looking for test.zip",
            recovery_suggestions=["Check the file path"]
        )
        
        assert error.error_type == ChatbookErrorType.FILE_NOT_FOUND
        assert "Test file not found" in error.get_user_message()
        assert "Check the file path" in error.get_full_message()
    
    def test_error_handler_file_not_found(self):
        """Test error handler for file not found."""
        from tldw_chatbook.Chatbooks.error_handler import ChatbookErrorHandler
        
        original_error = FileNotFoundError("test.zip not found")
        chatbook_error = ChatbookErrorHandler.handle_error(
            original_error,
            "loading chatbook"
        )
        
        assert chatbook_error.error_type == ChatbookErrorType.FILE_NOT_FOUND
        assert "loading chatbook" in chatbook_error.message
    
    def test_error_handler_permission_error(self):
        """Test error handler for permission errors."""
        from tldw_chatbook.Chatbooks.error_handler import ChatbookErrorHandler
        
        original_error = PermissionError("Access denied")
        chatbook_error = ChatbookErrorHandler.handle_error(
            original_error,
            "exporting chatbook"
        )
        
        assert chatbook_error.error_type == ChatbookErrorType.PERMISSION_ERROR
        assert "exporting chatbook" in chatbook_error.message
        assert len(chatbook_error.recovery_suggestions) > 0


class TestConflictResolution:
    """Test conflict resolution functionality."""
    
    def test_conflict_resolution_enum(self):
        """Test conflict resolution enum values."""
        assert ConflictResolution.SKIP.value == "skip"
        assert ConflictResolution.RENAME.value == "rename"
        assert ConflictResolution.REPLACE.value == "replace"
        assert ConflictResolution.MERGE.value == "merge"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])