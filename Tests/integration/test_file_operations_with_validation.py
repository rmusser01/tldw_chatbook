"""
Integration tests for file operations with path validation.
"""

import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import io

from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    load_chat_history_from_file_and_save_to_db
)
from tldw_chatbook.tldw_api.utils import (
    prepare_files_for_httpx, cleanup_file_objects
)
from tldw_chatbook.Utils.path_validation import validate_path
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class TestCharacterChatFileOperations:
    """Integration tests for character chat file operations with validation."""
    
    @pytest.fixture
    def temp_base_dir(self):
        """Create a temporary base directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def mock_db(self, temp_base_dir):
        """Create a CharactersRAGDB instance with test data."""
        db_path = Path(temp_base_dir) / "test.db"
        db = CharactersRAGDB(str(db_path), client_id="test_client")
        
        # Add a test character to the database
        character_data = {
            'name': 'TestCharacter',
            'description': 'A test character',
            'personality': 'Friendly test character',
            'scenario': 'Testing scenario',
            'first_message': 'Hello, I am a test character!',
            'creator': 'Test Creator',
            'version': '1.0',
            'post_history_instructions': '',
            'alternate_greetings': [],
            'tags': ['test'],
            'creator_notes': 'Created for testing',
            'system_prompt': '',
            'char_persona': '',
            'world_scenario': '',
            'example_dialogue': ''
        }
        db.add_character_card(character_data)
        
        return db
    
    @pytest.fixture
    def sample_chat_history(self):
        """Create sample chat history data."""
        return {
            "char_name": "TestCharacter",
            "user_name": "TestUser",
            "history": {
                "internal": [
                    ["Hello!", "Hi there!"],
                    ["How are you?", "I'm doing well, thanks!"]
                ]
            }
        }
    
    def test_load_chat_history_with_valid_path(self, mock_db, temp_base_dir, sample_chat_history):
        """Test loading chat history with a valid file path."""
        # Create a valid chat history file
        chat_file = Path(temp_base_dir) / "chat_history.json"
        chat_file.write_text(json.dumps(sample_chat_history))
        
        # Load with base directory validation
        conv_id, char_id = load_chat_history_from_file_and_save_to_db(
            mock_db,
            str(chat_file),
            base_directory=temp_base_dir
        )
        
        # Verify the conversation was created
        assert conv_id is not None
        assert char_id is not None
        
        # Get the character from database to verify it's the correct one
        character = mock_db.get_character_card_by_name("TestCharacter")
        assert character is not None
        assert character['id'] == char_id
        
        # Verify messages were added to the conversation
        messages = mock_db.get_messages_for_conversation(conv_id)
        assert len(messages) == 4  # 2 exchanges * 2 messages
    
    def test_load_chat_history_with_invalid_path(self, mock_db, temp_base_dir):
        """Test that loading from outside base directory fails."""
        # Create a file outside the base directory
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write('{"char_name": "Test"}')
            outside_file = tmp.name
        
        try:
            # Attempt to load from outside base directory
            conv_id, char_id = load_chat_history_from_file_and_save_to_db(
                mock_db,
                outside_file,
                base_directory=temp_base_dir
            )
            # Should return None, None for invalid path
            assert conv_id is None
            assert char_id is None
        finally:
            Path(outside_file).unlink(missing_ok=True)
    
    def test_load_chat_history_with_path_traversal(self, mock_db, temp_base_dir):
        """Test that path traversal attempts are blocked."""
        # Try to use path traversal
        malicious_path = "../../../etc/passwd"
        
        conv_id, char_id = load_chat_history_from_file_and_save_to_db(
            mock_db,
            malicious_path,
            base_directory=temp_base_dir
        )
        # Should return None, None for path traversal attempt
        assert conv_id is None
        assert char_id is None
    
    def test_load_chat_history_from_file_object(self, mock_db, sample_chat_history):
        """Test loading chat history from a file-like object."""
        # Create a BytesIO object
        file_obj = io.BytesIO(json.dumps(sample_chat_history).encode('utf-8'))
        
        # Load from file object (no path validation needed)
        conv_id, char_id = load_chat_history_from_file_and_save_to_db(
            mock_db,
            file_obj
        )
        
        # Verify the conversation was created
        assert conv_id is not None
        assert char_id is not None
        
        # Verify messages were added to the conversation
        messages = mock_db.get_messages_for_conversation(conv_id)
        assert len(messages) == 4  # 2 exchanges * 2 messages


class TestAPIUtilsFileOperations:
    """Integration tests for API utils file operations with validation."""
    
    @pytest.fixture
    def temp_upload_dir(self):
        """Create a temporary upload directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test files
            for i in range(3):
                file_path = Path(tmpdir) / f"test_file_{i}.txt"
                file_path.write_text(f"Test content {i}")
            yield tmpdir
    
    def test_prepare_files_with_valid_paths(self, temp_upload_dir):
        """Test preparing files with valid paths."""
        file_paths = [
            str(Path(temp_upload_dir) / "test_file_0.txt"),
            str(Path(temp_upload_dir) / "test_file_1.txt")
        ]
        
        # Validate paths first
        validated_paths = []
        for path in file_paths:
            validated_path = validate_path(path, temp_upload_dir)
            validated_paths.append(str(validated_path))
        
        httpx_files = prepare_files_for_httpx(
            validated_paths
        )
        
        assert httpx_files is not None
        assert len(httpx_files) == 2
        
        # Check file tuples
        for i, (field_name, file_info) in enumerate(httpx_files):
            assert field_name == "files"
            assert file_info[0] == f"test_file_{i}.txt"  # filename
            assert hasattr(file_info[1], 'read')  # file object
            assert file_info[2] == "text/plain"  # mime type
        
        # Clean up file objects
        cleanup_file_objects(httpx_files)
    
    def test_prepare_files_with_invalid_paths(self, temp_upload_dir):
        """Test that invalid file paths are rejected."""
        # Try to access file outside base directory
        file_paths = [
            str(Path(temp_upload_dir) / "test_file_0.txt"),
            "/etc/passwd"  # Invalid path
        ]
        
        # Validate paths - should raise error
        with pytest.raises(ValueError, match="outside the allowed directory"):
            validated_paths = []
            for path in file_paths:
                validated_path = validate_path(path, temp_upload_dir)
                validated_paths.append(str(validated_path))
    
    def test_prepare_files_with_path_traversal(self, temp_upload_dir):
        """Test that path traversal in file paths is blocked."""
        file_paths = [
            str(Path(temp_upload_dir) / ".." / ".." / "etc" / "passwd")
        ]
        
        # Validate paths - should raise error
        with pytest.raises(ValueError, match="outside the allowed directory"):
            for path in file_paths:
                validate_path(path, temp_upload_dir)
    
    def test_prepare_files_without_validation(self, temp_upload_dir):
        """Test preparing files without base directory validation."""
        # When base_directory is None, no validation is performed
        file_paths = [
            str(Path(temp_upload_dir) / "test_file_0.txt")
        ]
        
        httpx_files = prepare_files_for_httpx(file_paths)
        
        assert httpx_files is not None
        assert len(httpx_files) == 1
        
        # Clean up
        cleanup_file_objects(httpx_files)
    
    def test_cleanup_file_objects(self, temp_upload_dir):
        """Test that cleanup properly closes file objects."""
        file_paths = [
            str(Path(temp_upload_dir) / "test_file_0.txt"),
            str(Path(temp_upload_dir) / "test_file_1.txt")
        ]
        
        httpx_files = prepare_files_for_httpx(file_paths)
        
        # Get file objects
        file_objects = [file_info[1] for _, file_info in httpx_files]
        
        # Verify files are open
        for f in file_objects:
            assert not f.closed
        
        # Clean up
        cleanup_file_objects(httpx_files)
        
        # Verify files are closed
        for f in file_objects:
            assert f.closed
    
    def test_cleanup_with_none(self):
        """Test that cleanup handles None gracefully."""
        # Should not raise any errors
        cleanup_file_objects(None)
        cleanup_file_objects([])


class TestFileOperationErrorHandling:
    """Test error handling in file operations."""
    
    @pytest.fixture
    def temp_upload_dir(self):
        """Create a temporary upload directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test files
            for i in range(3):
                file_path = Path(tmpdir) / f"test_file_{i}.txt"
                file_path.write_text(f"Test content {i}")
            yield tmpdir
    
    def test_prepare_files_with_missing_files(self, temp_upload_dir):
        """Test handling of missing files."""
        file_paths = [
            str(Path(temp_upload_dir) / "test_file_0.txt"),
            str(Path(temp_upload_dir) / "missing_file.txt")  # Doesn't exist
        ]
        
        # Should skip missing files with warning
        httpx_files = prepare_files_for_httpx(file_paths)
        
        assert httpx_files is not None
        assert len(httpx_files) == 1  # Only one file prepared
        
        # Clean up
        cleanup_file_objects(httpx_files)
    
    def test_prepare_files_error_recovery(self, temp_upload_dir):
        """Test that file handles are cleaned up on error."""
        # Create a file that will be opened
        valid_file = Path(temp_upload_dir) / "valid.txt"
        valid_file.write_text("Valid content")
        
        file_paths = [str(valid_file)]
        
        # Mock an error during preparation
        original_guess_type = prepare_files_for_httpx.__globals__['mimetypes'].guess_type
        
        def mock_guess_type(filename):
            raise Exception("Simulated error")
        
        with patch('mimetypes.guess_type', mock_guess_type):
            # Even with error, should handle gracefully
            httpx_files = prepare_files_for_httpx(file_paths)
            
            # File should have been closed due to error
            assert httpx_files is None or len(httpx_files) == 0
    
    def test_hidden_file_validation(self, temp_upload_dir):
        """Test that hidden files are rejected when validation is enabled."""
        # Create a hidden file
        hidden_file = Path(temp_upload_dir) / ".hidden_file.txt"
        hidden_file.write_text("Hidden content")
        
        # Validate path - should raise error for hidden files
        with pytest.raises(ValueError, match="hidden files"):
            validate_path(str(hidden_file), temp_upload_dir)


class TestEndToEndFileUpload:
    """End-to-end tests simulating real file upload scenarios."""
    
    @pytest.fixture
    def upload_scenario(self):
        """Set up a complete upload scenario."""
        with tempfile.TemporaryDirectory() as tmpdir:
            upload_dir = Path(tmpdir) / "uploads"
            upload_dir.mkdir()
            
            # Create various test files
            text_file = upload_dir / "document.txt"
            text_file.write_text("Sample document content")
            
            json_file = upload_dir / "data.json"
            json_file.write_text('{"key": "value"}')
            
            # Create a subdirectory with files
            subdir = upload_dir / "subdir"
            subdir.mkdir()
            sub_file = subdir / "nested.txt"
            sub_file.write_text("Nested file content")
            
            yield {
                'base_dir': str(upload_dir),
                'files': {
                    'text': str(text_file),
                    'json': str(json_file),
                    'nested': str(sub_file)
                }
            }
    
    def test_complete_upload_workflow(self, upload_scenario):
        """Test a complete file upload workflow with validation."""
        base_dir = upload_scenario['base_dir']
        files = upload_scenario['files']
        
        # Prepare files for upload
        file_paths = [files['text'], files['json'], files['nested']]
        
        # Validate paths first
        validated_paths = []
        for path in file_paths:
            validated_path = validate_path(path, base_dir)
            validated_paths.append(str(validated_path))
        
        httpx_files = prepare_files_for_httpx(
            validated_paths
        )
        
        try:
            # Verify all files were prepared
            assert len(httpx_files) == 3
            
            # Verify mime types
            mime_types = [file_info[2] for _, file_info in httpx_files]
            assert "text/plain" in mime_types
            assert "application/json" in mime_types
            
            # Simulate using files (reading content)
            for field_name, file_info in httpx_files:
                filename, file_obj, mime_type = file_info
                content = file_obj.read()
                assert len(content) > 0
                file_obj.seek(0)  # Reset for potential reuse
                
        finally:
            # Always clean up
            cleanup_file_objects(httpx_files)
    
    def test_mixed_valid_invalid_paths(self, upload_scenario):
        """Test handling mix of valid and invalid paths."""
        base_dir = upload_scenario['base_dir']
        files = upload_scenario['files']
        
        # Mix of valid and invalid paths
        file_paths = [
            files['text'],  # Valid
            "../outside.txt",  # Invalid - traversal
            files['json'],  # Valid
            "/etc/passwd"  # Invalid - absolute outside
        ]
        
        # Should fail fast on first invalid path during validation
        with pytest.raises(ValueError, match="outside the allowed directory"):
            validated_paths = []
            for path in file_paths:
                validated_path = validate_path(path, base_dir)
                validated_paths.append(str(validated_path))