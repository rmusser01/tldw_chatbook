"""
Integration tests for file extraction workflow.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from textual.app import App
from textual.widgets import Button, DataTable

from tldw_chatbook.Widgets.file_extraction_dialog import FileExtractionDialog
from tldw_chatbook.Utils.file_extraction import FileExtractor, ExtractedFile
from tldw_chatbook.Widgets.chat_message_enhanced import ChatMessageEnhanced


class MockApp(App):
    """Mock app for testing."""
    def notify(self, message, severity="info"):
        """Mock notify method."""
        self.last_notification = (message, severity)


class TestFileExtractionIntegration:
    """Integration tests for the complete file extraction workflow."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock app instance."""
        return MockApp()
    
    @pytest.fixture
    def sample_files(self):
        """Create sample extracted files for testing."""
        return [
            ExtractedFile(
                filename="script.py",
                content="print('Hello, World!')",
                language="python",
                start_pos=0,
                end_pos=100
            ),
            ExtractedFile(
                filename="data.json",
                content='{"name": "test", "value": 42}',
                language="json",
                start_pos=101,
                end_pos=200
            ),
            ExtractedFile(
                filename="config.yaml",
                content="database:\n  host: localhost\n  port: 5432",
                language="yaml",
                start_pos=201,
                end_pos=300
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_file_extraction_dialog_creation(self, mock_app, sample_files):
        """Test creating and mounting the file extraction dialog."""
        dialog = FileExtractionDialog(sample_files)
        dialog.app = mock_app
        
        # Test initial state
        assert len(dialog.extracted_files) == 3
        assert dialog.selected_index is None
        assert len(dialog._selected_files) == 3  # All selected by default
    
    @pytest.mark.asyncio
    async def test_chat_message_file_detection(self):
        """Test that ChatMessageEnhanced detects extractable files."""
        message_content = '''
        Here's a Python script:
        ```python
        def hello():
            print("Hello!")
        ```
        
        And a JSON config:
        ```json
        {
            "debug": true
        }
        ```
        '''
        
        # Create a chat message widget
        message = ChatMessageEnhanced(
            message_id=1,
            author="assistant",
            content=message_content,
            timestamp="2024-01-01 12:00:00"
        )
        
        # The message should detect that files can be extracted
        extractor = FileExtractor()
        files = extractor.extract_files(message_content)
        assert len(files) == 2
        assert files[0].language == "python"
        assert files[1].language == "json"
    
    @pytest.mark.asyncio
    async def test_save_files_workflow(self, mock_app, sample_files):
        """Test the complete file saving workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the home directory
            with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                downloads_path = Path(temp_dir) / "Downloads"
                downloads_path.mkdir()
                
                dialog = FileExtractionDialog(sample_files)
                dialog.app = mock_app
                
                # Save all files
                saved_files = await dialog._save_files(sample_files)
                
                # Verify files were saved
                assert len(saved_files) == 3
                for saved in saved_files:
                    file_path = Path(saved['path'])
                    assert file_path.exists()
                    assert file_path.parent == downloads_path
                
                # Verify notification
                assert mock_app.last_notification[0] == "Saved 3 files to Downloads"
                assert mock_app.last_notification[1] == "success"
    
    @pytest.mark.asyncio
    async def test_file_validation_during_save(self, mock_app):
        """Test that invalid files are caught during save."""
        invalid_files = [
            ExtractedFile(
                filename="bad<>file.txt",  # Invalid filename
                content="content",
                language="text",
                start_pos=0,
                end_pos=10
            )
        ]
        
        dialog = FileExtractionDialog(invalid_files)
        dialog.app = mock_app
        
        # Mock button press event
        save_button = Mock(id="save-all")
        event = Mock(button=save_button)
        
        with patch.object(dialog, 'dismiss') as mock_dismiss:
            await dialog.on_button_pressed(event)
            
            # Should show error notification
            assert "Invalid filename" in mock_app.last_notification[0]
            assert mock_app.last_notification[1] == "error"
            
            # Should not dismiss the dialog
            mock_dismiss.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_file_preview_update(self, mock_app, sample_files):
        """Test file preview updates when selection changes."""
        dialog = FileExtractionDialog(sample_files)
        dialog.app = mock_app
        
        # Simulate mounting
        dialog._preview_content = Mock()
        
        # Update preview for first file
        dialog._update_preview(0)
        
        # The preview should show the Python file content
        assert dialog._preview_content.call_args[0][0] == "```python\nprint('Hello, World!')\n```"
        
        # Update preview for JSON file
        dialog._update_preview(1)
        assert "json" in dialog._preview_content.call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_filename_editing(self, mock_app, sample_files):
        """Test editing filenames in the dialog."""
        dialog = FileExtractionDialog(sample_files)
        dialog.app = mock_app
        dialog.selected_index = 0
        
        # Mock the table update
        with patch.object(dialog, 'query_one') as mock_query:
            mock_table = Mock()
            mock_query.return_value = mock_table
            
            # Simulate filename change
            mock_input = Mock(id="filename-input")
            event = Mock(input=mock_input, value="new_script.py")
            dialog.on_input_changed(event)
            
            # Verify filename was updated
            assert dialog.extracted_files[0].filename == "new_script.py"
            mock_table.update_cell.assert_called()
    
    @pytest.mark.asyncio
    async def test_selected_files_only_save(self, mock_app, sample_files):
        """Test saving only selected files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                downloads_path = Path(temp_dir) / "Downloads"
                downloads_path.mkdir()
                
                dialog = FileExtractionDialog(sample_files)
                dialog.app = mock_app
                
                # Deselect the second file
                dialog._selected_files = {0, 2}  # Only first and third
                
                # Mock button press for save selected
                save_button = Mock(id="save-selected")
                event = Mock(button=save_button)
                
                with patch.object(dialog, 'dismiss'):
                    await dialog.on_button_pressed(event)
                
                # Check that only 2 files were saved
                saved_files = list(downloads_path.glob("*"))
                assert len(saved_files) == 2
                assert any("script.py" in f.name for f in saved_files)
                assert any("config.yaml" in f.name for f in saved_files)
                assert not any("data.json" in f.name for f in saved_files)
    
    @pytest.mark.asyncio
    async def test_duplicate_filename_handling(self, mock_app):
        """Test handling of duplicate filenames."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                downloads_path = Path(temp_dir) / "Downloads"
                downloads_path.mkdir()
                
                # Create existing file
                existing_file = downloads_path / "test.py"
                existing_file.write_text("existing content")
                
                # Try to save file with same name
                files = [
                    ExtractedFile(
                        filename="test.py",
                        content="new content",
                        language="python",
                        start_pos=0,
                        end_pos=10
                    )
                ]
                
                dialog = FileExtractionDialog(files)
                dialog.app = mock_app
                
                saved_files = await dialog._save_files(files)
                
                # Should create test_1.py
                assert len(saved_files) == 1
                assert saved_files[0]['filename'] == "test.py"
                assert "test_1.py" in saved_files[0]['path']
                
                # Verify both files exist
                assert existing_file.exists()
                assert (downloads_path / "test_1.py").exists()
    
    
    @pytest.mark.asyncio
    async def test_large_file_preview_truncation(self, mock_app):
        """Test that large file previews are truncated."""
        large_content = "x" * 10000
        files = [
            ExtractedFile(
                filename="large.txt",
                content=large_content,
                language="text",
                start_pos=0,
                end_pos=10000
            )
        ]
        
        dialog = FileExtractionDialog(files)
        dialog.app = mock_app
        dialog._preview_content = Mock()
        
        dialog._update_preview(0)
        
        # Verify content was truncated
        preview_text = dialog._preview_content.call_args[0][0]
        assert len(preview_text) < 6000  # 5000 + markup
        assert "truncated" in preview_text
    
    @pytest.mark.asyncio
    async def test_callback_execution(self, mock_app, sample_files):
        """Test that callbacks are executed correctly."""
        callback_data = None
        
        def test_callback(data):
            nonlocal callback_data
            callback_data = data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                downloads_path = Path(temp_dir) / "Downloads"
                downloads_path.mkdir()
                
                dialog = FileExtractionDialog(sample_files, callback=test_callback)
                dialog.app = mock_app
                
                # Mock button press
                save_button = Mock(id="save-all")
                event = Mock(button=save_button)
                
                with patch.object(dialog, 'dismiss'):
                    await dialog.on_button_pressed(event)
                
                # Verify callback was called
                assert callback_data is not None
                assert callback_data['action'] == 'save-all'
                assert len(callback_data['files']) == 3
                assert callback_data['selected_indices'] == [0, 1, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])