# Tests/UI/test_plaintext_ingestion.py
"""
Unit and integration tests for plaintext ingestion functionality.
Tests both local plaintext processing and TLDW API plaintext ingestion.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from textual.widgets import Button, Select, Input, Checkbox, TextArea, ListView

from tldw_chatbook.UI.Ingest_Window import IngestWindow
from tldw_chatbook.Widgets.IngestTldwApiPlaintextWindow import IngestTldwApiPlaintextWindow
from tldw_chatbook.tldw_api import ProcessPlaintextRequest
from tldw_chatbook.tldw_api.client import TLDWAPIClient


class TestPlaintextWindow:
    """Test plaintext window UI components."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock app instance."""
        app = Mock()
        app.app_config = {
            "tldw_api": {"base_url": "http://test.api"},
            "api_settings": {"test_provider": {"api_key": "test_key"}},
            "user_data_path": Path("/test/path")
        }
        app.media_db = Mock()
        app.notify = Mock()
        return app
    
    def test_plaintext_window_creation(self, mock_app):
        """Test that plaintext window can be created."""
        window = IngestTldwApiPlaintextWindow(mock_app)
        assert window is not None
        assert window.app_instance == mock_app
        assert window.selected_local_files == []
    
    @pytest.mark.asyncio
    async def test_plaintext_window_compose(self, mock_app):
        """Test plaintext window compose method creates expected widgets."""
        from textual.app import App
        
        # Create a minimal app to provide context
        class TestApp(App):
            def compose(self):
                window = IngestTldwApiPlaintextWindow(mock_app)
                yield from window.compose()
        
        async with TestApp().run_test() as pilot:
            # Check for expected widget types
            assert pilot.app.query_one(Select) is not None  # For encoding and line ending
            assert pilot.app.query_one(Checkbox) is not None  # For remove whitespace, convert paragraphs
            assert pilot.app.query_one(Input) is not None  # For split pattern
            assert pilot.app.query_one(Button) is not None  # For submit button
            assert pilot.app.query_one(TextArea) is not None  # For status area


class TestPlaintextProcessing:
    """Test plaintext processing functionality."""
    
    @pytest.fixture
    def ingest_window(self, mock_app):
        """Create an IngestWindow instance."""
        return IngestWindow(mock_app)
    
    @pytest.fixture
    def sample_text_file(self, tmp_path):
        """Create a sample text file."""
        file_path = tmp_path / "sample.txt"
        file_path.write_text("This is a test.\n\nThis is another paragraph.\n")
        return file_path
    
    @pytest.mark.asyncio
    async def test_read_text_file_utf8(self, ingest_window, sample_text_file):
        """Test reading a UTF-8 text file."""
        content = await ingest_window._read_text_file(sample_text_file, "utf-8")
        assert content == "This is a test.\n\nThis is another paragraph.\n"
    
    @pytest.mark.asyncio
    async def test_read_text_file_auto_detect(self, ingest_window, tmp_path):
        """Test auto-detecting file encoding."""
        # Create a file with Latin-1 encoding
        file_path = tmp_path / "latin1.txt"
        file_path.write_bytes("café".encode('latin-1'))
        
        content = await ingest_window._read_text_file(file_path, "auto")
        assert "caf" in content  # Should contain at least the ASCII part
    
    def test_normalize_line_endings_lf(self, ingest_window):
        """Test normalizing to LF line endings."""
        content = "Line 1\r\nLine 2\rLine 3\n"
        result = ingest_window._normalize_line_endings(content, "lf")
        assert result == "Line 1\nLine 2\nLine 3\n"
    
    def test_normalize_line_endings_crlf(self, ingest_window):
        """Test normalizing to CRLF line endings."""
        content = "Line 1\nLine 2\rLine 3\r\n"
        result = ingest_window._normalize_line_endings(content, "crlf")
        assert result == "Line 1\r\nLine 2\r\nLine 3\r\n"
    
    def test_remove_extra_whitespace(self, ingest_window):
        """Test removing extra whitespace."""
        content = "Too   many   spaces.\n\n\n\nToo many newlines."
        result = ingest_window._remove_extra_whitespace(content)
        assert result == "Too many spaces.\n\nToo many newlines."
    
    def test_convert_to_paragraphs(self, ingest_window):
        """Test converting text to paragraph format."""
        content = "Line 1\nLine 2\n\nParagraph 2\nContinued"
        result = ingest_window._convert_to_paragraphs(content)
        assert result == "Line 1 Line 2\n\nParagraph 2 Continued"


class TestPlaintextIngestionIntegration:
    """Integration tests for plaintext ingestion."""
    
    @pytest.fixture
    def mock_app_with_db(self):
        """Create a mock app with database."""
        app = Mock()
        app.app_config = {
            "user_data_path": Path("/test/path")
        }
        app.media_db = Mock()
        app.media_db.add_media_with_keywords = Mock(return_value=(1, "uuid-123", "Success"))
        app.notify = Mock()
        app.query_one = Mock()
        return app
    
    @pytest.fixture
    def mock_ui_elements(self, mock_app_with_db):
        """Mock UI elements for testing."""
        def query_one_side_effect(selector, widget_type=None):
            mocks = {
                "#ingest-local-plaintext-loading": Mock(display=False),
                "#ingest-local-plaintext-status": Mock(clear=Mock(), load_text=Mock(), display=False),
                "#ingest-local-plaintext-process": Mock(disabled=False),
                "#ingest-local-plaintext-encoding": Mock(value="utf-8"),
                "#ingest-local-plaintext-line-ending": Mock(value="auto"),
                "#ingest-local-plaintext-remove-whitespace": Mock(value=True),
                "#ingest-local-plaintext-paragraphs": Mock(value=False),
                "#ingest-local-plaintext-split-pattern": Mock(value=""),
                "#ingest-local-plaintext-title": Mock(value=""),
                "#ingest-local-plaintext-author": Mock(value="Test Author"),
                "#ingest-local-plaintext-keywords": Mock(text="keyword1, keyword2"),
            }
            return mocks.get(selector, Mock())
        
        mock_app_with_db.query_one = Mock(side_effect=query_one_side_effect)
        return mock_app_with_db
    
    @pytest.mark.asyncio
    async def test_handle_local_plaintext_process(self, mock_ui_elements, tmp_path):
        """Test the full local plaintext processing flow."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        # Create IngestWindow with mocked app
        window = IngestWindow(mock_ui_elements)
        window.selected_local_files = {"local_plaintext": [test_file]}
        
        # Run the process handler
        await window.handle_local_plaintext_process()
        
        # Verify database was called
        mock_ui_elements.media_db.add_media_with_keywords.assert_called_once()
        call_args = mock_ui_elements.media_db.add_media_with_keywords.call_args[1]
        
        assert call_args["media_type"] == "plaintext"
        assert call_args["content"] == "Test content"
        assert call_args["keywords"] == ["keyword1", "keyword2"]
        assert call_args["author"] == "Test Author"
        
        # Verify notification was sent
        mock_ui_elements.notify.assert_called()


class TestTLDWAPIPlaintextIntegration:
    """Test TLDW API plaintext integration."""
    
    @pytest.fixture
    def process_plaintext_request(self):
        """Create a sample ProcessPlaintextRequest."""
        return ProcessPlaintextRequest(
            urls=["http://example.com/test.txt"],
            encoding="utf-8",
            line_ending="auto",
            remove_extra_whitespace=True,
            convert_to_paragraphs=False,
            split_pattern=None,
            keywords=["test", "plaintext"],
            author="Test Author"
        )
    
    @pytest.mark.asyncio
    async def test_tldw_api_process_plaintext(self, process_plaintext_request):
        """Test TLDW API client process_plaintext method."""
        client = TLDWAPIClient("http://test.api", token="test_token")
        
        # Mock the _request method
        mock_response = {
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
            "results": [{
                "status": "Success",
                "input_ref": "test.txt",
                "media_type": "plaintext",
                "content": "Processed content"
            }]
        }
        
        with patch.object(client, '_request', new=AsyncMock(return_value=mock_response)):
            response = await client.process_plaintext(process_plaintext_request, ["test.txt"])
            
            assert response.processed_count == 1
            assert response.errors_count == 0
            assert len(response.results) == 1
            assert response.results[0].media_type == "plaintext"