# Tests/Event_Handlers/test_plaintext_ingest_events.py
"""
Tests for plaintext ingestion event handlers.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from tldw_chatbook.Event_Handlers.ingest_events import (
    _collect_plaintext_specific_data,
    handle_tldw_api_submit_button_pressed
)
from tldw_chatbook.tldw_api import ProcessPlaintextRequest
from textual.widgets import Button, Select, Input, Checkbox


class TestPlaintextEventHandlers:
    """Test plaintext-specific event handlers."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock app instance for testing."""
        app = Mock()
        app.notify = Mock()
        app.media_db = Mock()
        app.app_config = {
            "tldw_api": {"auth_token": "test_token"},
            "api_settings": {"test_provider": {}}
        }
        
        # Mock query_one to return appropriate widgets
        def query_one_side_effect(selector, widget_type=None):
            widget_mocks = {
                "#tldw-api-encoding-plaintext": Mock(spec=Select, value="utf-8"),
                "#tldw-api-line-ending-plaintext": Mock(spec=Select, value="auto"),
                "#tldw-api-remove-whitespace-plaintext": Mock(spec=Checkbox, value=True),
                "#tldw-api-convert-paragraphs-plaintext": Mock(spec=Checkbox, value=False),
                "#tldw-api-split-pattern-plaintext": Mock(spec=Input, value=""),
                "#tldw-api-loading-indicator-plaintext": Mock(display=False),
                "#tldw-api-status-area-plaintext": Mock(clear=Mock(), load_text=Mock()),
                "#tldw-api-endpoint-url-plaintext": Mock(value="http://test.api"),
                "#tldw-api-auth-method-plaintext": Mock(value="config_token"),
            }
            return widget_mocks.get(selector, Mock())
        
        app.query_one = Mock(side_effect=query_one_side_effect)
        return app
    
    def test_collect_plaintext_specific_data(self, mock_app):
        """Test collecting plaintext-specific form data."""
        common_data = {
            "keywords_str": "test, plaintext",
            "title": "Test Title",
            "author": "Test Author",
            "perform_analysis": True
        }
        
        result = _collect_plaintext_specific_data(mock_app, common_data, "plaintext")
        
        assert isinstance(result, ProcessPlaintextRequest)
        assert result.encoding == "utf-8"
        assert result.line_ending == "auto"
        assert result.remove_extra_whitespace is True
        assert result.convert_to_paragraphs is False
        assert result.split_pattern is None
        assert result.keywords == ["test", "plaintext"]
    
    def test_collect_plaintext_with_split_pattern(self, mock_app):
        """Test collecting plaintext data with a split pattern."""
        # Update the split pattern mock
        mock_app.query_one("#tldw-api-split-pattern-plaintext", Input).value = r"\n\n+"
        
        common_data = {
            "keywords_str": "",
            "title": "Test",
            "perform_chunking": True
        }
        
        result = _collect_plaintext_specific_data(mock_app, common_data, "plaintext")
        
        assert result.split_pattern == r"\n\n+"
    
    @pytest.mark.asyncio
    async def test_handle_tldw_api_plaintext_submit(self, mock_app):
        """Test handling TLDW API submit for plaintext."""
        # Create a mock button event
        mock_button = Mock(spec=Button)
        mock_button.id = "tldw-api-submit-plaintext"
        mock_button.disabled = False
        
        mock_event = Mock()
        mock_event.button = mock_button
        
        # Mock the worker
        with patch.object(mock_app, 'run_worker') as mock_run_worker:
            # Mock common form data collection
            with patch('tldw_chatbook.Event_Handlers.ingest_events._collect_common_form_data') as mock_collect_common:
                mock_collect_common.return_value = {
                    "urls": ["http://example.com/test.txt"],
                    "keywords_str": "test",
                    "overwrite_existing_db": False
                }
                
                # Mock plaintext-specific data collection
                with patch('tldw_chatbook.Event_Handlers.ingest_events._collect_plaintext_specific_data') as mock_collect_plaintext:
                    mock_collect_plaintext.return_value = ProcessPlaintextRequest(
                        urls=["http://example.com/test.txt"],
                        encoding="utf-8",
                        keywords=["test"]
                    )
                    
                    await handle_tldw_api_submit_button_pressed(mock_app, mock_event)
                    
                    # Verify form data was collected
                    mock_collect_common.assert_called_once()
                    mock_collect_plaintext.assert_called_once()
                    
                    # Verify worker was started
                    mock_run_worker.assert_called_once()


class TestPlaintextIngestionFlow:
    """Test the complete plaintext ingestion flow."""
    
    @pytest.fixture
    def mock_media_db(self):
        """Mock media database."""
        db = Mock()
        db.add_media_with_keywords = Mock(return_value=(1, "uuid-123", "Success"))
        return db
    
    @pytest.mark.asyncio
    async def test_plaintext_api_to_db_flow(self, mock_app, mock_media_db):
        """Test the flow from API response to database ingestion."""
        mock_app.media_db = mock_media_db
        
        # Simulate API response
        api_response = {
            "processed_count": 1,
            "errors_count": 0,
            "results": [{
                "status": "Success",
                "input_ref": "test.txt",
                "media_type": "plaintext",
                "content": "This is processed plaintext content.",
                "metadata": {
                    "title": "Test Document",
                    "author": "Test Author",
                    "keywords": ["test", "plaintext"]
                }
            }]
        }
        
        # This would normally be handled by on_worker_success
        # We'll test the database call directly
        result = api_response["results"][0]
        
        media_id, uuid, msg = mock_media_db.add_media_with_keywords(
            url=result["input_ref"],
            title=result["metadata"]["title"],
            media_type=result["media_type"],
            content=result["content"],
            keywords=result["metadata"]["keywords"],
            author=result["metadata"]["author"]
        )
        
        assert media_id == 1
        assert uuid == "uuid-123"
        assert msg == "Success"
        
        # Verify the call was made with correct parameters
        mock_media_db.add_media_with_keywords.assert_called_once_with(
            url="test.txt",
            title="Test Document",
            media_type="plaintext",
            content="This is processed plaintext content.",
            keywords=["test", "plaintext"],
            author="Test Author"
        )