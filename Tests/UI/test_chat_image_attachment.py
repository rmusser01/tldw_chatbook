# test_chat_image_attachment.py
# Test for chat image attachment functionality
#
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced

def test_image_filters_are_comprehensive():
    """Test that image filters include common formats."""
    from tldw_chatbook.Third_Party.textual_fspicker import Filters
    
    # Create the same filters used in the attachment handler
    image_filters = Filters(
        ("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.webp;*.bmp;*.tiff;*.tif;*.svg"),
        ("PNG Images", "*.png"),
        ("JPEG Images", "*.jpg;*.jpeg"),
        ("GIF Images", "*.gif"),
        ("WebP Images", "*.webp"),
        ("All Files", "*.*")
    )
    
    # Check that filters were created
    assert len(image_filters._filters) > 0
    
    # Check that the first filter includes common image formats
    image_filter = image_filters._filters[0]
    # The filter is a tuple of (name, pattern)
    filter_name, filter_pattern = image_filter
    assert "Image Files" in filter_name
    assert "*.png" in filter_pattern
    assert "*.jpg" in filter_pattern
    assert "*.gif" in filter_pattern

def test_process_image_attachment_stores_data():
    """Test that process_file_attachment properly stores image data."""
    # Create a mock app instance
    mock_app = Mock()
    mock_app.notify = Mock()
    
    # Create the chat window
    chat_window = ChatWindowEnhanced(app_instance=mock_app)
    chat_window.app_instance = mock_app
    
    # Mock the query_one method to return mock widgets
    mock_attach_button = Mock()
    mock_indicator = Mock()
    
    def mock_query_one(selector, widget_type=None):
        if selector == "#attach-image":
            return mock_attach_button
        elif selector == "#image-attachment-indicator":
            return mock_indicator
        raise Exception(f"Widget not found: {selector}")
    
    chat_window.query_one = mock_query_one
    
    # Create test image data
    test_image_data = b"fake_image_data"
    test_mime_type = "image/png"
    test_path = "/tmp/test_image.png"
    
    # Mock the file handler registry
    with patch('tldw_chatbook.Utils.file_handlers.file_handler_registry.process_file') as mock_process:
        # Create a mock processed file result
        mock_result = Mock()
        mock_result.insert_mode = "attachment"
        mock_result.attachment_data = test_image_data
        mock_result.attachment_mime_type = test_mime_type
        mock_result.display_name = "test_image.png"
        mock_result.file_type = "image"
        
        # Create an async mock that returns the mock result
        import asyncio
        
        async def mock_process_file(path):
            return mock_result
        
        mock_process.side_effect = mock_process_file
        
        # Run the process_file_attachment method
        asyncio.run(chat_window.process_file_attachment(test_path))
    
    # Check that pending_attachment was set correctly (new unified system)
    assert chat_window.pending_attachment is not None
    assert chat_window.pending_attachment['data'] == test_image_data
    assert chat_window.pending_attachment['mime_type'] == test_mime_type
    assert chat_window.pending_attachment['path'] == test_path
    assert chat_window.pending_attachment['display_name'] == "test_image.png"
    assert chat_window.pending_attachment['file_type'] == "image"
    assert chat_window.pending_attachment['insert_mode'] == "attachment"
    
    # Check that pending_image was also set for backward compatibility
    assert chat_window.pending_image is not None
    assert chat_window.pending_image['data'] == test_image_data
    assert chat_window.pending_image['mime_type'] == test_mime_type
    assert chat_window.pending_image['path'] == test_path
    
    # Check UI updates
    assert mock_attach_button.label == "📎✓"
    mock_indicator.update.assert_called_once()
    mock_indicator.remove_class.assert_called_with("hidden")
    
    # Check notification
    mock_app.notify.assert_called_with("test_image.png attached")

def test_process_text_file_attachment():
    """Test that process_file_attachment properly handles text files (inline mode)."""
    # Create a mock app instance
    mock_app = Mock()
    mock_app.notify = Mock()
    
    # Create the chat window
    chat_window = ChatWindowEnhanced(app_instance=mock_app)
    chat_window.app_instance = mock_app
    
    # Mock the query_one method to return mock widgets
    mock_chat_input = Mock()
    mock_chat_input.text = "Existing text"
    
    def mock_query_one(selector, widget_type=None):
        if selector == "#chat-input":
            return mock_chat_input
        raise Exception(f"Widget not found: {selector}")
    
    chat_window.query_one = mock_query_one
    
    # Create test text file data
    test_content = "--- Contents of test.txt ---\nThis is test content\n--- End of test.txt ---"
    test_path = "/tmp/test.txt"
    
    # Mock the file handler registry
    with patch('tldw_chatbook.Utils.file_handlers.file_handler_registry.process_file') as mock_process:
        # Create a mock processed file result for text (inline mode)
        mock_result = Mock()
        mock_result.insert_mode = "inline"
        mock_result.content = test_content
        mock_result.display_name = "test.txt"
        mock_result.file_type = "text"
        
        # Create an async mock that returns the mock result
        import asyncio
        
        async def mock_process_file(path):
            return mock_result
        
        mock_process.side_effect = mock_process_file
        
        # Run the process_file_attachment method
        asyncio.run(chat_window.process_file_attachment(test_path))
    
    # Check that text was inserted into chat input
    expected_text = "Existing text\n\n" + test_content
    assert mock_chat_input.text == expected_text
    assert mock_chat_input.cursor_location == len(expected_text)
    
    # Check that pending_attachment and pending_image were NOT set (inline mode)
    assert not hasattr(chat_window, 'pending_attachment') or chat_window.pending_attachment is None
    assert chat_window.pending_image is None
    
    # Check notification
    mock_app.notify.assert_called_with("📄 test.txt content inserted")

def test_clear_image_attachment():
    """Test that clearing image attachment works properly."""
    # Create a mock app instance
    mock_app = Mock()
    mock_app.notify = Mock()
    
    # Create the chat window
    chat_window = ChatWindowEnhanced(app_instance=mock_app)
    chat_window.app_instance = mock_app
    
    # Set a pending image
    chat_window.pending_image = {
        'data': b"test_data",
        'mime_type': "image/png",
        'path': "/tmp/test.png"
    }
    
    # Mock the query_one method
    mock_attach_button = Mock()
    mock_indicator = Mock()
    
    def mock_query_one(selector, widget_type=None):
        if selector == "#attach-image":
            return mock_attach_button
        elif selector == "#image-attachment-indicator":
            return mock_indicator
        raise Exception(f"Widget not found: {selector}")
    
    chat_window.query_one = mock_query_one
    
    # Clear the image
    import asyncio
    asyncio.run(chat_window.handle_clear_image_button(mock_app, None))
    
    # Check that pending_image was cleared
    assert chat_window.pending_image is None
    
    # Check UI updates
    assert mock_attach_button.label == "📎"
    mock_indicator.add_class.assert_called_with("hidden")
    
    # Check notification
    mock_app.notify.assert_called_with("Image attachment cleared")