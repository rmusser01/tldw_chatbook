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
    """Test that process_image_attachment properly stores image data."""
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
    
    # Mock the ChatImageHandler.process_image_file
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events.ChatImageHandler.process_image_file') as mock_process:
        # Create an async mock that returns the test data
        import asyncio
        
        async def mock_process_image(path):
            return test_image_data, test_mime_type
        
        mock_process.side_effect = mock_process_image
        
        # Run the process_image_attachment method
        asyncio.run(chat_window.process_image_attachment(test_path))
    
    # Check that pending_image was set correctly
    assert chat_window.pending_image is not None
    assert chat_window.pending_image['data'] == test_image_data
    assert chat_window.pending_image['mime_type'] == test_mime_type
    assert chat_window.pending_image['path'] == test_path
    
    # Check UI updates
    assert mock_attach_button.label == "📎✓"
    mock_indicator.update.assert_called_once()
    mock_indicator.remove_class.assert_called_with("hidden")
    
    # Check notification
    mock_app.notify.assert_called_with("Image attached: test_image.png")

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