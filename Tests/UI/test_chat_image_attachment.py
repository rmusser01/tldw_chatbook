# test_chat_image_attachment.py
# Tests for chat file attachment behavior under the current modular contract.

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced


class _MockChatInput:
    """Minimal TextArea-like test double for inline attachment insertion."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.cursor_location = None

    def load_text(self, value: str) -> None:
        self.text = value

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, value: str) -> None:
        self.text = value


def test_image_filters_are_comprehensive():
    """The attachment picker still includes common image formats."""
    from tldw_chatbook.Widgets.enhanced_file_picker import Filters

    image_filters = Filters(
        ("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.webp;*.bmp;*.tiff;*.tif;*.svg"),
        ("PNG Images", "*.png"),
        ("JPEG Images", "*.jpg;*.jpeg"),
        ("GIF Images", "*.gif"),
        ("WebP Images", "*.webp"),
        ("All Files", "*.*"),
    )

    assert len(image_filters._filters) > 0

    filter_name, filter_pattern = image_filters._filters[0]
    assert "Image Files" in filter_name
    assert "*.png" in filter_pattern
    assert "*.jpg" in filter_pattern
    assert "*.gif" in filter_pattern


def test_process_image_attachment_stores_data():
    """Direct processing outside a mounted app stores the attachment contract."""
    mock_app = Mock()
    mock_app.notify = Mock()
    mock_app.chat_attached_files = {}
    mock_app.active_session_id = "default"

    chat_window = ChatWindowEnhanced(app_instance=mock_app)
    chat_window._attachment_indicator = Mock()

    test_path = "/tmp/test_image.png"
    processed_file = SimpleNamespace(
        insert_mode="attachment",
        attachment_data=b"fake_image_data",
        attachment_mime_type="image/png",
        display_name="test_image.png",
        file_type="image",
    )

    chat_window.attachment_handler._load_processed_file = AsyncMock(return_value=processed_file)

    asyncio.run(chat_window.process_file_attachment(test_path))

    assert chat_window.pending_attachment == {
        "data": b"fake_image_data",
        "mime_type": "image/png",
        "path": test_path,
        "display_name": "test_image.png",
        "file_type": "image",
        "insert_mode": "attachment",
    }
    assert chat_window.pending_image == {
        "data": b"fake_image_data",
        "mime_type": "image/png",
        "path": test_path,
    }
    assert mock_app.chat_attached_files["default"] == [
        {
            "path": test_path,
            "type": "image",
            "content": None,
            "mime_type": "image/png",
        }
    ]
    chat_window._attachment_indicator.update.assert_called_with("📎 test_image.png")
    chat_window._attachment_indicator.add_class.assert_called_with("has-attachment")
    mock_app.notify.assert_called_with("test_image.png attached")


def test_process_text_file_attachment():
    """Inline attachments append content to the chat input without pending state."""
    mock_app = Mock()
    mock_app.notify = Mock()
    mock_app.chat_attached_files = {}
    mock_app.active_session_id = "default"

    chat_window = ChatWindowEnhanced(app_instance=mock_app)
    chat_window._chat_input = _MockChatInput("Existing text")

    test_content = "--- Contents of test.txt ---\nThis is test content\n--- End of test.txt ---"
    test_path = "/tmp/test.txt"
    processed_file = SimpleNamespace(
        insert_mode="inline",
        content=test_content,
        display_name="test.txt",
        file_type="text",
    )

    chat_window.attachment_handler._load_processed_file = AsyncMock(return_value=processed_file)

    asyncio.run(chat_window.process_file_attachment(test_path))

    expected_text = "Existing text\n\n" + test_content
    assert chat_window._chat_input.text == expected_text
    assert chat_window._chat_input.cursor_location == (4, len("--- End of test.txt ---"))
    assert chat_window.pending_attachment is None
    assert chat_window.pending_image is None
    mock_app.notify.assert_called_with("📄 test.txt content inserted")


def test_clear_image_attachment():
    """Clearing an attachment resets state, session storage, and the indicator."""
    mock_app = Mock()
    mock_app.notify = Mock()
    mock_app.chat_attached_files = {
        "default": [
            {
                "path": "/tmp/test.png",
                "type": "image",
                "content": None,
                "mime_type": "image/png",
            }
        ]
    }
    mock_app.active_session_id = "default"

    chat_window = ChatWindowEnhanced(app_instance=mock_app)
    chat_window._attachment_indicator = Mock()
    chat_window.pending_attachment = {
        "data": b"test_data",
        "mime_type": "image/png",
        "path": "/tmp/test.png",
        "display_name": "test.png",
        "file_type": "image",
        "insert_mode": "attachment",
    }
    chat_window.pending_image = {
        "data": b"test_data",
        "mime_type": "image/png",
        "path": "/tmp/test.png",
    }

    asyncio.run(chat_window.handle_clear_image_button(mock_app, None))

    assert chat_window.pending_attachment is None
    assert chat_window.pending_image is None
    assert mock_app.chat_attached_files["default"] == []
    chat_window._attachment_indicator.update.assert_called_with("")
    chat_window._attachment_indicator.remove_class.assert_called_with("has-attachment")
    mock_app.notify.assert_called_with("File attachment cleared")
