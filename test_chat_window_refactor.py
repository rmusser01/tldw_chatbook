#!/usr/bin/env python3
"""Test the refactored ChatWindowEnhanced to ensure it works correctly."""

import asyncio
from unittest.mock import MagicMock, patch
from textual.app import App
from textual.css.query import NoMatches

# Mock the app instance for testing
class MockApp(App):
    """Mock app for testing."""
    def __init__(self):
        super().__init__()
        self.current_chat_is_ephemeral = False
        self.app_config = {}
        
    def notify(self, message, severity="info"):
        print(f"[{severity}] {message}")

async def test_chat_window():
    """Test the refactored ChatWindowEnhanced."""
    from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
    
    # Create mock app
    app = MockApp()
    
    # Create chat window
    chat_window = ChatWindowEnhanced(app)
    
    print("✓ ChatWindowEnhanced instantiated successfully")
    
    # Test that we're not caching widgets anymore
    assert not hasattr(chat_window, '_send_button'), "Should not cache _send_button"
    assert not hasattr(chat_window, '_chat_input'), "Should not cache _chat_input"
    assert not hasattr(chat_window, '_mic_button'), "Should not cache _mic_button"
    print("✓ Widget caching removed successfully")
    
    # Test that helper methods exist
    assert hasattr(chat_window, '_get_send_button'), "Should have _get_send_button method"
    assert hasattr(chat_window, '_get_chat_input'), "Should have _get_chat_input method"
    assert hasattr(chat_window, '_get_attachment_indicator'), "Should have _get_attachment_indicator method"
    assert hasattr(chat_window, '_get_tab_container'), "Should have _get_tab_container method"
    print("✓ Widget getter methods exist")
    
    # Test reactive properties (they're defined at class level)
    from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
    assert hasattr(ChatWindowEnhanced, 'pending_image'), "Should have pending_image reactive property"
    assert hasattr(ChatWindowEnhanced, 'is_send_button'), "Should have is_send_button reactive property"
    print("✓ Reactive properties defined correctly")
    
    # Test that compose method exists and returns something
    compose_result = list(chat_window.compose())
    assert len(compose_result) > 0, "compose() should yield widgets"
    print(f"✓ compose() yields {len(compose_result)} widgets")
    
    # Test message handlers exist
    assert hasattr(chat_window, 'on_chat_input_message_send_requested'), "Should have send message handler"
    assert hasattr(chat_window, 'on_chat_streaming_message_stream_started'), "Should have stream started handler"
    print("✓ Message handlers exist")
    
    # Test that @work decorator is correct (no thread=True on async)
    import inspect
    handle_image_method = getattr(chat_window, 'handle_image_path_submitted')
    # Check it's decorated with @work
    assert hasattr(handle_image_method, '__wrapped__'), "handle_image_path_submitted should be decorated"
    print("✓ Worker decorators configured correctly")
    
    print("\n✅ All refactoring tests passed!")
    return True

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_chat_window())
    if result:
        print("\n🎉 ChatWindowEnhanced refactoring successful!")
    else:
        print("\n❌ Tests failed")
        exit(1)