"""Tests for keyboard shortcuts implementation."""
import pytest
from textual.binding import Binding

from tldw_chatbook.app import TldwCli


class TestKeyboardShortcuts:
    """Test suite for keyboard shortcuts."""
    
    def test_keyboard_bindings_defined(self):
        """Verify keyboard bindings are defined in the app."""
        # Check that bindings are defined
        bindings = TldwCli.BINDINGS
        
        # Expected bindings
        expected_bindings = {
            "ctrl+q": "quit",
            "ctrl+p": "command_palette",
            "ctrl+n": "new_conversation",
            "ctrl+\\[": "toggle_left_sidebar",
            "ctrl+\\]": "toggle_right_sidebar",
            "ctrl+k": "search_conversations",
            "ctrl+?": "show_shortcuts",
            "f1": "show_shortcuts"
        }
        
        # Convert bindings list to dict for easier testing
        binding_dict = {b.key: b.action for b in bindings}
        
        # Verify each expected binding exists
        for key, action in expected_bindings.items():
            assert key in binding_dict, f"Binding for {key} not found"
            assert binding_dict[key] == action, f"Binding for {key} has wrong action: {binding_dict[key]}"
    
    def test_action_methods_exist(self):
        """Verify action methods are implemented."""
        # Check that action methods exist
        action_methods = [
            "action_quit",
            "action_new_conversation",
            "action_toggle_left_sidebar",
            "action_toggle_right_sidebar",
            "action_search_conversations",
            "action_show_shortcuts"
        ]
        
        for method_name in action_methods:
            assert hasattr(TldwCli, method_name), f"Action method {method_name} not found"
            method = getattr(TldwCli, method_name)
            assert callable(method), f"{method_name} is not callable"
    
    def test_shortcuts_modal_css_exists(self):
        """Verify CSS styling for shortcuts modal exists."""
        with open("tldw_chatbook/css/tldw_cli.tcss", "r") as f:
            css_content = f.read()
        
        # Check for shortcuts modal styling
        assert "ShortcutsScreen" in css_content, "ShortcutsScreen CSS not found"
        assert "#shortcuts-container" in css_content, "shortcuts-container CSS not found"
    
    def test_enter_key_handler_in_chat_window(self):
        """Verify Enter key handler exists in ChatWindow."""
        with open("tldw_chatbook/UI/Chat_Window.py", "r") as f:
            source = f.read()
        
        # Check for on_key method
        assert "async def on_key(self, event)" in source, "on_key method not found"
        assert 'event.key == "enter"' in source, "Enter key handling not found"
        assert "handle_chat_send_button_pressed" in source, "Send button handler call not found in Enter key handler"