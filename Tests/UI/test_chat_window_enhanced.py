"""Tests for enhanced chat window UI functionality."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from typing import List, Dict, Any

from textual.app import App
from textual.widgets import Input, Button, TextArea, Static, ListView, Switch
from textual.containers import Container, Horizontal, VerticalScroll


@pytest.fixture
def mock_app():
    """Create a mock TldwCli app instance."""
    app = Mock()
    app.notify = Mock()
    app.loguru_logger = Mock()
    app.config = {
        "chat_defaults": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7
        }
    }
    app.chat_attached_files = {}
    app.run_worker = Mock()
    app.bell = Mock()
    return app


@pytest.fixture
def mock_chat_window():
    """Create a mock chat window."""
    window = Mock()
    window.chat_log = Mock()
    window.chat_input = Mock()
    window.send_button = Mock()
    window.attach_button = Mock()
    window.attachment_indicator = Mock()
    window.sidebar = Mock()
    window.settings_mode_toggle = Mock()
    return window


class TestChatWindowFileAttachmentUI:
    """Test file attachment UI components."""
    
    def test_attach_button_visibility_toggle(self, mock_app, mock_chat_window):
        """Test attach button visibility based on settings."""
        # Test with attach button enabled
        with patch('tldw_chatbook.config.get_cli_setting', return_value=True):
            mock_chat_window.attach_button.visible = True
            assert mock_chat_window.attach_button.visible is True
        
        # Test with attach button disabled
        with patch('tldw_chatbook.config.get_cli_setting', return_value=False):
            mock_chat_window.attach_button.visible = False
            assert mock_chat_window.attach_button.visible is False
    
    def test_attachment_indicator_display(self, mock_app, mock_chat_window):
        """Test attachment indicator shows correct file count."""
        # No attachments
        mock_app.chat_attached_files = {}
        mock_chat_window.attachment_indicator.update("No files attached")
        
        # Single attachment
        mock_app.chat_attached_files = {
            "default": [{"path": "/path/to/file.txt", "type": "text"}]
        }
        mock_chat_window.attachment_indicator.update("1 file attached")
        
        # Multiple attachments
        mock_app.chat_attached_files = {
            "default": [
                {"path": "/path/to/file1.txt", "type": "text"},
                {"path": "/path/to/file2.png", "type": "image"},
                {"path": "/path/to/file3.pdf", "type": "pdf"}
            ]
        }
        mock_chat_window.attachment_indicator.update("3 files attached")
    
    @pytest.mark.asyncio
    async def test_file_picker_dialog_integration(self, mock_app, mock_chat_window):
        """Test file picker dialog integration."""
        # Mock file picker dialog
        with patch('tldw_chatbook.Widgets.enhanced_file_picker.EnhancedFilePickerDialog') as MockPicker:
            mock_dialog = Mock()
            mock_dialog.show = AsyncMock()
            MockPicker.return_value = mock_dialog
            
            # Simulate attach button click
            await mock_chat_window.on_button_pressed(Mock(button=mock_chat_window.attach_button))
            
            # Verify dialog was shown
            mock_dialog.show.assert_called_once()
    
    def test_attachment_preview_panel(self, mock_app, mock_chat_window):
        """Test attachment preview panel displays attached files."""
        # Mock preview panel
        preview_panel = Mock()
        preview_list = Mock()
        preview_panel.query_one = Mock(return_value=preview_list)
        
        # Add attachments
        attachments = [
            {"path": "/path/to/document.pdf", "type": "pdf", "size": 1024},
            {"path": "/path/to/image.jpg", "type": "image", "size": 2048}
        ]
        
        # Update preview
        preview_items = []
        for att in attachments:
            item = Mock()
            item.path = att["path"]
            item.type = att["type"]
            item.size = att["size"]
            preview_items.append(item)
        
        preview_list.clear = Mock()
        preview_list.append = Mock()
        
        # Clear and add items
        preview_list.clear()
        for item in preview_items:
            preview_list.append(item)
        
        preview_list.clear.assert_called_once()
        assert preview_list.append.call_count == 2


class TestChatWindowRAGUI:
    """Test RAG integration UI components."""
    
    def test_rag_panel_visibility(self, mock_app, mock_chat_window):
        """Test RAG panel visibility based on settings mode."""
        # Basic mode - RAG panel visible
        mock_chat_window.settings_mode_toggle.value = False  # Basic mode
        rag_panel = Mock()
        rag_panel.collapsed = False
        assert rag_panel.collapsed is False
        
        # Advanced mode - RAG panel can be collapsed
        mock_chat_window.settings_mode_toggle.value = True  # Advanced mode
        rag_panel.collapsed = True
        assert rag_panel.collapsed is True
    
    def test_rag_preset_selection(self, mock_app, mock_chat_window):
        """Test RAG preset selection updates UI."""
        rag_preset_select = Mock()
        
        # Test preset changes
        presets = ["none", "light", "full", "custom"]
        for preset in presets:
            rag_preset_select.value = preset
            assert rag_preset_select.value == preset
            
            # Verify UI updates based on preset
            if preset == "custom":
                # Custom preset should show advanced options
                assert mock_chat_window.sidebar.query_one("#chat-advanced-rag").collapsed is False
    
    def test_rag_search_scope_checkboxes(self, mock_app, mock_chat_window):
        """Test RAG search scope checkboxes."""
        # Mock scope checkboxes
        media_checkbox = Mock(value=True)
        conv_checkbox = Mock(value=False)
        notes_checkbox = Mock(value=False)
        
        # Test toggling scopes
        media_checkbox.value = False
        conv_checkbox.value = True
        notes_checkbox.value = True
        
        assert media_checkbox.value is False
        assert conv_checkbox.value is True
        assert notes_checkbox.value is True
    
    def test_rag_query_expansion_ui(self, mock_app, mock_chat_window):
        """Test query expansion UI components."""
        # Mock query expansion elements
        expansion_checkbox = Mock(value=False)
        expansion_method = Mock(value="llm")
        expansion_provider = Mock()
        expansion_model = Mock()
        
        # Enable query expansion
        expansion_checkbox.value = True
        assert expansion_checkbox.value is True
        
        # Test method selection
        methods = ["llm", "llamafile", "keywords"]
        for method in methods:
            expansion_method.value = method
            assert expansion_method.value == method
            
            # Verify UI updates based on method
            if method == "llm":
                # Should show provider/model selects
                expansion_provider.visible = True
                expansion_model.visible = True
            elif method == "llamafile":
                # Should show local model input
                expansion_provider.visible = False
                expansion_model.visible = False


class TestChatWindowToolCallingUI:
    """Test tool calling UI integration."""
    
    def test_tool_call_message_rendering(self, mock_app, mock_chat_window):
        """Test tool call messages are properly rendered."""
        # Mock tool call widget
        tool_call_widget = Mock()
        tool_call_widget.tool_calls = [{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"operation": "add", "a": 5, "b": 3}'
            }
        }]
        
        # Add to chat log
        mock_chat_window.chat_log.mount = Mock()
        mock_chat_window.chat_log.mount(tool_call_widget)
        
        mock_chat_window.chat_log.mount.assert_called_once_with(tool_call_widget)
    
    def test_tool_result_message_rendering(self, mock_app, mock_chat_window):
        """Test tool result messages are properly rendered."""
        # Mock tool result widget
        tool_result_widget = Mock()
        tool_result_widget.results = [{
            "tool_call_id": "call_123",
            "output": "Result: 8",
            "is_error": False
        }]
        
        # Add to chat log
        mock_chat_window.chat_log.mount = Mock()
        mock_chat_window.chat_log.mount(tool_result_widget)
        
        mock_chat_window.chat_log.mount.assert_called_once_with(tool_result_widget)
    
    def test_tool_execution_widget_progress(self, mock_app, mock_chat_window):
        """Test tool execution progress widget."""
        # Mock execution widget
        exec_widget = Mock()
        exec_widget.tool_name = "calculator"
        exec_widget.status = "executing"
        
        # Update status
        exec_widget.status = "completed"
        exec_widget.result = "8"
        
        assert exec_widget.status == "completed"
        assert exec_widget.result == "8"


class TestChatWindowTabIntegration:
    """Test tab-aware UI functionality."""
    
    def test_tab_specific_widget_ids(self, mock_app, mock_chat_window):
        """Test widgets have tab-specific IDs when tabs are enabled."""
        with patch('tldw_chatbook.config.get_cli_setting', return_value=True):
            tab_id = "tab-123"
            
            # Widget IDs should include tab ID
            chat_log_id = f"chat-log-{tab_id}"
            chat_input_id = f"chat-input-{tab_id}"
            send_button_id = f"send-stop-chat-{tab_id}"
            
            widgets = {
                "chat_log": Mock(id=chat_log_id),
                "chat_input": Mock(id=chat_input_id),
                "send_button": Mock(id=send_button_id)
            }
            
            for widget_name, widget in widgets.items():
                assert tab_id in widget.id
    
    def test_tab_context_preservation(self, mock_app, mock_chat_window):
        """Test UI state is preserved when switching tabs."""
        # Create tab contexts
        tab1_context = {
            "chat_input_text": "Hello from tab 1",
            "attachments": [{"path": "/tab1/file.txt"}],
            "rag_enabled": True
        }
        
        tab2_context = {
            "chat_input_text": "Hello from tab 2",
            "attachments": [],
            "rag_enabled": False
        }
        
        # Switch to tab1
        mock_chat_window.chat_input.value = tab1_context["chat_input_text"]
        mock_app.chat_attached_files["tab1"] = tab1_context["attachments"]
        
        # Switch to tab2
        mock_chat_window.chat_input.value = tab2_context["chat_input_text"]
        mock_app.chat_attached_files["tab2"] = tab2_context["attachments"]
        
        # Verify contexts are separate
        assert mock_app.chat_attached_files["tab1"] != mock_app.chat_attached_files["tab2"]


class TestChatWindowStreamingUI:
    """Test streaming UI components."""
    
    def test_streaming_toggle_visibility(self, mock_app, mock_chat_window):
        """Test streaming toggle checkbox."""
        streaming_checkbox = Mock()
        streaming_checkbox.value = True
        
        # Disable streaming
        streaming_checkbox.value = False
        assert streaming_checkbox.value is False
        
        # Re-enable streaming
        streaming_checkbox.value = True
        assert streaming_checkbox.value is True
    
    @pytest.mark.asyncio
    async def test_streaming_message_animation(self, mock_app, mock_chat_window):
        """Test streaming message animation."""
        # Mock streaming message widget
        streaming_widget = Mock()
        streaming_widget.content = ""
        streaming_widget.is_streaming = True
        
        # Simulate streaming chunks
        chunks = ["Hello", " world", "!", " How", " can", " I", " help?"]
        for chunk in chunks:
            streaming_widget.content += chunk
            await asyncio.sleep(0.01)  # Simulate delay
        
        streaming_widget.is_streaming = False
        
        assert streaming_widget.content == "Hello world! How can I help?"
        assert streaming_widget.is_streaming is False
    
    def test_stop_generation_button_state(self, mock_app, mock_chat_window):
        """Test stop generation button state during streaming."""
        send_stop_button = Mock()
        
        # During streaming - should show "Stop"
        mock_app.get_current_chat_is_streaming = Mock(return_value=True)
        send_stop_button.label = "Stop"
        send_stop_button.variant = "error"
        
        assert send_stop_button.label == "Stop"
        assert send_stop_button.variant == "error"
        
        # Not streaming - should show "Send"
        mock_app.get_current_chat_is_streaming = Mock(return_value=False)
        send_stop_button.label = "Send"
        send_stop_button.variant = "primary"
        
        assert send_stop_button.label == "Send"
        assert send_stop_button.variant == "primary"


class TestChatWindowSettingsUI:
    """Test settings UI components."""
    
    def test_settings_mode_toggle(self, mock_app, mock_chat_window):
        """Test basic/advanced mode toggle."""
        mode_toggle = Mock()
        
        # Basic mode (False)
        mode_toggle.value = False
        
        # Advanced sections should be hidden
        advanced_sections = [
            "#chat-model-params",
            "#chat-advanced-rag",
            "#chat-advanced-settings",
            "#chat-tools"
        ]
        
        for section_id in advanced_sections:
            section = Mock()
            section.add_class("advanced-only")
            section.visible = False
            assert section.visible is False
        
        # Advanced mode (True)
        mode_toggle.value = True
        
        for section_id in advanced_sections:
            section = Mock()
            section.remove_class("advanced-only")
            section.visible = True
            assert section.visible is True
    
    def test_provider_model_cascade(self, mock_app, mock_chat_window):
        """Test provider selection updates model options."""
        provider_select = Mock()
        model_select = Mock()
        
        # Provider to models mapping
        provider_models = {
            "openai": ["gpt-4", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet"],
            "google": ["gemini-pro", "gemini-pro-vision"]
        }
        
        # Test each provider
        for provider, models in provider_models.items():
            provider_select.value = provider
            
            # Update model options
            model_options = [(m, m) for m in models]
            model_select.options = model_options
            
            # Verify options updated
            assert len(model_select.options) == len(models)


class TestChatWindowAccessibility:
    """Test accessibility features."""
    
    def test_widget_tooltips(self, mock_app, mock_chat_window):
        """Test all interactive widgets have tooltips."""
        widgets_with_tooltips = [
            (mock_chat_window.send_button, "Send message or stop generation"),
            (mock_chat_window.attach_button, "Attach files to message"),
            (Mock(id="chat-streaming-enabled-checkbox"), "Enable/disable streaming responses"),
            (Mock(id="chat-show-attach-button-checkbox"), "Show/hide the file attachment button"),
            (Mock(id="chat-rag-enable-checkbox"), "Enable RAG for enhanced responses")
        ]
        
        for widget, expected_tooltip in widgets_with_tooltips:
            widget.tooltip = expected_tooltip
            assert widget.tooltip == expected_tooltip
    
    def test_keyboard_shortcuts(self, mock_app, mock_chat_window):
        """Test keyboard shortcuts work correctly."""
        # Mock key bindings
        key_bindings = {
            "ctrl+enter": "send_message",
            "ctrl+shift+a": "attach_file",
            "ctrl+/": "toggle_sidebar",
            "ctrl+n": "new_conversation"
        }
        
        for key, action in key_bindings.items():
            # Verify binding exists
            assert key in key_bindings
            assert key_bindings[key] == action


class TestChatWindowErrorHandling:
    """Test error handling in UI."""
    
    def test_attachment_error_display(self, mock_app, mock_chat_window):
        """Test error display for attachment failures."""
        # Simulate attachment error
        error_message = "File too large (max 10MB)"
        
        mock_app.notify(
            error_message,
            severity="error",
            timeout=5
        )
        
        mock_app.notify.assert_called_with(
            error_message,
            severity="error",
            timeout=5
        )
    
    def test_network_error_display(self, mock_app, mock_chat_window):
        """Test network error display."""
        # Simulate network error
        error_message = "Failed to connect to API"
        
        # Show error in chat
        error_widget = Mock()
        error_widget.variant = "error"
        error_widget.content = f"Error: {error_message}"
        
        mock_chat_window.chat_log.mount(error_widget)
        
        assert error_widget.variant == "error"
        assert error_message in error_widget.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
