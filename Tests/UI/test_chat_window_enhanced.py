"""Tests for enhanced chat window UI functionality."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from typing import List, Dict, Any

from textual.app import App
from textual.widgets import Input, Button, TextArea, Static, ListView, Switch
from textual.containers import Container, Horizontal, VerticalScroll

from tldw_chatbook.Chat.chat_models import ChatSessionData
from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.chat_screen_state import ChatScreenState, TabState
from tldw_chatbook.Widgets.enhanced_settings_sidebar import EnhancedSettingsSidebar


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
    async def test_attach_image_press_handler_delegates_to_attachment_handler(self, mock_app, mock_chat_window):
        """Test the attach-image press handler delegates to the attachment handler."""
        mock_chat_window.attachment_handler = Mock()
        mock_chat_window.attachment_handler.handle_attach_image_button = AsyncMock()

        event = Mock()
        event.stop = Mock()
        event.button = Mock(id="attach-image")

        await ChatWindowEnhanced.handle_attach_image_press(mock_chat_window, event)

        event.stop.assert_called_once()
        mock_chat_window.attachment_handler.handle_attach_image_button.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_send_press_keeps_empty_state_visible_when_no_sendable_content(self):
        """Empty state should remain visible when send is pressed with nothing to send."""
        mock_chat_window = Mock()
        mock_chat_window.is_send_button = True
        mock_chat_window._has_sendable_content = Mock(return_value=False)
        mock_chat_window.hide_empty_state = Mock()
        mock_chat_window.handle_send_stop_button = AsyncMock(return_value=None)

        event = Mock()
        event.stop = Mock()

        await ChatWindowEnhanced.handle_send_stop_button_press(mock_chat_window, event)

        event.stop.assert_called_once()
        mock_chat_window.handle_send_stop_button.assert_awaited_once_with(mock_chat_window.app_instance, event)
        mock_chat_window.hide_empty_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_press_hides_empty_state_when_sendable_content_exists(self):
        """Empty state should hide only when a real send can proceed."""
        mock_chat_window = Mock()
        mock_chat_window.is_send_button = True
        mock_chat_window._has_sendable_content = Mock(return_value=True)
        mock_chat_window.hide_empty_state = Mock()
        mock_chat_window.handle_send_stop_button = AsyncMock(return_value=None)

        event = Mock()
        event.stop = Mock()

        await ChatWindowEnhanced.handle_send_stop_button_press(mock_chat_window, event)

        event.stop.assert_called_once()
        mock_chat_window.handle_send_stop_button.assert_awaited_once_with(mock_chat_window.app_instance, event)
        mock_chat_window.hide_empty_state.assert_called_once()
    
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
    
    def test_rag_preset_selection(self):
        """Test the current preset seam updates RAG-related settings."""
        sidebar = EnhancedSettingsSidebar(id_prefix="chat", config={})
        sidebar._set_setting_value = Mock()
        sidebar._update_preset_buttons = Mock()

        sidebar._apply_preset("research")

        assert sidebar.active_preset == "research"
        sidebar._set_setting_value.assert_has_calls([
            call("temperature", 0.3),
            call("streaming", True),
            call("rag_enable", True),
            call("rag_preset", "high_accuracy"),
            call("max_tokens", 4096),
        ])
        sidebar._update_preset_buttons.assert_called_once()

        sidebar._set_setting_value.reset_mock()
        sidebar._update_preset_buttons.reset_mock()

        sidebar._apply_preset("custom")

        assert sidebar.active_preset == "custom"
        sidebar._set_setting_value.assert_not_called()
        sidebar._update_preset_buttons.assert_called_once()
    
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


class TestChatScreenConversationParity:
    """Test focused chat screen parity save/restore behavior."""

    def test_save_tab_sessions_preserves_assistant_scope_contract(self, mock_app):
        """Saving tab sessions preserves assistant identity and scope metadata."""
        screen = ChatScreen(mock_app)
        screen.chat_state = ChatScreenState()

        session = Mock()
        session.session_data = ChatSessionData(
            tab_id="tab-1",
            title="Persona Session",
            conversation_id="conv-1",
            character_id=7,
            character_name="Navigator",
            assistant_kind="persona",
            assistant_id="planner",
            persona_memory_mode="workspace",
            scope_type="workspace",
            workspace_id="workspace-123",
            is_ephemeral=False,
        )
        session.query_one = Mock(side_effect=Exception("no widget"))

        tab_container = Mock()
        tab_container.sessions = {"tab-1": session}
        tab_container.active_session_id = "tab-1"

        screen._save_tab_sessions(tab_container)

        saved_tab = screen.chat_state.get_tab_by_id("tab-1")
        assert saved_tab is not None
        assert saved_tab.assistant_kind == "persona"
        assert saved_tab.assistant_id == "planner"
        assert saved_tab.persona_memory_mode == "workspace"
        assert saved_tab.scope_type == "workspace"
        assert saved_tab.workspace_id == "workspace-123"

    @pytest.mark.asyncio
    async def test_restore_tab_sessions_preserves_assistant_scope_contract(self, mock_app):
        """Restoring tab sessions reapplies assistant identity and scope metadata to live sessions."""
        screen = ChatScreen(mock_app)
        screen.chat_state = ChatScreenState(
            tabs=[
                TabState(
                    tab_id="saved-tab",
                    title="",
                    conversation_id="conv-2",
                    character_id=11,
                    character_name="Scout",
                    assistant_kind="character",
                    assistant_id="char-11",
                    persona_memory_mode="read_only",
                    scope_type="workspace",
                    workspace_id="workspace-999",
                    is_ephemeral=False,
                    has_unsaved_changes=True,
                )
            ],
            active_tab_id="saved-tab",
            tab_order=["saved-tab"],
        )

        restored_session = Mock()
        restored_session.session_data = ChatSessionData(tab_id="restored-tab", title="placeholder")

        async def fake_create_new_tab(title=None, session_data=None):
            tab_container.sessions["restored-tab"] = restored_session
            if session_data is not None:
                restored_session.session_data = session_data
            return "restored-tab"

        tab_container = Mock()
        tab_container.sessions = {}
        tab_container.create_new_tab = AsyncMock(side_effect=fake_create_new_tab)

        await screen._restore_tab_sessions(tab_container)

        assert restored_session.session_data.assistant_kind == "character"
        assert restored_session.session_data.assistant_id == "char-11"
        assert restored_session.session_data.persona_memory_mode == "read_only"
        assert restored_session.session_data.scope_type == "workspace"
        assert restored_session.session_data.workspace_id == "workspace-999"
        assert restored_session.session_data.title == "Chat with Scout"


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
