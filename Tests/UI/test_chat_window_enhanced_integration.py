"""Integration tests for ChatWindowEnhanced using a lightweight widget harness."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards


def _text(widget: Static) -> str:
    return str(widget.render())


def _textarea_text(widget: TextArea) -> str:
    return getattr(widget, "text", getattr(widget, "value", ""))


class _ChatWindowHarnessApp(App):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self._app_instance = app_instance

    def compose(self) -> ComposeResult:
        yield ChatWindowEnhanced(self._app_instance)


@pytest.fixture
def mock_chat_host():
    host = Mock()
    host.app_config = {
        "chat_defaults": {
            "provider": "openai",
            "model": "gpt-4.1",
            "temperature": 0.7,
        }
    }
    host.chat_attached_files = {}
    host.active_session_id = "default"
    host.is_streaming = False
    host.chat_sidebar_collapsed = False
    host.chat_right_sidebar_collapsed = False
    host.notify = Mock()
    host.run_worker = Mock()
    host.bell = Mock()
    host.push_screen = Mock()
    host.call_later = Mock()
    return host


@pytest.fixture
def chat_window_settings(monkeypatch):
    def get_setting(section, key, default=None):
        overrides = {
            ("chat_defaults", "enable_tabs"): False,
            ("chat.voice", "show_mic_button"): True,
            ("chat.images", "show_attach_button"): True,
        }
        return overrides.get((section, key), default)

    providers = {"openai": ["gpt-4.1"]}

    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", get_setting)
    monkeypatch.setattr("tldw_chatbook.UI.Chat_Window_Enhanced.get_cli_setting", get_setting)
    monkeypatch.setattr("tldw_chatbook.Widgets.compact_model_bar.get_cli_setting", get_setting)
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.compact_model_bar.get_cli_providers_and_models",
        lambda: providers,
    )
    monkeypatch.setattr("tldw_chatbook.Widgets.enhanced_settings_sidebar.get_cli_setting", get_setting)
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.enhanced_settings_sidebar.get_cli_providers_and_models",
        lambda: providers,
    )


@pytest.fixture
def chat_app(chat_window_settings, mock_chat_host):
    return _ChatWindowHarnessApp(mock_chat_host)


class TestChatWindowEnhancedIntegration:
    @pytest.mark.asyncio
    async def test_widget_initialization(self, chat_app):
        """Core chat widgets mount under the current shell contract."""
        async with chat_app.run_test() as pilot:
            chat_window = pilot.app.query_one(ChatWindowEnhanced)

            assert pilot.app.query_one("#send-stop-chat", Button) is not None
            assert pilot.app.query_one("#chat-input", TextArea) is not None
            assert pilot.app.query_one("#chat-task-surface", ChatTaskCards) is not None
            assert chat_window._send_button is not None
            assert chat_window._chat_input is not None
            assert chat_window._attachment_indicator is not None

    @pytest.mark.asyncio
    async def test_send_button_state_changes(self, chat_app):
        """The send button reflects the current streaming state."""
        async with chat_app.run_test() as pilot:
            chat_window = pilot.app.query_one(ChatWindowEnhanced)
            send_button = pilot.app.query_one("#send-stop-chat", Button)

            assert send_button.tooltip == "Send message"

            chat_window.is_send_button = False
            await pilot.pause(0.05)

            assert send_button.tooltip == "Stop generation"

            chat_window.is_send_button = True
            await pilot.pause(0.05)

            assert send_button.tooltip == "Send message"

    @pytest.mark.asyncio
    async def test_attachment_indicator_updates(self, chat_app):
        """Setting a pending image updates the inline attachment affordance."""
        async with chat_app.run_test() as pilot:
            chat_window = pilot.app.query_one(ChatWindowEnhanced)
            indicator = pilot.app.query_one("#image-attachment-indicator", Static)

            assert _text(indicator) == ""

            chat_window.pending_image = {
                "path": "/tmp/reference-notes.txt",
                "data": b"",
                "mime_type": "text/plain",
            }
            await pilot.pause(0.05)

            assert "reference-notes.txt" in _text(indicator)

            chat_window._clear_attachment_state()
            await pilot.pause(0.05)

            assert _text(indicator) == ""

    @pytest.mark.asyncio
    async def test_sidebar_toggle_functionality(self, chat_app):
        """The left sidebar toggle still controls the settings rail."""
        async with chat_app.run_test() as pilot:
            chat_window = pilot.app.query_one(ChatWindowEnhanced)
            sidebar = pilot.app.query_one("#chat-left-sidebar")
            initial_display = sidebar.display

            await pilot.click("#toggle-chat-left-sidebar")
            await pilot.pause(0.05)
            assert sidebar.display != initial_display
            assert chat_window.app_instance.chat_sidebar_collapsed is True

    @pytest.mark.asyncio
    async def test_keyboard_shortcut_toggles_voice_input(self, chat_app):
        """Ctrl+M routes through the chat window voice toggle action."""
        async with chat_app.run_test() as pilot:
            chat_window = pilot.app.query_one(ChatWindowEnhanced)

            def fake_toggle() -> None:
                chat_window.voice_handler.is_voice_recording = True

            chat_window.voice_handler.toggle_voice_input = fake_toggle

            await pilot.press("ctrl+m")
            await pilot.pause(0.05)

            assert chat_window.is_voice_recording is True

    @pytest.mark.asyncio
    async def test_chat_input_focus_and_editing(self, chat_app):
        """The chat input can take focus and accept text updates."""
        async with chat_app.run_test() as pilot:
            chat_input = pilot.app.query_one("#chat-input", TextArea)

            await pilot.click("#chat-input")
            await pilot.pause(0.05)
            assert chat_input.has_focus

            chat_input.load_text("Hello")
            assert _textarea_text(chat_input) == "Hello"

            chat_input.clear()
            assert _textarea_text(chat_input) == ""

    @pytest.mark.asyncio
    async def test_file_attachment_workflow_updates_session_state(self, chat_app, mock_chat_host):
        """Processed attachments update both the indicator and session attachment list."""
        async with chat_app.run_test() as pilot:
            chat_window = pilot.app.query_one(ChatWindowEnhanced)
            indicator = pilot.app.query_one("#image-attachment-indicator", Static)

            processed_file = SimpleNamespace(
                insert_mode="attachment",
                file_type="text",
                path="/tmp/test-brief.txt",
                content="Test file content",
                mime_type="text/plain",
            )

            chat_window.attachment_handler._handle_processed_file(processed_file)
            await pilot.pause(0.05)

            assert mock_chat_host.chat_attached_files["default"][0]["path"] == "/tmp/test-brief.txt"
            assert "test-brief.txt" in _text(indicator)

    @pytest.mark.asyncio
    async def test_attachment_errors_notify_the_user(self, chat_app, mock_chat_host):
        """Malformed processed file data fails closed with a user-visible notification."""
        async with chat_app.run_test() as pilot:
            chat_window = pilot.app.query_one(ChatWindowEnhanced)
            mock_chat_host.notify.reset_mock()

            chat_window.attachment_handler._handle_processed_file(object())

            mock_chat_host.notify.assert_called_with("Invalid file data", severity="error")

    @pytest.mark.asyncio
    async def test_cached_widget_references_are_reused(self, chat_app):
        """Mount-time widget caches point at the live chat widgets."""
        async with chat_app.run_test() as pilot:
            chat_window = pilot.app.query_one(ChatWindowEnhanced)

            assert chat_window._get_send_button() is pilot.app.query_one("#send-stop-chat", Button)
            assert chat_window._get_chat_input() is pilot.app.query_one("#chat-input", TextArea)
            assert chat_window._get_task_cards() is pilot.app.query_one("#chat-task-surface", ChatTaskCards)


class TestChatWindowEnhancedAccessibility:
    @pytest.mark.asyncio
    async def test_tooltips_present(self, chat_app):
        """Primary controls expose non-empty tooltips."""
        async with chat_app.run_test() as pilot:
            for button_id in (
                "#send-stop-chat",
                "#toggle-chat-left-sidebar",
                "#attach-image",
                "#mic-button",
            ):
                button = pilot.app.query_one(button_id, Button)
                assert button.tooltip is not None
                assert len(str(button.tooltip)) > 0

    @pytest.mark.asyncio
    async def test_keyboard_navigation(self, chat_app):
        """Tab navigation should move focus to a focusable control."""
        async with chat_app.run_test() as pilot:
            await pilot.press("tab")
            await pilot.pause(0.05)
            assert pilot.app.focused is not None
