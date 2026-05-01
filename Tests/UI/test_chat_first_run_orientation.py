"""First-run Chat orientation tests."""

from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced


def _text(widget: Static) -> str:
    return str(widget.render())


class _ChatFirstRunHarnessApp(App):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self._app_instance = app_instance

    def compose(self) -> ComposeResult:
        yield ChatWindowEnhanced(self._app_instance)


@pytest.fixture
def first_run_settings(monkeypatch):
    def get_setting(section, key, default=None):
        overrides = {
            ("chat_defaults", "enable_tabs"): False,
            ("chat.voice", "show_mic_button"): True,
            ("chat.images", "show_attach_button"): True,
        }
        return overrides.get((section, key), default)

    providers = {"OpenAI": ["gpt-4o"]}

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
def first_run_host():
    host = Mock()
    host.app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4o",
            "temperature": 0.7,
        },
        "api_settings": {
            "openai": {
                "api_key": "",
                "api_key_env_var": "OPENAI_API_KEY",
            }
        },
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


@pytest.mark.asyncio
async def test_chat_first_run_exposes_readiness_and_context_sources(first_run_settings, first_run_host):
    app = _ChatFirstRunHarnessApp(first_run_host)

    async with app.run_test() as pilot:
        empty_state = pilot.app.query_one("#chat-empty-state", Static)
        text = _text(empty_state)

        assert "agentic control surface" in text
        assert "OpenAI is not ready" in text
        assert "OPENAI_API_KEY" in text
        assert "Notes, Media, Search/RAG, Workspaces" in text
        assert "Study flashcards/quizzes" in text
        assert "personas" in text
        assert "Chatbooks" in text
        assert "Ctrl+P" in text
