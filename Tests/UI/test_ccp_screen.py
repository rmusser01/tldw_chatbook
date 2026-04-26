"""Focused CCP screen tests for dual character/persona management."""

from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI.CCP_Modules import PersonaMessage
from tldw_chatbook.UI.Screens.ccp_screen import CCPScreen, CCPScreenState
from tldw_chatbook.Widgets.CCP_Widgets import (
    CCPPersonaCardWidget,
    CCPPersonaEditorWidget,
    ContinueConversationRequested,
    StartChatRequested,
    StartPersonaChatRequested,
)


@pytest.fixture
def mock_app_instance():
    """Create a minimal app instance for CCP screen tests."""
    app = Mock()
    app.app_config = {"api_endpoints": {}, "chat_defaults": {}, "ui_settings": {}}
    app.character_service = Mock()
    app.character_persona_scope_service = Mock()
    app.character_persona_scope_service.list_persona_profiles = AsyncMock(return_value=[])
    app.prompt_service = Mock()
    app.dictionary_service = Mock()
    return app


class CCPTestApp(App):
    """Minimal Textual app that mounts the CCP screen."""

    def __init__(self, mock_app_instance):
        super().__init__()
        self.character_service = mock_app_instance.character_service
        self.character_persona_scope_service = mock_app_instance.character_persona_scope_service
        self.prompt_service = mock_app_instance.prompt_service
        self.dictionary_service = mock_app_instance.dictionary_service

    def on_mount(self) -> None:
        self.push_screen(CCPScreen(self))


class TestCCPScreenState:
    """State coverage for the persona-aware CCP screen."""

    def test_selected_persona_id_is_first_class_state(self):
        state = CCPScreenState()

        assert state.selected_character_id is None
        assert state.selected_persona_id is None
        assert state.selected_conversation_id is None

    def test_current_runtime_backend_prefers_authoritative_runtime_policy(self, mock_app_instance):
        mock_app_instance.runtime_policy = Mock(state=RuntimeSourceState(active_source="server"))
        mock_app_instance.current_runtime_backend = "local"
        mock_app_instance.runtime_backend = "local"
        mock_app_instance.get_authoritative_runtime_source = Mock(return_value="server")

        screen = CCPScreen(mock_app_instance)

        assert screen._current_runtime_backend() == "server"

    def test_mode_switch_clears_selected_entity_and_session(self):
        state = CCPScreenState(
            selected_character_id="char.local.alice",
            selected_persona_id="persona.local.alice",
            selected_conversation_id="conv-1",
            conversation_search_results=[{"id": "conv-1"}],
        )

        state.reset_for_backend_change()

        assert state.selected_character_id is None
        assert state.selected_persona_id is None
        assert state.selected_conversation_id is None
        assert state.conversation_search_results == []


@pytest.mark.asyncio
class TestCCPScreenIntegration:
    """Integration coverage for persona CCP behavior."""

    async def test_persona_selection_is_first_class_in_ccp_screen(self, mock_app_instance):
        app = CCPTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = pilot.app.screen

            screen.state.active_view = "persona_profiles"
            screen.state.selected_persona_id = "persona.local.alice"

            assert screen.state.active_view == "persona_profiles"
            assert screen.state.selected_persona_id == "persona.local.alice"

    async def test_validate_state_accepts_persona_profiles_view(self, mock_app_instance):
        screen = CCPScreen(mock_app_instance)

        validated = screen.validate_state(CCPScreenState(active_view="persona_profiles"))

        assert validated.active_view == "persona_profiles"

    async def test_prompt_editor_mounts_usage_and_version_controls(self, mock_app_instance):
        app = CCPTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = pilot.app.screen

            assert isinstance(screen.query_one("#ccp-editor-prompt-usage-display"), Static)
            assert isinstance(screen.query_one("#ccp-editor-prompt-version-input"), Input)
            assert isinstance(screen.query_one("#ccp-editor-prompt-record-usage-button"), Button)
            assert isinstance(screen.query_one("#ccp-editor-prompt-list-versions-button"), Button)
            assert isinstance(screen.query_one("#ccp-editor-prompt-restore-version-button"), Button)
            assert isinstance(screen.query_one("#ccp-editor-prompt-version-status"), Static)

    async def test_persona_loaded_message_updates_state_and_widgets(self, mock_app_instance):
        app = CCPTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = pilot.app.screen

            persona_msg = PersonaMessage.Loaded(
                persona_id="persona.local.alice",
                persona_data={
                    "id": "persona.local.alice",
                    "name": "Alice Persona",
                    "mode": "session_scoped",
                    "system_prompt": "Be friendly and direct.",
                },
            )

            await screen.on_persona_message_loaded(persona_msg)

            assert screen.state.selected_persona_id == "persona.local.alice"
            assert screen.state.selected_character_id is None

            persona_card = screen.query_one(CCPPersonaCardWidget)
            persona_editor = screen.query_one(CCPPersonaEditorWidget)
            assert persona_card.persona_data["name"] == "Alice Persona"
            assert persona_editor.persona_data["id"] == "persona.local.alice"

    async def test_save_restore_persists_persona_selection(self, mock_app_instance):
        screen = CCPScreen(mock_app_instance)
        screen.state.selected_persona_id = "persona.local.alice"
        screen.state.active_view = "persona_profiles"

        saved = screen.save_state()

        restored = CCPScreen(mock_app_instance)
        restored.restore_state(saved)

        assert restored.state.selected_persona_id == "persona.local.alice"
        assert restored.state.active_view == "persona_profiles"

    async def test_continue_conversation_focuses_existing_chat_tab(self, mock_app_instance, monkeypatch):
        app = CCPTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = pilot.app.screen
            screen.state.selected_conversation_id = "conv-1"
            screen.conversation_handler.current_conversation_data = {
                "id": "conv-1",
                "title": "Scoped Conversation",
                "runtime_backend": "server",
                "discovery_owner": "ccp_persona",
                "discovery_entity_id": "persona.local.alice",
                "assistant_kind": "persona",
                "assistant_id": "persona.local.alice",
            }

            existing_session = Mock()
            existing_session.session_data = Mock(conversation_id="conv-1")
            tab_container = Mock()
            tab_container.sessions = {"tab-1": existing_session}
            tab_container.create_new_tab = AsyncMock(return_value="tab-1")
            tab_container.switch_to_tab_async = AsyncMock()

            chat_window = Mock()
            chat_window._get_tab_container = Mock(return_value=tab_container)
            pilot.app.query_one = Mock(return_value=chat_window)

            display_mock = AsyncMock()
            monkeypatch.setattr(
                "tldw_chatbook.Event_Handlers.Chat_Events.chat_events_tabs.display_conversation_in_chat_tab_ui_with_tabs",
                display_mock,
            )

            await screen.on_continue_conversation_requested(ContinueConversationRequested())

            tab_container.create_new_tab.assert_awaited_once()
            tab_container.switch_to_tab_async.assert_awaited_once_with("tab-1")
            display_mock.assert_not_called()

    async def test_start_character_chat_launches_blank_main_chat_session(self, mock_app_instance, monkeypatch):
        app = CCPTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = pilot.app.screen
            screen.state.selected_character_id = "7"
            screen.state.selected_character_name = "Alice Character"
            screen.state.selected_character_data = {"id": 7, "name": "Alice Character"}

            tab_container = Mock()
            tab_container.sessions = {}
            tab_container.create_new_tab = AsyncMock(return_value="tab-1")
            tab_container.switch_to_tab_async = AsyncMock()

            chat_window = Mock()
            chat_window._get_tab_container = Mock(return_value=tab_container)
            pilot.app.query_one = Mock(return_value=chat_window)

            display_mock = AsyncMock()
            monkeypatch.setattr(
                "tldw_chatbook.Event_Handlers.Chat_Events.chat_events_tabs.display_conversation_in_chat_tab_ui_with_tabs",
                display_mock,
            )

            await screen.on_start_chat_requested(StartChatRequested(7))

            session_data = tab_container.create_new_tab.await_args.kwargs["session_data"]
            assert session_data.conversation_id is None
            assert session_data.is_ephemeral is True
            assert session_data.character_id == 7
            assert session_data.character_name == "Alice Character"
            assert session_data.assistant_kind == "character"
            assert session_data.assistant_id == "7"
            assert session_data.discovery_owner == "ccp_character"
            assert session_data.discovery_entity_id == "7"
            tab_container.switch_to_tab_async.assert_awaited_once_with("tab-1")
            display_mock.assert_not_called()

    async def test_start_persona_chat_launches_blank_main_chat_session(self, mock_app_instance, monkeypatch):
        app = CCPTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = pilot.app.screen
            screen.state.selected_persona_id = "persona.local.alice"
            screen.state.selected_character_name = "Alice Persona"
            screen.persona_handler.current_persona_data = {
                "id": "persona.local.alice",
                "name": "Alice Persona",
                "mode": "session_scoped",
            }

            tab_container = Mock()
            tab_container.sessions = {}
            tab_container.create_new_tab = AsyncMock(return_value="tab-1")
            tab_container.switch_to_tab_async = AsyncMock()

            chat_window = Mock()
            chat_window._get_tab_container = Mock(return_value=tab_container)
            pilot.app.query_one = Mock(return_value=chat_window)

            display_mock = AsyncMock()
            monkeypatch.setattr(
                "tldw_chatbook.Event_Handlers.Chat_Events.chat_events_tabs.display_conversation_in_chat_tab_ui_with_tabs",
                display_mock,
            )

            await screen.on_start_persona_chat_requested(StartPersonaChatRequested("persona.local.alice"))

            session_data = tab_container.create_new_tab.await_args.kwargs["session_data"]
            assert session_data.conversation_id is None
            assert session_data.is_ephemeral is True
            assert session_data.character_id is None
            assert session_data.character_name == "Alice Persona"
            assert session_data.assistant_kind == "persona"
            assert session_data.assistant_id == "persona.local.alice"
            assert session_data.discovery_owner == "ccp_persona"
            assert session_data.discovery_entity_id == "persona.local.alice"
            tab_container.switch_to_tab_async.assert_awaited_once_with("tab-1")
            display_mock.assert_not_called()
