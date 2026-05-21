"""Focused CCP screen tests for dual character/persona management."""

from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI.CCP_Modules import PersonaMessage
from tldw_chatbook.UI.CCP_Modules.ccp_character_handler import _coerce_local_character_id
from tldw_chatbook.UI.Screens.ccp_screen import CCPScreen, CCPScreenState
from tldw_chatbook.Widgets.CCP_Widgets import (
    CCPPersonaCardWidget,
    CCPPersonaEditorWidget,
    ContinueConversationRequested,
    StartChatRequested,
    StartPersonaChatRequested,
)


async def _wait_for(predicate, pilot, *, timeout: float = 5.0) -> None:
    """Poll mounted Textual state until a UI condition is true."""
    deadline = pilot.app._loop.time() + timeout
    while pilot.app._loop.time() < deadline:
        if predicate():
            return
        await pilot.pause(0.1)
    raise AssertionError("Timed out waiting for CCP UI condition")


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

    def test_character_id_coercion_preserves_non_numeric_destination_ids(self):
        assert _coerce_local_character_id("local:42") == 42
        assert _coerce_local_character_id("persona.local.alice") == "persona.local.alice"


@pytest.mark.asyncio
class TestCCPScreenIntegration:
    """Integration coverage for persona CCP behavior."""

    async def test_ccp_route_uses_destination_native_personas_workbench(self, mock_app_instance):
        app = CCPTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = pilot.app.screen

            assert screen.query_one("#ccp-destination-title", Static).renderable == (
                "Personas | Behavior, characters, prompts, lore | Ready | Local/Server"
            )
            assert isinstance(screen.query_one("#ccp-mode-strip"), object)
            assert isinstance(screen.query_one("#ccp-character-library-pane"), object)
            assert isinstance(screen.query_one("#ccp-behavior-detail-pane"), object)
            assert isinstance(screen.query_one("#ccp-attachment-inspector-pane"), object)
            assert screen.query_one("#ccp-characters-mode-button", Button).label.plain == "Characters"

            with pytest.raises(NoMatches):
                screen.query_one("#ccp-sidebar")

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

    async def test_imported_character_loads_view_and_prefills_editor(
        self,
        mock_app_instance,
        monkeypatch,
    ):
        character = {
            "id": 3,
            "name": "Bean_RPG",
            "description": "A test imported card for role-play validation.",
            "personality": "Curious and helpful.",
            "scenario": "Testing imported character rendering.",
            "first_message": "Hello from Bean.",
            "tags": ["rpg", "fun"],
            "character_version": "1.0",
            "version": 1,
        }
        monkeypatch.setattr(
            "tldw_chatbook.UI.CCP_Modules.ccp_character_handler.fetch_all_characters",
            Mock(return_value=[{"id": "3", "name": "Bean_RPG"}]),
        )
        monkeypatch.setattr(
            "tldw_chatbook.UI.CCP_Modules.ccp_character_handler.fetch_character_by_id",
            Mock(return_value=character),
        )
        app = CCPTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = pilot.app.screen

            await screen.character_handler.load_character("3")
            await _wait_for(lambda: screen.state.selected_character_id == "3", pilot)

            assert screen.state.active_view == "character_card"
            assert screen.query_one("#ccp-card-name-display", Static).renderable == "Bean_RPG"
            assert (
                screen.query_one("#ccp-card-description-display", TextArea).text
                == "A test imported card for role-play validation."
            )
            assert "hidden" not in screen.query_one("#character-details-container").classes

            screen.query_one("#edit-character-btn", Button).press()
            await _wait_for(lambda: screen.state.active_view == "character_editor", pilot)

            assert screen.query_one("#ccp-editor-name", Input).value == "Bean_RPG"
            assert screen.query_one("#ccp-editor-description", TextArea).text == (
                "A test imported card for role-play validation."
            )

    async def test_imported_character_editor_save_updates_loaded_character(
        self,
        mock_app_instance,
        monkeypatch,
    ):
        character = {
            "id": 3,
            "name": "Bean_RPG",
            "description": "Original description.",
            "personality": "Curious and helpful.",
            "scenario": "Testing imported character rendering.",
            "first_message": "Hello from Bean.",
            "tags": ["rpg", "fun"],
            "character_version": "1.0",
            "version": 1,
        }
        saved_payloads = []
        monkeypatch.setattr(
            "tldw_chatbook.UI.CCP_Modules.ccp_character_handler.fetch_all_characters",
            Mock(return_value=[{"id": "3", "name": "Bean_RPG"}]),
        )
        monkeypatch.setattr(
            "tldw_chatbook.UI.CCP_Modules.ccp_character_handler.fetch_character_by_id",
            Mock(return_value=character),
        )

        def capture_update(character_id, data):
            saved_payloads.append((character_id, data))
            return True

        monkeypatch.setattr(
            "tldw_chatbook.UI.CCP_Modules.ccp_character_handler.update_character",
            capture_update,
        )
        app = CCPTestApp(mock_app_instance)

        async with app.run_test() as pilot:
            screen = pilot.app.screen

            await screen.character_handler.load_character("3")
            await _wait_for(lambda: screen.state.selected_character_id == "3", pilot)
            screen.query_one("#edit-character-btn", Button).press()
            await _wait_for(lambda: screen.state.active_view == "character_editor", pilot)

            screen.query_one("#ccp-editor-description", TextArea).text = "Edited imported card."
            screen.query_one("#save-character-btn", Button).press()
            await _wait_for(lambda: bool(saved_payloads), pilot)

            assert saved_payloads[0][0] == "3"
            assert saved_payloads[0][1]["description"] == "Edited imported card."

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
