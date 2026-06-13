"""Focused CCP handler tests for dual character/persona management."""

from functools import partial
from unittest.mock import AsyncMock, Mock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import ListView, Static

from tldw_chatbook.Character_Chat.Character_Chat_Lib import fetch_all_dictionaries, fetch_character_names
from tldw_chatbook.Character_Chat.character_persona_scope_service import CharacterPersonaScopeService
from tldw_chatbook.UI.CCP_Modules import (
    CCPCharacterHandler,
    CCPConversationHandler,
    CCPMessageManager,
    CCPPersonaHandler,
    CCPPromptHandler,
    PersonaMessage,
    ViewChangeMessage,
)
from tldw_chatbook.UI.CCP_Modules.ccp_character_handler import fetch_all_characters
from tldw_chatbook.tldw_api import PersonaProfileCreate, PersonaProfileUpdate


@pytest.fixture
def mock_window():
    """Create a minimal CCP window double."""
    window = Mock()
    window.app_instance = Mock()
    window.state = Mock(
        runtime_backend="local",
        selected_conversation_id="chat.server.alice",
    )
    backend = Mock()
    backend.list_persona_profiles = AsyncMock(return_value=[])
    backend.get_persona_profile = AsyncMock(
        return_value={
            "id": "persona.local.alice",
            "name": "Alice Persona",
            "mode": "session_scoped",
            "system_prompt": "Be warm and thoughtful.",
            "version": 3,
        }
    )
    backend.create_persona_profile = AsyncMock(
        return_value={
            "id": "persona.local.created",
            "name": "Created Persona",
            "mode": "session_scoped",
            "system_prompt": "Be concise.",
            "version": 1,
        }
    )
    backend.update_persona_profile = AsyncMock(
        return_value={
            "id": "persona.local.alice",
            "name": "Alice Persona Updated",
            "mode": "persistent_scoped",
            "system_prompt": "Be concise and direct.",
            "version": 4,
        }
    )
    backend.list_chat_greetings = AsyncMock(
        return_value={
            "chat_id": "chat.server.alice",
            "greetings": [{"index": 0, "text": "Hello there.", "preview": "Hello there."}],
            "current_selection": 0,
        }
    )
    backend.select_chat_greeting = AsyncMock(
        return_value={
            "chat_id": "chat.server.alice",
            "selected_index": 0,
            "greeting_preview": "Hello there.",
            "checksum_updated": True,
        }
    )
    backend.list_chat_presets = AsyncMock(
        return_value={
            "presets": [{"preset_id": "default", "name": "Default", "builtin": True}],
        }
    )
    backend.create_chat_preset = AsyncMock(return_value={"preset_id": "custom-alpha", "name": "Custom Alpha"})
    backend.update_chat_preset = AsyncMock(return_value={"preset_id": "custom-alpha", "name": "Custom Beta"})
    backend.delete_chat_preset = AsyncMock(return_value={"status": "deleted", "preset_id": "custom-alpha"})
    window.app_instance.character_persona_scope_service = backend
    window.run_worker = Mock()
    window.post_message = Mock()
    window.notify = Mock()
    window.query_one = Mock()
    window.call_from_thread = Mock()
    return window


def test_legacy_ccp_character_list_wrapper_uses_current_db_api():
    db = Mock()
    db.list_character_cards.return_value = [
        {"id": 2, "name": "Beta"},
        {"id": 1, "name": "Alpha"},
    ]

    assert fetch_character_names(db) == [
        {"id": 1, "name": "Alpha"},
        {"id": 2, "name": "Beta"},
    ]


def test_ccp_fetch_all_characters_accepts_current_list_shape(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Character_Chat.Character_Chat_Lib.fetch_character_names",
        Mock(return_value=[{"id": 3, "name": "Gamma"}]),
    )

    assert fetch_all_characters() == [{"id": "3", "name": "Gamma"}]


def test_delete_character_returns_false_when_db_unavailable(monkeypatch):
    """A missing local DB must read as a failed delete, not an AttributeError."""
    from tldw_chatbook.UI.CCP_Modules import ccp_character_handler

    monkeypatch.setattr(ccp_character_handler, "_default_character_db", lambda: None)

    assert ccp_character_handler.delete_character(1, 1) is False


def test_delete_character_returns_false_for_non_numeric_id(monkeypatch):
    """A non-numeric id must read as a failed delete, not a ValueError."""
    from tldw_chatbook.UI.CCP_Modules import ccp_character_handler

    db = Mock()
    monkeypatch.setattr(ccp_character_handler, "_default_character_db", lambda: db)

    assert ccp_character_handler.delete_character("not-a-number", 1) is False
    db.soft_delete_character_card.assert_not_called()


def test_legacy_ccp_dictionary_list_wrapper_uses_current_db_api(monkeypatch):
    db = Mock()
    monkeypatch.setattr(
        "tldw_chatbook.Character_Chat.Chat_Dictionary_Lib.list_chat_dictionaries",
        Mock(return_value=[{"id": 4, "name": "Lore"}]),
    )

    assert fetch_all_dictionaries(db) == [{"id": 4, "name": "Lore"}]


class TestCCPConversationHandler:
    """Conversation handler coverage for string-first IDs."""

    @pytest.mark.asyncio
    async def test_load_conversation_wrapper_accepts_string_identifier(self, mock_window):
        handler = CCPConversationHandler(mock_window)

        await handler.load_conversation("conv-1")

        mock_window.run_worker.assert_called_once()
        call_args = mock_window.run_worker.call_args
        assert call_args[0][0] == handler._load_conversation_sync
        assert call_args[0][1] == "conv-1"

    def test_search_excludes_workspace_scoped_conversations_from_general_results(self, mock_window):
        class FakeConversationDb:
            def search_conversations_by_title(self, title_query, limit=100):
                return [
                    {
                        "id": "conv-global-1",
                        "title": "Alpha",
                        "discovery_owner": "general_chat",
                        "scope_type": "global",
                    },
                    {
                        "id": "conv-ws-1",
                        "title": "Alpha",
                        "discovery_owner": "general_chat",
                        "scope_type": "workspace",
                        "workspace_id": "ws-9",
                    },
                ]

        mock_window.app_instance.chachanotes_db = FakeConversationDb()
        mock_window.state.selected_character_id = None
        mock_window.state.selected_persona_id = None
        handler = CCPConversationHandler(mock_window)

        CCPConversationHandler._search_conversations_sync.__wrapped__(handler, "Alpha", "title")

        assert [row["id"] for row in handler.search_results] == ["conv-global-1"]


class TestCCPCharacterHandler:
    """Character handler coverage for string-friendly selected IDs."""

    @pytest.mark.asyncio
    async def test_load_character_wrapper_accepts_string_identifier(self, mock_window):
        handler = CCPCharacterHandler(mock_window)

        await handler.load_character("char.local.alice")

        mock_window.run_worker.assert_called_once()
        call_args = mock_window.run_worker.call_args
        # The handler dispatches a functools.partial so worker args travel
        # with the callable instead of as positional run_worker arguments.
        worker_callable = call_args[0][0]
        assert isinstance(worker_callable, partial)
        assert worker_callable.func == handler._load_character_sync
        assert worker_callable.args == ("char.local.alice",)

    @pytest.mark.asyncio
    async def test_list_chat_greetings_routes_via_scope_service(self, mock_window):
        handler = CCPCharacterHandler(mock_window)
        mock_window.state.runtime_backend = "server"

        payload = await handler.list_chat_greetings()

        mock_window.app_instance.character_persona_scope_service.list_chat_greetings.assert_awaited_once_with(
            "chat.server.alice",
            mode="server",
        )
        assert payload["current_selection"] == 0


class TestCCPPersonaHandler:
    """Persona handler coverage."""

    @pytest.mark.asyncio
    async def test_refresh_persona_list_routes_through_scope_service(self, mock_window):
        handler = CCPPersonaHandler(mock_window)
        service = mock_window.app_instance.character_persona_scope_service
        service.list_persona_profiles = AsyncMock(
            return_value=[
                {
                    "id": "persona.local.alice",
                    "name": "Alice Persona",
                    "mode": "session_scoped",
                },
                {
                    "id": "persona.local.bob",
                    "name": "Bob Persona",
                    "mode": "persistent_scoped",
                },
            ]
        )

        personas = await handler.refresh_persona_list()

        service.list_persona_profiles.assert_awaited_once()
        assert personas[0]["id"] == "persona.local.alice"
        assert handler.persona_list[1]["name"] == "Bob Persona"

    @pytest.mark.asyncio
    async def test_load_persona_fetches_via_scope_service(self, mock_window):
        handler = CCPPersonaHandler(mock_window)

        await handler.load_persona("persona.local.alice")

        mock_window.app_instance.character_persona_scope_service.get_persona_profile.assert_awaited_once_with(
            "persona.local.alice",
            mode="local",
        )
        assert handler.current_persona_id == "persona.local.alice"
        assert handler.current_persona_data["name"] == "Alice Persona"
        loaded_messages = [
            args[0]
            for args, _ in mock_window.post_message.call_args_list
            if args and isinstance(args[0], PersonaMessage.Loaded)
        ]
        assert loaded_messages

    @pytest.mark.asyncio
    async def test_handle_create_persona_loads_blank_editor_state(self, mock_window):
        editor = Mock()
        mock_window.query_one.return_value = editor
        handler = CCPPersonaHandler(mock_window)

        await handler.handle_create_persona()

        editor.load_persona.assert_called_once()
        payload = editor.load_persona.call_args[0][0]
        assert payload["mode"] == "session_scoped"
        view_messages = [
            args[0]
            for args, _ in mock_window.post_message.call_args_list
            if args and isinstance(args[0], ViewChangeMessage.Requested)
        ]
        assert view_messages[-1].view_name == "persona_editor"

    @pytest.mark.asyncio
    async def test_save_persona_creates_profile_via_scope_service(self, mock_window):
        handler = CCPPersonaHandler(mock_window)

        await handler.save_persona(
            {
                "name": "Created Persona",
                "mode": "session_scoped",
                "system_prompt": "Be concise.",
            }
        )

        create_call = mock_window.app_instance.character_persona_scope_service.create_persona_profile.await_args
        request_data = create_call.args[0]
        assert isinstance(request_data, PersonaProfileCreate)
        assert request_data.name == "Created Persona"
        assert request_data.system_prompt == "Be concise."
        assert create_call.kwargs["mode"] == "local"
        assert handler.current_persona_id == "persona.local.created"

    @pytest.mark.asyncio
    async def test_save_persona_updates_profile_via_scope_service(self, mock_window):
        handler = CCPPersonaHandler(mock_window)
        handler.current_persona_id = "persona.local.alice"

        await handler.save_persona(
            {
                "id": "persona.local.alice",
                "name": "Alice Persona Updated",
                "mode": "persistent_scoped",
                "system_prompt": "Be concise and direct.",
                "version": 3,
            }
        )

        update_call = mock_window.app_instance.character_persona_scope_service.update_persona_profile.await_args
        assert update_call.args[0] == "persona.local.alice"
        request_data = update_call.args[1]
        assert isinstance(request_data, PersonaProfileUpdate)
        assert request_data.mode == "persistent_scoped"
        assert update_call.kwargs["expected_version"] == 3
        assert update_call.kwargs["mode"] == "local"

    @pytest.mark.asyncio
    async def test_list_chat_presets_routes_via_scope_service(self, mock_window):
        handler = CCPPersonaHandler(mock_window)
        mock_window.state.runtime_backend = "server"

        payload = await handler.list_chat_presets()

        mock_window.app_instance.character_persona_scope_service.list_chat_presets.assert_awaited_once_with(
            mode="server",
        )
        assert payload["presets"][0]["preset_id"] == "default"

    @pytest.mark.asyncio
    async def test_select_chat_greeting_notifies_when_current_mode_cannot_use_it(self, mock_window):
        handler = CCPPersonaHandler(mock_window)
        mock_window.state.runtime_backend = "local"
        mock_window.app_instance.character_persona_scope_service.select_chat_greeting = AsyncMock(
            side_effect=ValueError("Local chat greetings are not available yet.")
        )

        payload = await handler.select_chat_greeting(index=1)

        assert payload == {}
        mock_window.notify.assert_called_with(
            "Local chat greetings are not available yet.",
            severity="warning",
        )


class TestCCPPromptHandler:
    """Prompt handler Library empty-state coverage."""

    @pytest.mark.asyncio
    async def test_empty_prompt_search_results_keep_library_guidance(self, mock_window):
        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield ListView(id="ccp-prompts-listview")

        app = TestApp()
        async with app.run_test() as pilot:
            list_view = pilot.app.query_one("#ccp-prompts-listview", ListView)
            mock_window.query_one.return_value = list_view
            handler = CCPPromptHandler(mock_window)
            handler.search_results = []

            await handler._update_search_results_ui()
            await pilot.pause()

            assert len(list_view.children) == 1
            empty_text = str(list_view.children[0].query_one(Static).render())
            assert "No prompts yet." in empty_text
            assert "Create New Prompt" in empty_text
            assert "Chat instructions" in empty_text


class TestCCPMessageManager:
    """Message manager coverage for string session IDs."""

    @pytest.mark.asyncio
    async def test_load_conversation_messages_accepts_string_identifier(self, mock_window):
        manager = CCPMessageManager(mock_window)

        with patch(
            "tldw_chatbook.UI.CCP_Modules.ccp_message_manager.fetch_messages_for_conversation",
            return_value=[{"id": "msg-1", "role": "user", "content": "hello"}],
        ) as mock_fetch:
            await CCPMessageManager.load_conversation_messages.__wrapped__(manager, "conv-1")

        mock_fetch.assert_called_with("conv-1")
        assert manager.current_messages[0]["id"] == "msg-1"
