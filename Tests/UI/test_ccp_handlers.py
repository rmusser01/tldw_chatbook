"""Focused CCP handler tests for dual character/persona management."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from tldw_chatbook.Character_Chat.character_persona_scope_service import CharacterPersonaScopeService
from tldw_chatbook.UI.CCP_Modules import (
    CCPCharacterHandler,
    CCPConversationHandler,
    CCPMessageManager,
    CCPPersonaHandler,
    PersonaMessage,
    ViewChangeMessage,
)
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
    return window


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


class TestCCPCharacterHandler:
    """Character handler coverage for string-friendly selected IDs."""

    @pytest.mark.asyncio
    async def test_load_character_wrapper_accepts_string_identifier(self, mock_window):
        handler = CCPCharacterHandler(mock_window)

        await handler.load_character("char.local.alice")

        mock_window.run_worker.assert_called_once()
        call_args = mock_window.run_worker.call_args
        assert call_args[0][0] == handler._load_character_sync
        assert call_args[0][1] == "char.local.alice"

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
