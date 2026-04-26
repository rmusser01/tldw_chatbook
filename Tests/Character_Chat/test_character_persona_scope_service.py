from unittest.mock import Mock

import pytest

from tldw_chatbook.Character_Chat.character_persona_scope_service import (
    CharacterPersonaScopeService,
)
from tldw_chatbook.Character_Chat.server_character_persona_service import (
    ServerCharacterPersonaService,
)
from tldw_chatbook.runtime_policy import PolicyDecision, PolicyDeniedError


class FakeCharacterPersonaClient:
    def __init__(self):
        self.list_characters_calls = []
        self.search_characters_calls = []
        self.get_character_calls = []
        self.create_character_calls = []
        self.update_character_calls = []
        self.delete_character_calls = []
        self.restore_character_calls = []
        self.list_persona_profiles_calls = []
        self.get_persona_profile_calls = []
        self.create_persona_profile_calls = []
        self.update_persona_profile_calls = []
        self.delete_persona_profile_calls = []
        self.restore_persona_profile_calls = []
        self.list_persona_exemplars_calls = []
        self.get_persona_exemplar_calls = []
        self.create_persona_exemplar_calls = []
        self.import_persona_exemplars_calls = []
        self.update_persona_exemplar_calls = []
        self.review_persona_exemplar_calls = []
        self.delete_persona_exemplar_calls = []
        self.get_character_exemplar_calls = []
        self.create_character_exemplar_calls = []
        self.update_character_exemplar_calls = []
        self.delete_character_exemplar_calls = []
        self.search_character_exemplars_calls = []
        self.select_character_exemplars_debug_calls = []
        self.list_greetings_calls = []
        self.select_greeting_calls = []
        self.list_presets_calls = 0
        self.create_preset_calls = []
        self.update_preset_calls = []
        self.delete_preset_calls = []
        self.create_character_chat_session_calls = []
        self.list_character_chat_sessions_calls = []
        self.get_character_chat_session_calls = []
        self.update_character_chat_session_calls = []
        self.delete_character_chat_session_calls = []
        self.restore_character_chat_session_calls = []
        self.list_character_messages_calls = []
        self.get_character_message_calls = []
        self.create_character_message_calls = []
        self.update_character_message_calls = []
        self.delete_character_message_calls = []
        self.search_character_messages_calls = []
        self.get_chat_settings_calls = []
        self.update_chat_settings_calls = []
        self.export_chat_history_calls = []
        self.get_author_note_info_calls = []
        self.export_lorebook_diagnostics_calls = []

    async def list_characters(self, limit=100, offset=0):
        self.list_characters_calls.append({"limit": limit, "offset": offset})
        return [{"id": 1, "name": "Ada"}]

    async def search_characters(self, query, limit=10):
        self.search_characters_calls.append({"query": query, "limit": limit})
        return [{"id": 1, "name": "Ada", "description": query}]

    async def get_character(self, character_id):
        self.get_character_calls.append(character_id)
        return {"id": character_id, "name": "Ada", "version": 3}

    async def create_character(self, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.create_character_calls.append(payload)
        return {"id": 12, "version": 1, **payload}

    async def update_character(self, character_id, request_data, expected_version):
        payload = request_data.model_dump(exclude_unset=True, exclude_none=True, mode="json")
        self.update_character_calls.append(
            {"character_id": character_id, "payload": payload, "expected_version": expected_version}
        )
        return {"id": character_id, "version": expected_version + 1, **payload}

    async def delete_character(self, character_id, expected_version):
        self.delete_character_calls.append({"character_id": character_id, "expected_version": expected_version})
        return {"status": "deleted", "character_id": character_id}

    async def restore_character(self, character_id, expected_version):
        self.restore_character_calls.append({"character_id": character_id, "expected_version": expected_version})
        return {"id": character_id, "name": "Ada", "version": expected_version + 1, "deleted": False}

    async def list_persona_profiles(self, active_only=False, include_deleted=False, limit=100, offset=0):
        self.list_persona_profiles_calls.append(
            {
                "active_only": active_only,
                "include_deleted": include_deleted,
                "limit": limit,
                "offset": offset,
            }
        )
        return [{"id": "persona-1", "name": "Guide"}]

    async def get_persona_profile(self, persona_id):
        self.get_persona_profile_calls.append(persona_id)
        return {
            "id": persona_id,
            "name": "Guide",
            "mode": "session_scoped",
            "system_prompt": "Be useful.",
        }

    async def create_persona_profile(self, request_data):
        payload = request_data.model_dump(mode="json")
        self.create_persona_profile_calls.append(payload)
        return {
            "id": payload.get("id") or "persona-created",
            "name": payload["name"],
            "mode": payload.get("mode", "session_scoped"),
            "system_prompt": payload.get("system_prompt"),
            "version": 1,
        }

    async def update_persona_profile(self, persona_id, request_data, expected_version=None):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.update_persona_profile_calls.append(
            {
                "persona_id": persona_id,
                "payload": payload,
                "expected_version": expected_version,
            }
        )
        return {
            "id": persona_id,
            "name": payload.get("name", "Guide"),
            "mode": payload.get("mode", "session_scoped"),
            "system_prompt": payload.get("system_prompt"),
            "version": expected_version or 1,
        }

    async def delete_persona_profile(self, persona_id, expected_version=None):
        self.delete_persona_profile_calls.append(
            {"persona_id": persona_id, "expected_version": expected_version}
        )
        return {"status": "deleted", "persona_id": persona_id}

    async def restore_persona_profile(self, persona_id, expected_version):
        self.restore_persona_profile_calls.append(
            {"persona_id": persona_id, "expected_version": expected_version}
        )
        return {"id": persona_id, "name": "Guide", "version": expected_version + 1}

    async def list_persona_exemplars(
        self,
        persona_id,
        include_disabled=False,
        include_deleted=False,
        include_deleted_personas=False,
        limit=100,
        offset=0,
    ):
        self.list_persona_exemplars_calls.append(
            {
                "persona_id": persona_id,
                "include_disabled": include_disabled,
                "include_deleted": include_deleted,
                "include_deleted_personas": include_deleted_personas,
                "limit": limit,
                "offset": offset,
            }
        )
        return [{"id": "ex-1", "persona_id": persona_id, "content": "Use concise answers."}]

    async def get_persona_exemplar(self, persona_id, exemplar_id):
        self.get_persona_exemplar_calls.append({"persona_id": persona_id, "exemplar_id": exemplar_id})
        return {"id": exemplar_id, "persona_id": persona_id, "content": "Use concise answers."}

    async def create_persona_exemplar(self, persona_id, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.create_persona_exemplar_calls.append({"persona_id": persona_id, "payload": payload})
        return {"id": "ex-new", "persona_id": persona_id, **payload}

    async def import_persona_exemplars(self, persona_id, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.import_persona_exemplars_calls.append({"persona_id": persona_id, "payload": payload})
        return {"persona_id": persona_id, "created": 2}

    async def update_persona_exemplar(self, persona_id, exemplar_id, request_data):
        payload = request_data.model_dump(exclude_unset=True, exclude_none=True, mode="json")
        self.update_persona_exemplar_calls.append(
            {"persona_id": persona_id, "exemplar_id": exemplar_id, "payload": payload}
        )
        return {"id": exemplar_id, "persona_id": persona_id, **payload}

    async def review_persona_exemplar(self, persona_id, exemplar_id, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.review_persona_exemplar_calls.append(
            {"persona_id": persona_id, "exemplar_id": exemplar_id, "payload": payload}
        )
        return {"id": exemplar_id, "persona_id": persona_id, "review": payload}

    async def delete_persona_exemplar(self, persona_id, exemplar_id):
        self.delete_persona_exemplar_calls.append({"persona_id": persona_id, "exemplar_id": exemplar_id})
        return {"status": "deleted", "persona_id": persona_id, "exemplar_id": exemplar_id}

    async def get_character_exemplar(self, character_id, exemplar_id):
        self.get_character_exemplar_calls.append({"character_id": character_id, "exemplar_id": exemplar_id})
        return {"id": exemplar_id, "character_id": character_id, "text": "Use dry wit."}

    async def create_character_exemplar(self, character_id, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.create_character_exemplar_calls.append({"character_id": character_id, "payload": payload})
        return {"id": "char-ex-new", "character_id": character_id, **payload}

    async def update_character_exemplar(self, character_id, exemplar_id, request_data):
        payload = request_data.model_dump(exclude_unset=True, exclude_none=True, mode="json")
        self.update_character_exemplar_calls.append(
            {"character_id": character_id, "exemplar_id": exemplar_id, "payload": payload}
        )
        return {"id": exemplar_id, "character_id": character_id, **payload}

    async def delete_character_exemplar(self, character_id, exemplar_id):
        self.delete_character_exemplar_calls.append({"character_id": character_id, "exemplar_id": exemplar_id})
        return {"status": "deleted", "character_id": character_id, "exemplar_id": exemplar_id}

    async def search_character_exemplars(self, character_id, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.search_character_exemplars_calls.append({"character_id": character_id, "payload": payload})
        return {"items": [{"id": "char-ex-1", "character_id": character_id, "text": payload["query"]}]}

    async def select_character_exemplars_debug(self, character_id, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.select_character_exemplars_debug_calls.append({"character_id": character_id, "payload": payload})
        return {"selected": [{"id": "char-ex-1", "character_id": character_id}], "coverage": {}}

    async def list_greetings(self, chat_id):
        self.list_greetings_calls.append(chat_id)
        return {
            "chat_id": chat_id,
            "greetings": [
                {
                    "index": 0,
                    "text": "Hello there.",
                    "preview": "Hello there.",
                }
            ],
            "current_selection": 0,
        }

    async def select_greeting(self, chat_id, index):
        self.select_greeting_calls.append({"chat_id": chat_id, "index": index})
        return {
            "chat_id": chat_id,
            "selected_index": index,
            "greeting_preview": f"Greeting {index}",
            "checksum_updated": True,
        }

    async def list_presets(self):
        self.list_presets_calls += 1
        return {
            "presets": [
                {
                    "preset_id": "default",
                    "name": "Default",
                    "builtin": True,
                    "section_order": [],
                    "section_templates": {},
                }
            ]
        }

    async def create_preset(self, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.create_preset_calls.append(payload)
        return {
            "preset_id": payload["preset_id"],
            "name": payload["name"],
        }

    async def update_preset(self, preset_id, request_data):
        payload = request_data.model_dump(exclude_unset=True, exclude_none=True, mode="json")
        self.update_preset_calls.append({"preset_id": preset_id, "payload": payload})
        return {
            "preset_id": preset_id,
            "name": payload.get("name", "Updated Preset"),
        }

    async def delete_preset(self, preset_id):
        self.delete_preset_calls.append(preset_id)
        return {
            "status": "deleted",
            "preset_id": preset_id,
        }

    async def create_character_chat_session(self, request_data, **kwargs):
        self.create_character_chat_session_calls.append({"request_data": request_data, "kwargs": kwargs})
        return {"id": "chat-1", "title": "New Chat"}

    async def list_character_chat_sessions(self, **kwargs):
        self.list_character_chat_sessions_calls.append(kwargs)
        return {"chats": [{"id": "chat-1"}], "total": 1, "limit": kwargs.get("limit"), "offset": kwargs.get("offset")}

    async def get_character_chat_session(self, chat_id, **kwargs):
        self.get_character_chat_session_calls.append({"chat_id": chat_id, "kwargs": kwargs})
        return {"id": chat_id, "title": "Existing Chat"}

    async def update_character_chat_session(self, chat_id, request_data, **kwargs):
        self.update_character_chat_session_calls.append(
            {"chat_id": chat_id, "request_data": request_data, "kwargs": kwargs}
        )
        return {"id": chat_id, "title": "Updated Chat"}

    async def delete_character_chat_session(self, chat_id, **kwargs):
        self.delete_character_chat_session_calls.append({"chat_id": chat_id, "kwargs": kwargs})
        return {"status": "deleted", "chat_id": chat_id}

    async def restore_character_chat_session(self, chat_id, **kwargs):
        self.restore_character_chat_session_calls.append({"chat_id": chat_id, "kwargs": kwargs})
        return {"id": chat_id, "deleted": False}

    async def list_character_messages(self, chat_id, **kwargs):
        self.list_character_messages_calls.append({"chat_id": chat_id, "kwargs": kwargs})
        return {"messages": [{"id": "msg-1", "conversation_id": chat_id}], "total": 1}

    async def get_character_message(self, message_id, **kwargs):
        self.get_character_message_calls.append({"message_id": message_id, "kwargs": kwargs})
        return {"id": message_id, "conversation_id": "chat-1", "content": "Hello"}

    async def create_character_message(self, chat_id, request_data, **kwargs):
        self.create_character_message_calls.append(
            {"chat_id": chat_id, "request_data": request_data, "kwargs": kwargs}
        )
        return {"id": "msg-new", "conversation_id": chat_id, "content": "Hello"}

    async def update_character_message(self, message_id, request_data, **kwargs):
        self.update_character_message_calls.append(
            {"message_id": message_id, "request_data": request_data, "kwargs": kwargs}
        )
        return {"id": message_id, "conversation_id": "chat-1", "content": "Updated"}

    async def delete_character_message(self, message_id, **kwargs):
        self.delete_character_message_calls.append({"message_id": message_id, "kwargs": kwargs})
        return {"status": "deleted", "message_id": message_id}

    async def search_character_messages(self, chat_id, query, **kwargs):
        self.search_character_messages_calls.append({"chat_id": chat_id, "query": query, "kwargs": kwargs})
        return {"messages": [{"id": "msg-1", "conversation_id": chat_id}], "total": 1}

    async def get_chat_settings(self, chat_id, **kwargs):
        self.get_chat_settings_calls.append({"chat_id": chat_id, "kwargs": kwargs})
        return {"conversation_id": chat_id, "settings": {}}

    async def update_chat_settings(self, chat_id, request_data, **kwargs):
        self.update_chat_settings_calls.append({"chat_id": chat_id, "request_data": request_data, "kwargs": kwargs})
        return {"conversation_id": chat_id, "settings": getattr(request_data, "settings", {})}

    async def export_chat_history(self, chat_id, **kwargs):
        self.export_chat_history_calls.append({"chat_id": chat_id, "kwargs": kwargs})
        return {"chat_id": chat_id, "format": kwargs.get("format", "json")}

    async def get_author_note_info(self, chat_id):
        self.get_author_note_info_calls.append(chat_id)
        return {"chat_id": chat_id, "text": "note"}

    async def export_lorebook_diagnostics(self, chat_id, **kwargs):
        self.export_lorebook_diagnostics_calls.append({"chat_id": chat_id, "kwargs": kwargs})
        return {"chat_id": chat_id, "turns": []}


class FakeLocalCharacterBackend:
    def __init__(self):
        self.list_character_cards_calls = []
        self.search_characters_calls = []
        self.get_character_calls = []
        self.create_character_calls = []
        self.update_character_calls = []
        self.delete_character_calls = []

    def list_character_cards(self, limit=100, offset=0):
        self.list_character_cards_calls.append({"limit": limit, "offset": offset})
        return [{"id": 2, "name": "Local Ada"}]

    def search_characters(self, query, limit=10):
        self.search_characters_calls.append({"query": query, "limit": limit})
        return [{"id": 2, "name": "Local Ada", "description": query}]

    def get_character(self, character_id):
        self.get_character_calls.append(character_id)
        return {"id": character_id, "name": "Local Ada", "version": 4}

    def create_character(self, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.create_character_calls.append(payload)
        return {"id": 22, "version": 1, **payload}

    def update_character(self, character_id, request_data, expected_version):
        payload = request_data.model_dump(exclude_unset=True, exclude_none=True, mode="json")
        self.update_character_calls.append(
            {"character_id": character_id, "payload": payload, "expected_version": expected_version}
        )
        return {"id": character_id, "version": expected_version + 1, **payload}

    def delete_character(self, character_id, expected_version):
        self.delete_character_calls.append({"character_id": character_id, "expected_version": expected_version})
        return {"status": "deleted", "character_id": character_id}


class FakeLocalCharacterSessionBackend(FakeLocalCharacterBackend):
    def __init__(self):
        super().__init__()
        self.calls = []

    def create_character_chat_session(self, request_data, **kwargs):
        self.calls.append(("create_character_chat_session", request_data, kwargs))
        return {"id": "local-chat-1", "backend": "local"}

    def list_character_chat_sessions(self, **kwargs):
        self.calls.append(("list_character_chat_sessions", kwargs))
        return {"chats": [{"id": "local-chat-1"}], "total": 1}

    def get_character_chat_session(self, chat_id, **kwargs):
        self.calls.append(("get_character_chat_session", chat_id, kwargs))
        return {"id": chat_id, "backend": "local"}

    def update_character_chat_session(self, chat_id, request_data, **kwargs):
        self.calls.append(("update_character_chat_session", chat_id, request_data, kwargs))
        return {"id": chat_id, "title": "Updated Local Chat"}

    def delete_character_chat_session(self, chat_id, **kwargs):
        self.calls.append(("delete_character_chat_session", chat_id, kwargs))
        return {"status": "deleted", "chat_id": chat_id}

    def restore_character_chat_session(self, chat_id, **kwargs):
        self.calls.append(("restore_character_chat_session", chat_id, kwargs))
        return {"id": chat_id, "deleted": False}

    def export_chat_history(self, chat_id, **kwargs):
        self.calls.append(("export_chat_history", chat_id, kwargs))
        return {"chat_id": chat_id, "format": kwargs.get("format", "json")}


class FakeLocalPersonaProfileBackend(FakeLocalCharacterBackend):
    def __init__(self):
        super().__init__()
        self.calls = []

    def list_persona_profiles(self, **kwargs):
        self.calls.append(("list_persona_profiles", kwargs))
        return [{"id": "local-persona-1", "name": "Local Guide"}]

    def get_persona_profile(self, persona_id):
        self.calls.append(("get_persona_profile", persona_id))
        return {"id": persona_id, "name": "Local Guide", "version": 1}

    def create_persona_profile(self, request_data):
        self.calls.append(("create_persona_profile", request_data))
        return {"id": "local-persona-created", "name": "Created", "version": 1}

    def update_persona_profile(self, persona_id, request_data, **kwargs):
        self.calls.append(("update_persona_profile", persona_id, request_data, kwargs))
        return {"id": persona_id, "name": "Updated", "version": kwargs.get("expected_version", 1) + 1}

    def delete_persona_profile(self, persona_id, **kwargs):
        self.calls.append(("delete_persona_profile", persona_id, kwargs))
        return {"status": "deleted", "persona_id": persona_id}

    def restore_persona_profile(self, persona_id, expected_version):
        self.calls.append(("restore_persona_profile", persona_id, expected_version))
        return {"id": persona_id, "deleted": False, "version": expected_version + 1}


class FakePolicyEnforcer:
    def __init__(self, denied_reason: str | None = None):
        self.denied_reason = denied_reason
        self.calls = []

    @classmethod
    def deny(cls, reason_code: str) -> "FakePolicyEnforcer":
        return cls(denied_reason=reason_code)

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)
        if self.denied_reason is None:
            return
        raise PolicyDeniedError(
            action_id=action_id,
            reason_code=self.denied_reason,
            user_message=f"{action_id} denied",
            effective_source="local",
            authority_owner="server",
        )


@pytest.mark.asyncio
async def test_scope_service_routes_to_server_backend_when_mode_is_server():
    local_service = Mock()
    server_service = FakeCharacterPersonaClient()
    scope_service = CharacterPersonaScopeService(
        local_service=local_service,
        server_service=server_service,
    )

    await scope_service.list_persona_profiles(mode="server")

    assert server_service.list_persona_profiles_calls == [
        {
            "active_only": False,
            "include_deleted": False,
            "limit": 100,
            "offset": 0,
        }
    ]
    local_service.list_persona_profiles.assert_not_called()


@pytest.mark.asyncio
async def test_character_persona_scope_service_denies_server_persona_listing_in_local_mode():
    scope_service = CharacterPersonaScopeService(
        local_service=Mock(),
        server_service=Mock(),
        policy_enforcer=FakePolicyEnforcer.deny("wrong_source"),
    )

    with pytest.raises(PolicyDeniedError):
        await scope_service.list_persona_profiles(mode="server")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "local"])
async def test_scope_service_routes_to_local_backend_when_mode_is_local_or_omitted(mode):
    local_service = FakeLocalCharacterBackend()
    server_service = Mock()
    scope_service = CharacterPersonaScopeService(
        local_service=local_service,
        server_service=server_service,
    )

    await scope_service.list_characters(mode=mode)

    assert local_service.list_character_cards_calls == [{"limit": 100, "offset": 0}]
    server_service.list_characters.assert_not_called()


@pytest.mark.asyncio
async def test_scope_service_uses_local_character_cards_fallback():
    local_service = FakeLocalCharacterBackend()
    scope_service = CharacterPersonaScopeService(
        local_service=local_service,
        server_service=Mock(),
    )

    result = await scope_service.list_characters(mode="local", limit=7, offset=3)

    assert result == [{"id": 2, "name": "Local Ada"}]
    assert local_service.list_character_cards_calls == [{"limit": 7, "offset": 3}]


@pytest.mark.asyncio
async def test_scope_service_routes_character_and_persona_parameters():
    local_service = FakeLocalCharacterBackend()
    server_service = FakeCharacterPersonaClient()
    scope_service = CharacterPersonaScopeService(
        local_service=local_service,
        server_service=server_service,
    )

    await scope_service.list_characters(mode="server", limit=7, offset=3)
    await scope_service.list_persona_profiles(
        mode="server",
        active_only=True,
        include_deleted=True,
        limit=11,
        offset=5,
    )

    assert server_service.list_characters_calls == [{"limit": 7, "offset": 3}]
    assert server_service.list_persona_profiles_calls == [
        {
            "active_only": True,
            "include_deleted": True,
            "limit": 11,
            "offset": 5,
        }
    ]
    assert local_service.list_character_cards_calls == []


@pytest.mark.asyncio
async def test_scope_service_routes_character_crud_to_selected_backend():
    local_service = FakeLocalCharacterBackend()
    server_service = FakeCharacterPersonaClient()
    scope_service = CharacterPersonaScopeService(local_service=local_service, server_service=server_service)

    local_search = await scope_service.search_characters("ada", mode="local", limit=4)
    local_detail = await scope_service.get_character(2, mode="local")
    local_created = await scope_service.create_character(
        Mock(model_dump=lambda **_: {"name": "Local New"}),
        mode="local",
    )
    local_updated = await scope_service.update_character(
        2,
        Mock(model_dump=lambda **_: {"name": "Local Ada 2"}),
        expected_version=4,
        mode="local",
    )
    local_deleted = await scope_service.delete_character(2, expected_version=5, mode="local")
    server_search = await scope_service.search_characters("ada", mode="server", limit=3)
    server_detail = await scope_service.get_character(1, mode="server")
    server_created = await scope_service.create_character(
        Mock(model_dump=lambda **_: {"name": "Server New"}),
        mode="server",
    )
    server_updated = await scope_service.update_character(
        1,
        Mock(model_dump=lambda **_: {"name": "Server Ada 2"}),
        expected_version=3,
        mode="server",
    )
    server_deleted = await scope_service.delete_character(1, expected_version=4, mode="server")
    server_restored = await scope_service.restore_character(1, expected_version=5, mode="server")

    assert local_search[0]["id"] == 2
    assert local_detail["id"] == 2
    assert local_created["id"] == 22
    assert local_updated["version"] == 5
    assert local_deleted == {"status": "deleted", "character_id": 2}
    assert server_search[0]["id"] == 1
    assert server_detail["id"] == 1
    assert server_created["id"] == 12
    assert server_updated["version"] == 4
    assert server_deleted == {"status": "deleted", "character_id": 1}
    assert server_restored["deleted"] is False


@pytest.mark.asyncio
async def test_scope_service_routes_persona_profile_crud_to_server_backend():
    scope_service = CharacterPersonaScopeService(
        local_service=FakeLocalCharacterBackend(),
        server_service=FakeCharacterPersonaClient(),
    )

    persona = await scope_service.get_persona_profile("persona-1", mode="server")
    created = await scope_service.create_persona_profile(
        Mock(model_dump=lambda mode="json": {"name": "Guide", "mode": "session_scoped"}),
        mode="server",
    )
    updated = await scope_service.update_persona_profile(
        "persona-1",
        Mock(model_dump=lambda exclude_none=True, mode="json": {"name": "Guide 2"}),
        expected_version=7,
        mode="server",
    )
    deleted = await scope_service.delete_persona_profile("persona-1", expected_version=8, mode="server")
    restored = await scope_service.restore_persona_profile("persona-1", expected_version=9, mode="server")

    assert persona["id"] == "persona-1"
    assert created["name"] == "Guide"
    assert updated["id"] == "persona-1"
    assert deleted == {"status": "deleted", "persona_id": "persona-1"}
    assert restored["id"] == "persona-1"
    assert restored["version"] == 10


@pytest.mark.asyncio
async def test_scope_service_routes_persona_profile_crud_to_local_backend():
    local_service = FakeLocalPersonaProfileBackend()
    policy = FakePolicyEnforcer()
    scope_service = CharacterPersonaScopeService(
        local_service=local_service,
        server_service=FakeCharacterPersonaClient(),
        policy_enforcer=policy,
    )
    create_data = Mock()
    update_data = Mock()

    listed = await scope_service.list_persona_profiles(mode="local", active_only=True, limit=5, offset=2)
    detail = await scope_service.get_persona_profile("local-persona-1", mode="local")
    created = await scope_service.create_persona_profile(create_data, mode="local")
    updated = await scope_service.update_persona_profile(
        "local-persona-1",
        update_data,
        expected_version=1,
        mode="local",
    )
    deleted = await scope_service.delete_persona_profile("local-persona-1", expected_version=2, mode="local")
    restored = await scope_service.restore_persona_profile("local-persona-1", expected_version=3, mode="local")

    assert listed[0]["id"] == "local-persona-1"
    assert detail["id"] == "local-persona-1"
    assert created["id"] == "local-persona-created"
    assert updated["version"] == 2
    assert deleted == {"status": "deleted", "persona_id": "local-persona-1"}
    assert restored["version"] == 4
    assert local_service.calls == [
        ("list_persona_profiles", {"active_only": True, "include_deleted": False, "limit": 5, "offset": 2}),
        ("get_persona_profile", "local-persona-1"),
        ("create_persona_profile", create_data),
        ("update_persona_profile", "local-persona-1", update_data, {"expected_version": 1}),
        ("delete_persona_profile", "local-persona-1", {"expected_version": 2}),
        ("restore_persona_profile", "local-persona-1", 3),
    ]
    assert policy.calls == [
        "character.persona.list.local",
        "character.persona.detail.local",
        "character.persona.create.local",
        "character.persona.update.local",
        "character.persona.delete.local",
        "character.persona.update.local",
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_persona_exemplar_crud_to_server_backend():
    scope_service = CharacterPersonaScopeService(
        local_service=FakeLocalCharacterBackend(),
        server_service=FakeCharacterPersonaClient(),
    )

    exemplars = await scope_service.list_persona_exemplars(
        "persona-1",
        mode="server",
        include_disabled=True,
        include_deleted=True,
        include_deleted_personas=True,
        limit=5,
        offset=2,
    )
    exemplar = await scope_service.get_persona_exemplar("persona-1", "ex-1", mode="server")
    created = await scope_service.create_persona_exemplar(
        "persona-1",
        Mock(model_dump=lambda **_: {"content": "hello"}),
        mode="server",
    )
    imported = await scope_service.import_persona_exemplars(
        "persona-1",
        Mock(model_dump=lambda **_: {"transcript": "hello world"}),
        mode="server",
    )
    updated = await scope_service.update_persona_exemplar(
        "persona-1",
        "ex-1",
        Mock(model_dump=lambda **_: {"content": "updated"}),
        mode="server",
    )
    reviewed = await scope_service.review_persona_exemplar(
        "persona-1",
        "ex-1",
        Mock(model_dump=lambda **_: {"action": "approve"}),
        mode="server",
    )
    deleted = await scope_service.delete_persona_exemplar("persona-1", "ex-1", mode="server")

    assert exemplars[0]["id"] == "ex-1"
    assert exemplar["id"] == "ex-1"
    assert created["id"] == "ex-new"
    assert imported["created"] == 2
    assert updated["content"] == "updated"
    assert reviewed["review"] == {"action": "approve"}
    assert deleted == {"status": "deleted", "persona_id": "persona-1", "exemplar_id": "ex-1"}


@pytest.mark.asyncio
async def test_scope_service_routes_character_exemplar_crud_to_server_backend():
    scope_service = CharacterPersonaScopeService(
        local_service=FakeLocalCharacterBackend(),
        server_service=FakeCharacterPersonaClient(),
    )

    search = await scope_service.search_character_exemplars(
        12,
        Mock(model_dump=lambda **_: {"query": "dry wit"}),
        mode="server",
    )
    exemplar = await scope_service.get_character_exemplar(12, "char-ex-1", mode="server")
    created = await scope_service.create_character_exemplar(
        12,
        Mock(model_dump=lambda **_: {"text": "hello"}),
        mode="server",
    )
    updated = await scope_service.update_character_exemplar(
        12,
        "char-ex-1",
        Mock(model_dump=lambda **_: {"text": "updated"}),
        mode="server",
    )
    debug = await scope_service.select_character_exemplars_debug(
        12,
        Mock(model_dump=lambda **_: {"user_turn": "why?"}),
        mode="server",
    )
    deleted = await scope_service.delete_character_exemplar(12, "char-ex-1", mode="server")

    assert search["items"][0]["id"] == "char-ex-1"
    assert exemplar["id"] == "char-ex-1"
    assert created["id"] == "char-ex-new"
    assert updated["text"] == "updated"
    assert debug["selected"][0]["id"] == "char-ex-1"
    assert deleted == {"status": "deleted", "character_id": 12, "exemplar_id": "char-ex-1"}


@pytest.mark.asyncio
async def test_scope_service_routes_chat_execution_support_to_server_backend():
    server_service = FakeCharacterPersonaClient()
    scope_service = CharacterPersonaScopeService(
        local_service=FakeLocalCharacterBackend(),
        server_service=server_service,
    )

    greetings = await scope_service.list_chat_greetings("chat.server.alice", mode="server")
    selected = await scope_service.select_chat_greeting("chat.server.alice", 0, mode="server")
    presets = await scope_service.list_chat_presets(mode="server")
    created = await scope_service.create_chat_preset(
        Mock(
            model_dump=lambda exclude_none=True, mode="json": {
                "preset_id": "custom-alpha",
                "name": "Custom Alpha",
                "section_order": ["system"],
                "section_templates": {"system": "Be concise."},
            }
        ),
        mode="server",
    )
    updated = await scope_service.update_chat_preset(
        "custom-alpha",
        Mock(model_dump=lambda exclude_unset=True, exclude_none=True, mode="json": {"name": "Custom Beta"}),
        mode="server",
    )
    deleted = await scope_service.delete_chat_preset("custom-alpha", mode="server")

    assert greetings["chat_id"] == "chat.server.alice"
    assert selected["selected_index"] == 0
    assert presets["presets"][0]["preset_id"] == "default"
    assert created["preset_id"] == "custom-alpha"
    assert updated["name"] == "Custom Beta"
    assert deleted["status"] == "deleted"


@pytest.mark.asyncio
async def test_scope_service_routes_character_chat_session_admin_to_server_backend():
    server_service = FakeCharacterPersonaClient()
    scope_service = CharacterPersonaScopeService(
        local_service=FakeLocalCharacterBackend(),
        server_service=server_service,
    )
    request_data = Mock()
    update_data = Mock()
    settings_data = Mock(settings={"authorNote": "Stay concise."})

    created = await scope_service.create_character_chat_session(
        request_data,
        mode="server",
        seed_first_message=True,
        greeting_strategy="alternate_index",
        alternate_index=1,
    )
    listed = await scope_service.list_character_chat_sessions(
        mode="server",
        character_id=12,
        include_settings=True,
        scope_type="workspace",
        workspace_id="ws-1",
    )
    detail = await scope_service.get_character_chat_session("chat-1", mode="server", include_settings=True)
    updated = await scope_service.update_character_chat_session(
        "chat-1",
        update_data,
        mode="server",
        expected_version=4,
    )
    deleted = await scope_service.delete_character_chat_session(
        "chat-1",
        mode="server",
        expected_version=5,
        hard_delete=True,
    )
    restored = await scope_service.restore_character_chat_session("chat-1", mode="server", expected_version=6)
    settings = await scope_service.get_chat_settings("chat-1", mode="server")
    updated_settings = await scope_service.update_chat_settings("chat-1", settings_data, mode="server")
    exported = await scope_service.export_chat_history("chat-1", mode="server", format="markdown")
    author_note = await scope_service.get_author_note_info("chat-1", mode="server")
    diagnostics = await scope_service.export_lorebook_diagnostics("chat-1", mode="server", order="desc")

    assert created["id"] == "chat-1"
    assert listed["total"] == 1
    assert detail["id"] == "chat-1"
    assert updated["title"] == "Updated Chat"
    assert deleted["status"] == "deleted"
    assert restored["deleted"] is False
    assert settings["conversation_id"] == "chat-1"
    assert updated_settings["settings"] == {"authorNote": "Stay concise."}
    assert exported["format"] == "markdown"
    assert author_note["text"] == "note"
    assert diagnostics["turns"] == []
    assert server_service.create_character_chat_session_calls[0]["kwargs"]["seed_first_message"] is True
    assert server_service.list_character_chat_sessions_calls[0]["scope_type"] == "workspace"
    assert server_service.update_character_chat_session_calls[0]["kwargs"]["expected_version"] == 4
    assert server_service.delete_character_chat_session_calls[0]["kwargs"]["hard_delete"] is True


@pytest.mark.asyncio
async def test_scope_service_routes_character_message_admin_to_server_backend():
    server_service = FakeCharacterPersonaClient()
    scope_service = CharacterPersonaScopeService(
        local_service=FakeLocalCharacterBackend(),
        server_service=server_service,
    )
    request_data = Mock()
    update_data = Mock()

    listed = await scope_service.list_character_messages(
        "chat-1",
        mode="server",
        limit=25,
        offset=5,
        include_metadata=True,
        scope_type="workspace",
        workspace_id="ws-1",
    )
    detail = await scope_service.get_character_message("msg-1", mode="server", include_metadata=True)
    created = await scope_service.create_character_message(
        "chat-1",
        request_data,
        mode="server",
        scope_type="workspace",
        workspace_id="ws-1",
    )
    updated = await scope_service.update_character_message(
        "msg-1",
        update_data,
        mode="server",
        expected_version=2,
    )
    deleted = await scope_service.delete_character_message("msg-1", mode="server", expected_version=3)
    results = await scope_service.search_character_messages("chat-1", "hello", mode="server", limit=10)

    assert listed["total"] == 1
    assert detail["id"] == "msg-1"
    assert created["id"] == "msg-new"
    assert updated["content"] == "Updated"
    assert deleted == {"status": "deleted", "message_id": "msg-1"}
    assert results["total"] == 1
    assert server_service.list_character_messages_calls[0] == {
        "chat_id": "chat-1",
        "kwargs": {
            "limit": 25,
            "offset": 5,
            "include_metadata": True,
            "scope_type": "workspace",
            "workspace_id": "ws-1",
        },
    }
    assert server_service.create_character_message_calls[0]["kwargs"]["scope_type"] == "workspace"
    assert server_service.update_character_message_calls[0]["kwargs"]["expected_version"] == 2
    assert server_service.delete_character_message_calls[0]["kwargs"]["expected_version"] == 3
    assert server_service.search_character_messages_calls[0]["query"] == "hello"


@pytest.mark.asyncio
async def test_scope_service_routes_character_chat_session_admin_to_local_backend():
    local_service = FakeLocalCharacterSessionBackend()
    policy = FakePolicyEnforcer()
    scope_service = CharacterPersonaScopeService(
        local_service=local_service,
        server_service=FakeCharacterPersonaClient(),
        policy_enforcer=policy,
    )
    request_data = Mock()
    update_data = Mock()

    created = await scope_service.create_character_chat_session(request_data, mode="local")
    listed = await scope_service.list_character_chat_sessions(mode="local", character_id=12)
    detail = await scope_service.get_character_chat_session("local-chat-1", mode="local")
    updated = await scope_service.update_character_chat_session(
        "local-chat-1",
        update_data,
        mode="local",
        expected_version=2,
    )
    deleted = await scope_service.delete_character_chat_session("local-chat-1", mode="local", expected_version=3)
    restored = await scope_service.restore_character_chat_session("local-chat-1", mode="local", expected_version=4)
    exported = await scope_service.export_chat_history("local-chat-1", mode="local", format="markdown")

    assert created["id"] == "local-chat-1"
    assert listed["total"] == 1
    assert detail["backend"] == "local"
    assert updated["title"] == "Updated Local Chat"
    assert deleted["status"] == "deleted"
    assert restored["deleted"] is False
    assert exported["format"] == "markdown"
    assert local_service.calls == [
        ("create_character_chat_session", request_data, {}),
        ("list_character_chat_sessions", {"character_id": 12}),
        ("get_character_chat_session", "local-chat-1", {}),
        ("update_character_chat_session", "local-chat-1", update_data, {"expected_version": 2}),
        ("delete_character_chat_session", "local-chat-1", {"expected_version": 3}),
        ("restore_character_chat_session", "local-chat-1", {"expected_version": 4}),
        ("export_chat_history", "local-chat-1", {"format": "markdown"}),
    ]
    assert policy.calls == [
        "character.sessions.create.local",
        "character.sessions.list.local",
        "character.sessions.detail.local",
        "character.sessions.update.local",
        "character.sessions.delete.local",
        "character.sessions.restore.local",
        "character.sessions.export.local",
    ]


@pytest.mark.asyncio
async def test_scope_service_raises_when_local_backend_lacks_persona_method():
    class LocalBackend:
        def list_characters(self, limit=100, offset=0):
            return []

    scope_service = CharacterPersonaScopeService(
        local_service=LocalBackend(),
        server_service=Mock(),
    )

    with pytest.raises(ValueError, match="Local persona profiles are not available yet"):
        await scope_service.list_persona_profiles(mode="local")

    with pytest.raises(ValueError, match="Local persona profiles are not available yet"):
        await scope_service.get_persona_profile("persona-1", mode="local")


@pytest.mark.asyncio
async def test_scope_service_raises_when_local_backend_lacks_chat_execution_support():
    scope_service = CharacterPersonaScopeService(
        local_service=FakeLocalCharacterBackend(),
        server_service=Mock(),
    )

    with pytest.raises(ValueError, match="Local chat greetings are not available yet"):
        await scope_service.list_chat_greetings("chat.local.alice", mode="local")

    with pytest.raises(ValueError, match="Local chat presets are not available yet"):
        await scope_service.list_chat_presets(mode="local")


@pytest.mark.asyncio
async def test_scope_service_rejects_invalid_mode():
    scope_service = CharacterPersonaScopeService(
        local_service=FakeLocalCharacterBackend(),
        server_service=Mock(),
    )

    with pytest.raises(ValueError, match="Invalid character/persona mode"):
        await scope_service.list_characters(mode="bogus")


def test_scope_service_reports_known_character_persona_capability_gaps():
    scope_service = CharacterPersonaScopeService(
        local_service=FakeLocalCharacterBackend(),
        server_service=FakeCharacterPersonaClient(),
    )

    local_report = scope_service.list_unsupported_capabilities(mode="local")
    server_report = scope_service.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "character.persona.profiles.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_scope_missing",
            "user_message": (
                "Local persona profile CRUD is still handled by older CCP/local chat paths and is not wrapped by "
                "the source-aware character/persona scope yet."
            ),
            "affected_action_ids": [
                "character.persona.create.local",
                "character.persona.delete.local",
                "character.persona.detail.local",
                "character.persona.list.local",
                "character.persona.update.local",
            ],
        },
        {
            "operation_id": "character.sessions.execution.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_scope_missing",
            "user_message": (
                "Local character greetings, presets, settings, and lorebook diagnostics still use legacy local CCP "
                "flows instead of this source-aware scope."
            ),
            "affected_action_ids": [
                "character.sessions.detail.local",
                "character.sessions.launch.local",
                "character.sessions.observe.local",
                "character.sessions.update.local",
            ],
        },
        {
            "operation_id": "character.persona.exemplars.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_scope_missing",
            "user_message": (
                "Local persona exemplar CRUD is not available through the source-aware character/persona scope yet."
            ),
            "affected_action_ids": [
                "character.persona.create.local",
                "character.persona.delete.local",
                "character.persona.detail.local",
                "character.persona.list.local",
                "character.persona.update.local",
            ],
        },
        {
            "operation_id": "character.exemplars.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_scope_missing",
            "user_message": (
                "Local character exemplar CRUD is not available through the source-aware character/persona scope yet."
            ),
            "affected_action_ids": [
                "character.persona.create.local",
                "character.persona.delete.local",
                "character.persona.detail.local",
                "character.persona.list.local",
                "character.persona.update.local",
            ],
        },
    ]
    assert server_report == []


def test_scope_service_does_not_report_local_session_crud_when_backend_wraps_it():
    scope_service = CharacterPersonaScopeService(
        local_service=FakeLocalCharacterSessionBackend(),
        server_service=FakeCharacterPersonaClient(),
    )

    operation_ids = {
        item["operation_id"]
        for item in scope_service.list_unsupported_capabilities(mode="local")
    }

    assert "character.sessions.admin.local" not in operation_ids
    assert "character.sessions.execution.local" in operation_ids


def test_scope_service_does_not_report_local_persona_profiles_when_backend_wraps_them():
    scope_service = CharacterPersonaScopeService(
        local_service=FakeLocalPersonaProfileBackend(),
        server_service=FakeCharacterPersonaClient(),
    )

    operation_ids = {
        item["operation_id"]
        for item in scope_service.list_unsupported_capabilities(mode="local")
    }

    assert "character.persona.profiles.local" not in operation_ids
    assert "character.persona.exemplars.local" in operation_ids


@pytest.mark.asyncio
async def test_scope_service_requires_local_backend_for_local_calls():
    scope_service = CharacterPersonaScopeService(
        local_service=None,
        server_service=Mock(),
    )

    with pytest.raises(ValueError, match="Local character/persona backend is unavailable"):
        await scope_service.list_characters(mode="local")


@pytest.mark.asyncio
async def test_server_character_persona_service_delegates_to_client():
    client = FakeCharacterPersonaClient()
    service = ServerCharacterPersonaService(client=client)

    characters = await service.list_characters(limit=13, offset=4)
    persona_profiles = await service.list_persona_profiles(
        active_only=True,
        include_deleted=True,
        limit=25,
        offset=9,
    )

    assert characters == [{"id": 1, "name": "Ada"}]
    assert persona_profiles == [{"id": "persona-1", "name": "Guide"}]
    assert client.list_characters_calls == [{"limit": 13, "offset": 4}]
    assert client.list_persona_profiles_calls == [
        {
            "active_only": True,
            "include_deleted": True,
            "limit": 25,
            "offset": 9,
        }
    ]


@pytest.mark.asyncio
async def test_server_character_persona_service_delegates_character_crud_to_client():
    client = FakeCharacterPersonaClient()
    service = ServerCharacterPersonaService(client=client)

    searched = await service.search_characters("ada", limit=4)
    detail = await service.get_character(1)
    created = await service.create_character(Mock(model_dump=lambda **_: {"name": "Ada"}))
    updated = await service.update_character(1, Mock(model_dump=lambda **_: {"name": "Ada v2"}), expected_version=3)
    deleted = await service.delete_character(1, expected_version=4)
    restored = await service.restore_character(1, expected_version=5)

    assert searched[0]["id"] == 1
    assert detail["id"] == 1
    assert created["id"] == 12
    assert updated["version"] == 4
    assert deleted == {"status": "deleted", "character_id": 1}
    assert restored["deleted"] is False
    assert client.search_characters_calls == [{"query": "ada", "limit": 4}]
    assert client.update_character_calls[0]["expected_version"] == 3


@pytest.mark.asyncio
async def test_server_character_persona_service_delegates_chat_execution_support_to_client():
    client = FakeCharacterPersonaClient()
    service = ServerCharacterPersonaService(client=client)

    greetings = await service.list_chat_greetings("chat.server.alice")
    selected = await service.select_chat_greeting("chat.server.alice", 0)
    presets = await service.list_chat_presets()
    created = await service.create_chat_preset(
        Mock(
            model_dump=lambda exclude_none=True, mode="json": {
                "preset_id": "custom-alpha",
                "name": "Custom Alpha",
                "section_order": ["system"],
                "section_templates": {"system": "Be concise."},
            }
        )
    )
    updated = await service.update_chat_preset(
        "custom-alpha",
        Mock(model_dump=lambda exclude_unset=True, exclude_none=True, mode="json": {"name": "Custom Beta"}),
    )
    deleted = await service.delete_chat_preset("custom-alpha")

    assert greetings["chat_id"] == "chat.server.alice"
    assert selected["selected_index"] == 0
    assert presets["presets"][0]["preset_id"] == "default"
    assert created["preset_id"] == "custom-alpha"
    assert updated["preset_id"] == "custom-alpha"
    assert deleted["status"] == "deleted"


@pytest.mark.asyncio
async def test_server_character_persona_service_delegates_persona_exemplars_to_client():
    client = FakeCharacterPersonaClient()
    service = ServerCharacterPersonaService(client=client)

    exemplars = await service.list_persona_exemplars("persona-1", include_disabled=True, limit=5, offset=2)
    exemplar = await service.get_persona_exemplar("persona-1", "ex-1")
    created = await service.create_persona_exemplar(
        "persona-1",
        Mock(model_dump=lambda **_: {"content": "hello"}),
    )
    imported = await service.import_persona_exemplars(
        "persona-1",
        Mock(model_dump=lambda **_: {"transcript": "hello world"}),
    )
    updated = await service.update_persona_exemplar(
        "persona-1",
        "ex-1",
        Mock(model_dump=lambda **_: {"content": "updated"}),
    )
    reviewed = await service.review_persona_exemplar(
        "persona-1",
        "ex-1",
        Mock(model_dump=lambda **_: {"action": "approve"}),
    )
    deleted = await service.delete_persona_exemplar("persona-1", "ex-1")

    assert exemplars[0]["id"] == "ex-1"
    assert exemplar["id"] == "ex-1"
    assert created["id"] == "ex-new"
    assert imported["created"] == 2
    assert updated["content"] == "updated"
    assert reviewed["review"] == {"action": "approve"}
    assert deleted == {"status": "deleted", "persona_id": "persona-1", "exemplar_id": "ex-1"}
    assert client.list_persona_exemplars_calls[0]["include_disabled"] is True


@pytest.mark.asyncio
async def test_server_character_persona_service_delegates_character_exemplars_to_client():
    client = FakeCharacterPersonaClient()
    service = ServerCharacterPersonaService(client=client)

    search = await service.search_character_exemplars(
        12,
        Mock(model_dump=lambda **_: {"query": "dry wit"}),
    )
    exemplar = await service.get_character_exemplar(12, "char-ex-1")
    created = await service.create_character_exemplar(
        12,
        Mock(model_dump=lambda **_: {"text": "hello"}),
    )
    updated = await service.update_character_exemplar(
        12,
        "char-ex-1",
        Mock(model_dump=lambda **_: {"text": "updated"}),
    )
    debug = await service.select_character_exemplars_debug(
        12,
        Mock(model_dump=lambda **_: {"user_turn": "why?"}),
    )
    deleted = await service.delete_character_exemplar(12, "char-ex-1")

    assert search["items"][0]["id"] == "char-ex-1"
    assert exemplar["id"] == "char-ex-1"
    assert created["id"] == "char-ex-new"
    assert updated["text"] == "updated"
    assert debug["selected"][0]["id"] == "char-ex-1"
    assert deleted == {"status": "deleted", "character_id": 12, "exemplar_id": "char-ex-1"}
    assert client.search_character_exemplars_calls[0]["payload"] == {"query": "dry wit"}


@pytest.mark.asyncio
async def test_server_character_persona_service_delegates_character_chat_session_admin_to_client():
    client = FakeCharacterPersonaClient()
    service = ServerCharacterPersonaService(client=client)
    request_data = Mock()
    update_data = Mock()
    settings_data = Mock(settings={"authorNote": "Stay concise."})

    created = await service.create_character_chat_session(request_data, seed_first_message=True)
    listed = await service.list_character_chat_sessions(character_id=12, include_settings=True)
    detail = await service.get_character_chat_session("chat-1", include_settings=True)
    updated = await service.update_character_chat_session("chat-1", update_data, expected_version=4)
    deleted = await service.delete_character_chat_session("chat-1", expected_version=5, hard_delete=True)
    restored = await service.restore_character_chat_session("chat-1", expected_version=6)
    messages = await service.list_character_messages("chat-1", include_metadata=True)
    message = await service.get_character_message("msg-1", include_metadata=True)
    created_message = await service.create_character_message("chat-1", Mock(), scope_type="workspace")
    updated_message = await service.update_character_message("msg-1", Mock(), expected_version=2)
    deleted_message = await service.delete_character_message("msg-1", expected_version=3)
    search_results = await service.search_character_messages("chat-1", "hello", limit=10)
    settings = await service.get_chat_settings("chat-1")
    updated_settings = await service.update_chat_settings("chat-1", settings_data)
    exported = await service.export_chat_history("chat-1", format="markdown")
    author_note = await service.get_author_note_info("chat-1")
    diagnostics = await service.export_lorebook_diagnostics("chat-1", order="desc")

    assert created["id"] == "chat-1"
    assert listed["total"] == 1
    assert detail["id"] == "chat-1"
    assert updated["title"] == "Updated Chat"
    assert deleted["status"] == "deleted"
    assert restored["deleted"] is False
    assert messages["total"] == 1
    assert message["id"] == "msg-1"
    assert created_message["id"] == "msg-new"
    assert updated_message["content"] == "Updated"
    assert deleted_message == {"status": "deleted", "message_id": "msg-1"}
    assert search_results["total"] == 1
    assert settings["conversation_id"] == "chat-1"
    assert updated_settings["settings"] == {"authorNote": "Stay concise."}
    assert exported["format"] == "markdown"
    assert author_note["text"] == "note"
    assert diagnostics["turns"] == []
    assert client.create_character_chat_session_calls[0]["kwargs"]["seed_first_message"] is True
    assert client.list_character_chat_sessions_calls[0]["character_id"] == 12
    assert client.update_character_chat_session_calls[0]["kwargs"]["expected_version"] == 4
    assert client.delete_character_chat_session_calls[0]["kwargs"]["hard_delete"] is True
    assert client.create_character_message_calls[0]["kwargs"]["scope_type"] == "workspace"
    assert client.update_character_message_calls[0]["kwargs"]["expected_version"] == 2


@pytest.mark.asyncio
async def test_server_character_persona_service_enforces_policy_actions():
    client = FakeCharacterPersonaClient()
    policy = Mock()
    service = ServerCharacterPersonaService(client=client, policy_enforcer=policy)

    await service.list_characters(limit=13, offset=4)
    await service.search_characters("ada")
    await service.get_character(1)
    await service.create_character(Mock(model_dump=lambda **_: {"name": "Ada"}))
    await service.update_character(1, Mock(model_dump=lambda **_: {"name": "Ada v2"}), expected_version=3)
    await service.delete_character(1, expected_version=4)
    await service.restore_character(1, expected_version=5)
    await service.list_persona_profiles(active_only=True)
    await service.get_persona_profile("persona-1")
    await service.create_persona_profile(Mock(model_dump=lambda **_: {"name": "Guide"}))
    await service.update_persona_profile("persona-1", Mock(model_dump=lambda **_: {"name": "Guide v2"}))
    await service.delete_persona_profile("persona-1", expected_version=8)
    await service.restore_persona_profile("persona-1", expected_version=9)
    await service.list_persona_exemplars("persona-1")
    await service.get_persona_exemplar("persona-1", "ex-1")
    await service.create_persona_exemplar("persona-1", Mock(model_dump=lambda **_: {"content": "hello"}))
    await service.import_persona_exemplars("persona-1", Mock(model_dump=lambda **_: {"transcript": "hello"}))
    await service.update_persona_exemplar("persona-1", "ex-1", Mock(model_dump=lambda **_: {"content": "updated"}))
    await service.review_persona_exemplar("persona-1", "ex-1", Mock(model_dump=lambda **_: {"action": "approve"}))
    await service.delete_persona_exemplar("persona-1", "ex-1")
    await service.search_character_exemplars(12, Mock(model_dump=lambda **_: {"query": "dry wit"}))
    await service.get_character_exemplar(12, "char-ex-1")
    await service.create_character_exemplar(12, Mock(model_dump=lambda **_: {"text": "hello"}))
    await service.update_character_exemplar(12, "char-ex-1", Mock(model_dump=lambda **_: {"text": "updated"}))
    await service.select_character_exemplars_debug(12, Mock(model_dump=lambda **_: {"user_turn": "why?"}))
    await service.delete_character_exemplar(12, "char-ex-1")
    await service.list_chat_greetings("chat.server.alice")
    await service.select_chat_greeting("chat.server.alice", 0)
    await service.list_chat_presets()
    await service.create_chat_preset(
        Mock(model_dump=lambda **_: {"preset_id": "custom-alpha", "name": "Custom Alpha"})
    )
    await service.update_chat_preset("custom-alpha", Mock(model_dump=lambda **_: {"name": "Custom Beta"}))
    await service.delete_chat_preset("custom-alpha")
    await service.create_character_chat_session(Mock())
    await service.list_character_chat_sessions()
    await service.get_character_chat_session("chat.server.alice")
    await service.update_character_chat_session("chat.server.alice", Mock(), expected_version=3)
    await service.delete_character_chat_session("chat.server.alice", expected_version=4)
    await service.restore_character_chat_session("chat.server.alice", expected_version=5)
    await service.list_character_messages("chat.server.alice")
    await service.get_character_message("msg.server.1")
    await service.create_character_message("chat.server.alice", Mock())
    await service.update_character_message("msg.server.1", Mock(), expected_version=6)
    await service.delete_character_message("msg.server.1", expected_version=7)
    await service.search_character_messages("chat.server.alice", "hello")
    await service.get_chat_settings("chat.server.alice")
    await service.update_chat_settings("chat.server.alice", Mock())
    await service.export_chat_history("chat.server.alice")
    await service.get_author_note_info("chat.server.alice")
    await service.export_lorebook_diagnostics("chat.server.alice")

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "character.persona.list.server",
        "character.persona.list.server",
        "character.persona.detail.server",
        "character.persona.create.server",
        "character.persona.update.server",
        "character.persona.delete.server",
        "character.persona.update.server",
        "character.persona.list.server",
        "character.persona.detail.server",
        "character.persona.create.server",
        "character.persona.update.server",
        "character.persona.delete.server",
        "character.persona.update.server",
        "character.persona.list.server",
        "character.persona.detail.server",
        "character.persona.create.server",
        "character.persona.create.server",
        "character.persona.update.server",
        "character.persona.update.server",
        "character.persona.delete.server",
        "character.persona.list.server",
        "character.persona.detail.server",
        "character.persona.create.server",
        "character.persona.update.server",
        "character.persona.list.server",
        "character.persona.delete.server",
        "character.sessions.launch.server",
        "character.sessions.launch.server",
        "character.persona.list.server",
        "character.persona.create.server",
        "character.persona.update.server",
        "character.persona.delete.server",
        "character.sessions.create.server",
        "character.sessions.list.server",
        "character.sessions.detail.server",
        "character.sessions.update.server",
        "character.sessions.delete.server",
        "character.sessions.restore.server",
        "character.messages.list.server",
        "character.messages.detail.server",
        "character.messages.create.server",
        "character.messages.update.server",
        "character.messages.delete.server",
        "character.messages.list.server",
        "character.sessions.detail.server",
        "character.sessions.update.server",
        "character.sessions.export.server",
        "character.sessions.detail.server",
        "character.sessions.observe.server",
    ]


@pytest.mark.asyncio
async def test_server_character_persona_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeCharacterPersonaClient()
    service = ServerCharacterPersonaService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_characters(limit=13, offset=4)

    assert exc.value.reason_code == "server_unreachable"
    assert client.list_characters_calls == []


def test_server_character_persona_service_from_config_uses_api_client(monkeypatch):
    sentinel_client = Mock()
    build_client = Mock(return_value=sentinel_client)

    monkeypatch.setattr(
        "tldw_chatbook.Character_Chat.server_character_persona_service.build_runtime_api_client_from_config",
        build_client,
    )

    service = ServerCharacterPersonaService.from_config({"tldw_api": {"base_url": "https://example.com"}})

    assert service.client is sentinel_client
    build_client.assert_called_once_with({"tldw_api": {"base_url": "https://example.com"}})


def test_app_wires_character_persona_services(monkeypatch):
    from tldw_chatbook import app as app_module

    server_service = Mock()
    captured = {}

    monkeypatch.setattr(
        app_module.ServerCharacterPersonaService,
        "from_config",
        Mock(return_value=server_service),
    )
    original_scope_service = app_module.CharacterPersonaScopeService

    def scope_service_factory(*, local_service, server_service, policy_enforcer=None):
        captured["local_service"] = local_service
        captured["server_service"] = server_service
        captured["policy_enforcer"] = policy_enforcer
        return original_scope_service(
            local_service=local_service,
            server_service=server_service,
            policy_enforcer=policy_enforcer,
        )

    monkeypatch.setattr(app_module, "CharacterPersonaScopeService", scope_service_factory)

    fake_app = Mock()
    fake_app.app_config = {"tldw_api": {"base_url": "https://example.com"}}
    fake_app.chachanotes_db = object()

    app_module.TldwCli._wire_character_persona_services(fake_app)

    assert fake_app.server_character_persona_service is server_service
    assert isinstance(captured["local_service"], app_module.LocalCharacterPersonaService)
    assert captured["local_service"].db is fake_app.chachanotes_db
    assert fake_app.local_character_persona_service is captured["local_service"]
    assert captured["server_service"] is server_service
    assert captured["policy_enforcer"] is fake_app.service_policy_enforcer
