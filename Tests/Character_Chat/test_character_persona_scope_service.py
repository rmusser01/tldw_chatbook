from unittest.mock import Mock

import pytest

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Character_Chat.character_persona_scope_service import (
    CharacterPersonaScopeService,
)
from tldw_chatbook.Character_Chat.server_character_persona_service import (
    ServerCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class FakeCharacterPersonaClient:
    def __init__(self):
        self.list_characters_calls = []
        self.character_calls = []
        self.list_persona_profiles_calls = []
        self.get_persona_profile_calls = []
        self.create_persona_profile_calls = []
        self.update_persona_profile_calls = []
        self.list_greetings_calls = []
        self.select_greeting_calls = []
        self.list_presets_calls = 0
        self.create_preset_calls = []
        self.update_preset_calls = []
        self.delete_preset_calls = []
        self.session_calls = []
        self.message_calls = []
        self.memory_calls = []

    @staticmethod
    def _payload_dict(payload):
        if hasattr(payload, "model_dump"):
            return payload.model_dump(exclude_none=True, mode="json")
        return dict(payload)

    async def list_characters(self, limit=100, offset=0):
        self.list_characters_calls.append({"limit": limit, "offset": offset})
        return [{"id": 1, "name": "Ada"}]

    async def get_character(self, character_id):
        self.character_calls.append(("detail", character_id))
        return {"id": character_id, "name": "Ada"}

    async def create_character(self, payload):
        payload = self._payload_dict(payload)
        self.character_calls.append(("create", payload))
        return {"id": 42, "name": payload.get("name"), "version": 1}

    async def update_character(self, character_id, payload, expected_version):
        payload = self._payload_dict(payload)
        self.character_calls.append(("update", character_id, payload, expected_version))
        return {"id": character_id, "name": payload.get("name", "Ada"), "version": expected_version + 1}

    async def delete_character(self, character_id, expected_version):
        self.character_calls.append(("delete", character_id, expected_version))
        return {"deleted": True, "id": character_id}

    async def restore_character(self, character_id, expected_version):
        self.character_calls.append(("restore", character_id, expected_version))
        return {"id": character_id, "deleted": False, "version": expected_version + 1}

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

    async def create_character_chat_session(self, payload, **kwargs):
        payload = self._payload_dict(payload)
        self.session_calls.append(("create", payload, kwargs))
        return {"id": "chat-1", "title": payload.get("title")}

    async def list_character_chat_sessions(self, **kwargs):
        self.session_calls.append(("list", kwargs))
        return {"chats": [{"id": "chat-1"}], "total": 1}

    async def get_character_chat_session(self, chat_id, **kwargs):
        self.session_calls.append(("detail", chat_id, kwargs))
        return {"id": chat_id}

    async def update_character_chat_session(self, chat_id, payload, *, expected_version, **kwargs):
        payload = self._payload_dict(payload)
        self.session_calls.append(("update", chat_id, payload, expected_version, kwargs))
        return {"id": chat_id, "title": payload.get("title"), "version": expected_version + 1}

    async def delete_character_chat_session(self, chat_id, **kwargs):
        self.session_calls.append(("delete", chat_id, kwargs))
        return {"deleted": True}

    async def restore_character_chat_session(self, chat_id, **kwargs):
        self.session_calls.append(("restore", chat_id, kwargs))
        return {"id": chat_id, "deleted": False}

    async def get_character_chat_settings(self, chat_id, **kwargs):
        self.session_calls.append(("settings-detail", chat_id, kwargs))
        return {"conversation_id": chat_id, "settings": {}}

    async def update_character_chat_settings(self, chat_id, payload, **kwargs):
        payload = self._payload_dict(payload)
        self.session_calls.append(("settings-update", chat_id, payload, kwargs))
        return {"conversation_id": chat_id, "settings": payload.get("settings", {})}

    async def create_character_chat_message(self, chat_id, payload, **kwargs):
        payload = self._payload_dict(payload)
        self.message_calls.append(("create", chat_id, payload, kwargs))
        return {"id": "msg-1", "conversation_id": chat_id, "content": payload.get("content")}

    async def list_character_chat_messages(self, chat_id, **kwargs):
        self.message_calls.append(("list", chat_id, kwargs))
        return {"messages": [{"id": "msg-1", "conversation_id": chat_id}], "total": 1}

    async def get_character_chat_message(self, message_id, **kwargs):
        self.message_calls.append(("detail", message_id, kwargs))
        return {"id": message_id}

    async def update_character_chat_message(self, message_id, payload, *, expected_version, **kwargs):
        payload = self._payload_dict(payload)
        self.message_calls.append(("update", message_id, payload, expected_version, kwargs))
        return {"id": message_id, "version": expected_version + 1}

    async def delete_character_chat_message(self, message_id, **kwargs):
        self.message_calls.append(("delete", message_id, kwargs))
        return {"deleted": True}

    async def search_character_chat_messages(self, chat_id, query, **kwargs):
        self.message_calls.append(("search", chat_id, query, kwargs))
        return {"messages": [], "total": 0}

    async def list_character_memories(self, character_id, **kwargs):
        self.memory_calls.append(("list", character_id, kwargs))
        return {"memories": [], "total": 0}

    async def create_character_memory(self, character_id, payload):
        payload = self._payload_dict(payload)
        self.memory_calls.append(("create", character_id, payload))
        return {"id": "mem-1", "character_id": character_id, "content": payload.get("content")}

    async def update_character_memory(self, character_id, memory_id, payload):
        payload = self._payload_dict(payload)
        self.memory_calls.append(("update", character_id, memory_id, payload))
        return {"id": memory_id, "character_id": character_id, "content": payload.get("content")}

    async def delete_character_memory(self, character_id, memory_id):
        self.memory_calls.append(("delete", character_id, memory_id))
        return {"deleted": True}

    async def archive_character_memory(self, character_id, memory_id, payload):
        payload = self._payload_dict(payload)
        self.memory_calls.append(("archive", character_id, memory_id, payload))
        return {"id": memory_id, "archived": payload.get("archived")}

    async def extract_character_memories(self, character_id, payload):
        payload = self._payload_dict(payload)
        self.memory_calls.append(("extract", character_id, payload))
        return {"extracted": 1, "skipped_duplicates": 0, "memories": []}


class FakeLocalCharacterBackend:
    def __init__(self):
        self.list_character_cards_calls = []

    def list_character_cards(self, limit=100, offset=0):
        self.list_character_cards_calls.append({"limit": limit, "offset": offset})
        return [{"id": 2, "name": "Local Ada"}]


class FakePolicyEnforcer:
    def __init__(self):
        self.actions = []

    def require_allowed(self, *, action_id):
        self.actions.append(action_id)


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
async def test_scope_service_routes_character_catalog_crud_to_server_with_policy():
    server_service = FakeCharacterPersonaClient()
    policy = FakePolicyEnforcer()
    scope_service = CharacterPersonaScopeService(
        local_service=FakeLocalCharacterBackend(),
        server_service=server_service,
        policy_enforcer=policy,
    )

    detail = await scope_service.get_character(12, mode="server")
    created = await scope_service.create_character({"name": "Ada"}, mode="server")
    updated = await scope_service.update_character(12, {"name": "Ada v2"}, expected_version=3, mode="server")
    deleted = await scope_service.delete_character(12, expected_version=4, mode="server")
    restored = await scope_service.restore_character(12, expected_version=5, mode="server")

    assert detail["id"] == 12
    assert created["id"] == 42
    assert updated["version"] == 4
    assert deleted["deleted"] is True
    assert restored["deleted"] is False
    assert policy.actions == [
        "character.persona.detail.server",
        "character.persona.create.server",
        "character.persona.update.server",
        "character.persona.delete.server",
        "character.persona.update.server",
    ]
    assert server_service.character_calls == [
        ("detail", 12),
        ("create", {"name": "Ada"}),
        ("update", 12, {"name": "Ada v2"}, 3),
        ("delete", 12, 4),
        ("restore", 12, 5),
    ]


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

    assert persona["id"] == "persona-1"
    assert created["name"] == "Guide"
    assert updated["id"] == "persona-1"


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
async def test_scope_service_routes_server_ccp_sessions_messages_and_memory_with_policy():
    server_service = FakeCharacterPersonaClient()
    policy = FakePolicyEnforcer()
    scope_service = CharacterPersonaScopeService(
        local_service=FakeLocalCharacterBackend(),
        server_service=server_service,
        policy_enforcer=policy,
    )

    await scope_service.create_character_chat_session(
        {"character_id": 12, "title": "Ada"},
        mode="server",
        seed_first_message=True,
    )
    await scope_service.list_character_chat_sessions(mode="server", character_id=12)
    await scope_service.get_character_chat_session("chat-1", mode="server")
    await scope_service.update_character_chat_session(
        "chat-1",
        {"title": "Ada v2"},
        expected_version=4,
        mode="server",
    )
    await scope_service.delete_character_chat_session("chat-1", expected_version=5, mode="server")
    await scope_service.restore_character_chat_session("chat-1", expected_version=6, mode="server")
    await scope_service.get_character_chat_settings("chat-1", mode="server")
    await scope_service.update_character_chat_settings("chat-1", {"settings": {"presetScope": "chat"}}, mode="server")
    await scope_service.create_character_chat_message("chat-1", {"role": "user", "content": "Hello"}, mode="server")
    await scope_service.list_character_chat_messages("chat-1", mode="server")
    await scope_service.get_character_chat_message("msg-1", mode="server")
    await scope_service.update_character_chat_message("msg-1", {"content": "Updated"}, expected_version=7, mode="server")
    await scope_service.delete_character_chat_message("msg-1", expected_version=8, mode="server")
    await scope_service.search_character_chat_messages("chat-1", "hello", mode="server")
    await scope_service.list_character_memories("12", mode="server", include_archived=True)
    await scope_service.create_character_memory("12", {"content": "likes tea"}, mode="server")
    await scope_service.update_character_memory("12", "mem-1", {"content": "likes coffee"}, mode="server")
    await scope_service.archive_character_memory("12", "mem-1", {"archived": True}, mode="server")
    await scope_service.delete_character_memory("12", "mem-1", mode="server")
    await scope_service.extract_character_memories("12", {"chat_id": "chat-1"}, mode="server")

    assert policy.actions == [
        "character.sessions.create.server",
        "character.sessions.list.server",
        "character.sessions.detail.server",
        "character.sessions.update.server",
        "character.sessions.delete.server",
        "character.sessions.update.server",
        "character.sessions.detail.server",
        "character.sessions.update.server",
        "character.messages.create.server",
        "character.messages.list.server",
        "character.messages.detail.server",
        "character.messages.update.server",
        "character.messages.delete.server",
        "character.messages.list.server",
        "character.memory.list.server",
        "character.memory.create.server",
        "character.memory.update.server",
        "character.memory.update.server",
        "character.memory.delete.server",
        "character.memory.launch.server",
    ]
    assert server_service.session_calls[0][0] == "create"
    assert server_service.message_calls[0][0] == "create"
    assert server_service.memory_calls[0][0] == "list"


@pytest.mark.asyncio
async def test_scope_service_routes_local_ccp_sessions_messages_and_memory_with_policy():
    policy = FakePolicyEnforcer()
    local_service = FakeCharacterPersonaClient()
    scope_service = CharacterPersonaScopeService(
        local_service=local_service,
        server_service=FakeCharacterPersonaClient(),
        policy_enforcer=policy,
    )

    await scope_service.create_character_chat_session({"character_id": 12, "title": "Ada"}, mode="local")
    await scope_service.create_character_chat_message("chat-1", {"role": "user", "content": "Hi"}, mode="local")
    await scope_service.list_character_memories("12", mode="local")

    assert policy.actions == [
        "character.sessions.create.local",
        "character.messages.create.local",
        "character.memory.list.local",
    ]
    assert local_service.session_calls[0][0] == "create"
    assert local_service.message_calls[0][0] == "create"
    assert local_service.memory_calls[0][0] == "list"


def test_local_character_persona_service_persists_sessions_messages_settings_and_memory(tmp_path):
    db = CharactersRAGDB(tmp_path / "ccp.sqlite", "test_client")
    character_id = db.add_character_card(
        {
            "name": "Local Ada",
            "description": "Offline character",
            "first_message": "Hello.",
        }
    )
    service = LocalCharacterPersonaService(db)

    session = service.create_character_chat_session(
        {
            "character_id": character_id,
            "title": "Local Ada Chat",
        }
    )
    listed_sessions = service.list_character_chat_sessions(character_id=character_id)
    updated_settings = service.update_character_chat_settings(
        session["id"],
        {"settings": {"temperature": 0.4}},
    )
    fetched_settings = service.get_character_chat_settings(session["id"])
    message = service.create_character_chat_message(
        session["id"],
        {"role": "user", "content": "Hello"},
    )
    listed_messages = service.list_character_chat_messages(session["id"])
    updated_message = service.update_character_chat_message(
        message["id"],
        {"content": "Updated"},
        expected_version=message["version"],
    )
    memory = service.create_character_memory(str(character_id), {"content": "likes tea"})
    listed_memories = service.list_character_memories(str(character_id))
    archived_memory = service.archive_character_memory(
        str(character_id),
        memory["id"],
        {"archived": True},
    )

    assert session["id"]
    assert session["character_id"] == character_id
    assert session["runtime_backend"] == "local"
    assert session["discovery_owner"] == "ccp_character"
    assert listed_sessions["total"] == 1
    assert updated_settings["settings"] == {"temperature": 0.4}
    assert fetched_settings["settings"] == {"temperature": 0.4}
    assert message["conversation_id"] == session["id"]
    assert listed_messages["total"] == 1
    assert updated_message["content"] == "Updated"
    assert memory["content"] == "likes tea"
    assert listed_memories["total"] == 1
    assert archived_memory["archived"] is True

    service.create_character_chat_message(
        session["id"],
        {"role": "user", "content": "Remember that I keep a travel notebook."},
    )
    extracted = service.extract_character_memories(str(character_id), {"chat_id": session["id"], "message_limit": 5})
    duplicate_extract = service.extract_character_memories(str(character_id), {"chat_id": session["id"], "message_limit": 5})

    assert extracted["extracted"] == 1
    assert extracted["skipped_duplicates"] == 0
    assert extracted["memories"][0]["memory_type"] == "extracted"
    assert extracted["memories"][0]["content"] == "I keep a travel notebook."
    assert duplicate_extract["extracted"] == 0
    assert duplicate_extract["skipped_duplicates"] == 1

    assert service.delete_character_memory(str(character_id), memory["id"]) == {"deleted": True}


def test_local_character_persona_service_persists_character_catalog_crud_and_restore(tmp_path):
    db = CharactersRAGDB(tmp_path / "ccp-character-catalog.sqlite", "test_client")
    service = LocalCharacterPersonaService(db)

    created = service.create_character({"name": "Local Ada", "first_message": "Hello."})
    fetched = service.get_character(created["id"])
    updated = service.update_character(
        created["id"],
        {"name": "Local Ada v2", "tags": ["offline"]},
        expected_version=created["version"],
    )
    deleted = service.delete_character(created["id"], expected_version=updated["version"])

    with pytest.raises(ValueError, match="Local character '.*' not found"):
        service.get_character(created["id"])

    restored = service.restore_character(created["id"], expected_version=updated["version"] + 1)

    assert created["name"] == "Local Ada"
    assert fetched["id"] == created["id"]
    assert updated["name"] == "Local Ada v2"
    assert updated["tags"] == ["offline"]
    assert deleted == {"deleted": True, "id": str(created["id"])}
    assert restored["name"] == "Local Ada v2"
    assert restored["version"] == updated["version"] + 2
    db.close_connection()


@pytest.mark.asyncio
async def test_scope_service_routes_local_world_book_crud_with_policy(tmp_path):
    db = CharactersRAGDB(tmp_path / "ccp-world-books.sqlite", "test_client")
    policy = FakePolicyEnforcer()
    scope_service = CharacterPersonaScopeService(
        local_service=LocalCharacterPersonaService(db),
        server_service=FakeCharacterPersonaClient(),
        policy_enforcer=policy,
    )

    created = await scope_service.create_character_world_book(
        {
            "name": "Shared Lore",
            "description": "Reusable context",
            "scan_depth": 4,
            "token_budget": 900,
            "recursive_scanning": True,
        },
        mode="local",
    )
    listed = await scope_service.list_character_world_books(mode="local", include_disabled=True)
    fetched = await scope_service.get_character_world_book(created["id"], mode="local")
    updated = await scope_service.update_character_world_book(
        created["id"],
        {"description": "Updated context", "enabled": False},
        expected_version=created["version"],
        mode="local",
    )
    deleted = await scope_service.delete_character_world_book(
        created["id"],
        expected_version=updated["version"],
        mode="local",
    )

    assert created["name"] == "Shared Lore"
    assert created["recursive_scanning"] is True
    assert listed["world_books"][0]["id"] == created["id"]
    assert fetched["description"] == "Reusable context"
    assert updated["description"] == "Updated context"
    assert updated["enabled"] is False
    assert deleted == {"deleted": True, "id": str(created["id"])}
    assert policy.actions == [
        "character.world_books.create.local",
        "character.world_books.list.local",
        "character.world_books.detail.local",
        "character.world_books.update.local",
        "character.world_books.delete.local",
    ]
    db.close_connection()


@pytest.mark.asyncio
async def test_scope_service_routes_local_world_book_entries_with_policy(tmp_path):
    db = CharactersRAGDB(tmp_path / "ccp-world-book-entries.sqlite", "test_client")
    policy = FakePolicyEnforcer()
    scope_service = CharacterPersonaScopeService(
        local_service=LocalCharacterPersonaService(db),
        server_service=FakeCharacterPersonaClient(),
        policy_enforcer=policy,
    )
    world_book = await scope_service.create_character_world_book({"name": "Entry Lore"}, mode="local")
    policy.actions.clear()

    entry = await scope_service.create_character_world_book_entry(
        world_book["id"],
        {"keys": ["gate"], "content": "The old gate is sealed."},
        mode="local",
    )
    listed = await scope_service.list_character_world_book_entries(world_book["id"], mode="local")
    fetched = await scope_service.get_character_world_book_entry(entry["id"], mode="local")
    updated = await scope_service.update_character_world_book_entry(
        entry["id"],
        {"content": "The old gate is open."},
        mode="local",
    )
    deleted = await scope_service.delete_character_world_book_entry(entry["id"], mode="local")

    assert listed["entries"][0]["id"] == entry["id"]
    assert fetched["content"] == "The old gate is sealed."
    assert updated["content"] == "The old gate is open."
    assert deleted == {"deleted": True, "id": str(entry["id"])}
    assert policy.actions == [
        "character.world_book_entries.create.local",
        "character.world_book_entries.list.local",
        "character.world_book_entries.detail.local",
        "character.world_book_entries.update.local",
        "character.world_book_entries.delete.local",
    ]
    db.close_connection()


def test_local_character_persona_service_persists_world_book_entries_links_and_import_export(tmp_path):
    db = CharactersRAGDB(tmp_path / "ccp-world-book-links.sqlite", "test_client")
    character_id = db.add_character_card({"name": "Lorekeeper", "first_message": "Ask."})
    service = LocalCharacterPersonaService(db)
    session = service.create_character_chat_session({"character_id": character_id, "title": "Lore Chat"})

    world_book = service.create_character_world_book({"name": "City Lore", "description": "Places"})
    entry = service.create_character_world_book_entry(
        world_book["id"],
        {
            "keys": ["city", "district"],
            "content": "The city is built in rings.",
            "position": "before_char",
            "secondary_keys": ["ring"],
            "selective": True,
        },
    )
    listed_entries = service.list_character_world_book_entries(world_book["id"])
    updated_entry = service.update_character_world_book_entry(
        entry["id"],
        {"content": "The city is built in seven rings.", "enabled": False},
    )
    service.attach_character_world_book_to_session(session["id"], world_book["id"], {"priority": 7})
    linked = service.list_session_world_books(session["id"], include_disabled=True)
    exported = service.export_character_world_book(world_book["id"])
    imported = service.import_character_world_book(exported, name_override="City Lore Copy")
    service.detach_character_world_book_from_session(session["id"], world_book["id"])
    after_detach = service.list_session_world_books(session["id"], include_disabled=True)
    deleted_entry = service.delete_character_world_book_entry(entry["id"])

    assert listed_entries["entries"][0]["keys"] == ["city", "district"]
    assert updated_entry["content"] == "The city is built in seven rings."
    assert updated_entry["enabled"] is False
    assert linked["world_books"][0]["id"] == world_book["id"]
    assert linked["world_books"][0]["priority"] == 7
    assert exported["name"] == "City Lore"
    assert imported["name"] == "City Lore Copy"
    assert after_detach["world_books"] == []
    assert deleted_entry == {"deleted": True, "id": str(entry["id"])}
    db.close_connection()


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
async def test_server_character_persona_service_delegates_character_catalog_crud_to_client():
    client = FakeCharacterPersonaClient()
    service = ServerCharacterPersonaService(client=client)

    detail = await service.get_character(12)
    created = await service.create_character(Mock(model_dump=lambda exclude_none=True, mode="json": {"name": "Ada"}))
    updated = await service.update_character(
        12,
        Mock(model_dump=lambda exclude_unset=True, exclude_none=True, mode="json": {"name": "Ada v2"}),
        expected_version=3,
    )
    deleted = await service.delete_character(12, expected_version=4)
    restored = await service.restore_character(12, expected_version=5)

    assert detail["id"] == 12
    assert created["name"] == "Ada"
    assert updated["version"] == 4
    assert deleted["deleted"] is True
    assert restored["deleted"] is False
    assert client.character_calls == [
        ("detail", 12),
        ("create", {"name": "Ada"}),
        ("update", 12, {"name": "Ada v2"}, 3),
        ("delete", 12, 4),
        ("restore", 12, 5),
    ]


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
async def test_server_character_persona_service_delegates_ccp_session_message_and_memory_to_client():
    client = FakeCharacterPersonaClient()
    service = ServerCharacterPersonaService(client=client)

    session = await service.create_character_chat_session({"character_id": 12, "title": "Ada"})
    message = await service.create_character_chat_message("chat-1", {"role": "user", "content": "Hello"})
    memory = await service.create_character_memory("12", {"content": "likes tea"})

    assert session["id"] == "chat-1"
    assert message["id"] == "msg-1"
    assert memory["id"] == "mem-1"
    assert client.session_calls[0][0] == "create"
    assert client.message_calls[0][0] == "create"
    assert client.memory_calls[0][0] == "create"


def test_server_character_persona_service_from_config_uses_api_client(monkeypatch):
    sentinel_client = Mock()
    build_client = Mock(return_value=sentinel_client)

    monkeypatch.setattr(
        "tldw_chatbook.Character_Chat.server_character_persona_service.build_tldw_api_client_from_config",
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
    fake_app.service_policy_enforcer = object()

    app_module.TldwCli._wire_character_persona_services(fake_app)

    assert fake_app.server_character_persona_service is server_service
    assert isinstance(captured["local_service"], LocalCharacterPersonaService)
    assert captured["local_service"].db is fake_app.chachanotes_db
    assert captured["server_service"] is server_service
    assert captured["policy_enforcer"] is fake_app.service_policy_enforcer
