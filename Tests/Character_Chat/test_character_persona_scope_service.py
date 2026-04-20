from unittest.mock import Mock

import pytest

from tldw_chatbook.Character_Chat.character_persona_scope_service import (
    CharacterPersonaScopeService,
)
from tldw_chatbook.Character_Chat.server_character_persona_service import (
    ServerCharacterPersonaService,
)


class FakeCharacterPersonaClient:
    def __init__(self):
        self.list_characters_calls = []
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

    async def list_characters(self, limit=100, offset=0):
        self.list_characters_calls.append({"limit": limit, "offset": offset})
        return [{"id": 1, "name": "Ada"}]

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


class FakeLocalCharacterBackend:
    def __init__(self):
        self.list_character_cards_calls = []

    def list_character_cards(self, limit=100, offset=0):
        self.list_character_cards_calls.append({"limit": limit, "offset": offset})
        return [{"id": 2, "name": "Local Ada"}]


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

    def scope_service_factory(*, local_service, server_service):
        captured["local_service"] = local_service
        captured["server_service"] = server_service
        return original_scope_service(local_service=local_service, server_service=server_service)

    monkeypatch.setattr(app_module, "CharacterPersonaScopeService", scope_service_factory)

    fake_app = Mock()
    fake_app.app_config = {"tldw_api": {"base_url": "https://example.com"}}
    fake_app.chachanotes_db = object()

    app_module.TldwCli._wire_character_persona_services(fake_app)

    assert fake_app.server_character_persona_service is server_service
    assert captured["local_service"] is fake_app.chachanotes_db
    assert captured["server_service"] is server_service
