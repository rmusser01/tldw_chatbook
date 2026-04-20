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


@pytest.mark.asyncio
async def test_scope_service_routes_to_server_backend_when_mode_is_server():
    local_service = Mock()
    server_service = Mock()
    scope_service = CharacterPersonaScopeService(
        local_service=local_service,
        server_service=server_service,
    )

    await scope_service.list_persona_profiles(mode="server")

    server_service.list_persona_profiles.assert_called_once_with(
        active_only=False,
        include_deleted=False,
        limit=100,
        offset=0,
    )
    local_service.list_persona_profiles.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "local"])
async def test_scope_service_routes_to_local_backend_when_mode_is_local_or_omitted(mode):
    local_service = Mock()
    server_service = Mock()
    scope_service = CharacterPersonaScopeService(
        local_service=local_service,
        server_service=server_service,
    )

    await scope_service.list_characters(mode=mode)

    local_service.list_characters.assert_called_once_with(limit=100, offset=0)
    server_service.list_characters.assert_not_called()


@pytest.mark.asyncio
async def test_scope_service_routes_character_and_persona_parameters():
    local_service = Mock()
    server_service = Mock()
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

    server_service.list_characters.assert_called_once_with(limit=7, offset=3)
    server_service.list_persona_profiles.assert_called_once_with(
        active_only=True,
        include_deleted=True,
        limit=11,
        offset=5,
    )
    local_service.list_characters.assert_not_called()
    local_service.list_persona_profiles.assert_not_called()


@pytest.mark.asyncio
async def test_scope_service_raises_when_local_backend_lacks_persona_method():
    class LocalBackend:
        def list_characters(self, limit=100, offset=0):
            return []

    scope_service = CharacterPersonaScopeService(
        local_service=LocalBackend(),
        server_service=Mock(),
    )

    with pytest.raises(AttributeError, match="list_persona_profiles"):
        await scope_service.list_persona_profiles(mode="local")


@pytest.mark.asyncio
async def test_server_character_persona_service_delegates_to_client():
    client = FakeCharacterPersonaClient()
    service = ServerCharacterPersonaService(client=client)

    characters = await service.list_characters()
    persona_profiles = await service.list_persona_profiles(
        active_only=True,
        include_deleted=True,
        limit=25,
        offset=9,
    )

    assert characters == [{"id": 1, "name": "Ada"}]
    assert persona_profiles == [{"id": "persona-1", "name": "Guide"}]
    assert client.list_characters_calls == [{"limit": 100, "offset": 0}]
    assert client.list_persona_profiles_calls == [
        {
            "active_only": True,
            "include_deleted": True,
            "limit": 25,
            "offset": 9,
        }
    ]


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
    notes_server_service = Mock()
    captured = {}

    monkeypatch.setattr(
        app_module.ServerCharacterPersonaService,
        "from_config",
        Mock(return_value=server_service),
    )
    monkeypatch.setattr(
        app_module.ServerNotesWorkspaceService,
        "from_config",
        Mock(return_value=notes_server_service),
    )
    original_scope_service = app_module.CharacterPersonaScopeService

    def scope_service_factory(*, local_service, server_service):
        captured["local_service"] = local_service
        captured["server_service"] = server_service
        return original_scope_service(local_service=local_service, server_service=server_service)

    monkeypatch.setattr(app_module, "CharacterPersonaScopeService", scope_service_factory)

    app = app_module.TldwCli()

    assert app.server_character_persona_service is server_service
    assert captured["local_service"] is app.chachanotes_db
    assert captured["server_service"] is server_service
