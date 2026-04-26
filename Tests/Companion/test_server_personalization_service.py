from unittest.mock import Mock

import pytest

from tldw_chatbook.Personalization_Interop import ServerPersonalizationService
from tldw_chatbook.runtime_policy import PolicyDecision, PolicyDeniedError


PROFILE_PAYLOAD = {
    "enabled": True,
    "response_style": "balanced",
    "preferred_format": "auto",
    "companion_reflections_enabled": True,
}


class FakePersonalizationClient:
    def __init__(self):
        self.calls = []

    async def get_personalization_profile(self):
        self.calls.append(("get_personalization_profile",))
        return PROFILE_PAYLOAD

    async def set_personalization_opt_in(self, request_data):
        self.calls.append(("set_personalization_opt_in", request_data))
        return PROFILE_PAYLOAD | {"enabled": request_data.enabled}

    async def update_personalization_preferences(self, request_data):
        self.calls.append(("update_personalization_preferences", request_data))
        return PROFILE_PAYLOAD | {"response_style": request_data.response_style}

    async def purge_personalization_data(self):
        self.calls.append(("purge_personalization_data",))
        return {"status": "ok", "deleted_counts": {"memories": 1}, "enabled": False}

    async def list_personalization_memories(self, **kwargs):
        self.calls.append(("list_personalization_memories", kwargs))
        return {"items": [{"id": "mem-1", "type": "semantic", "content": "Memory"}], "total": 1}

    async def export_personalization_memories(self):
        self.calls.append(("export_personalization_memories",))
        return {"memories": [{"id": "mem-1", "content": "Memory"}], "total": 1}

    async def get_personalization_memory(self, memory_id):
        self.calls.append(("get_personalization_memory", memory_id))
        return {"id": memory_id, "type": "semantic", "content": "Memory"}

    async def create_personalization_memory(self, request_data):
        self.calls.append(("create_personalization_memory", request_data))
        return {"id": "mem-created", "type": "semantic", "content": request_data.content}

    async def update_personalization_memory(self, memory_id, request_data):
        self.calls.append(("update_personalization_memory", memory_id, request_data))
        return {"id": memory_id, "type": "semantic", "content": request_data.content}

    async def delete_personalization_memory(self, memory_id):
        self.calls.append(("delete_personalization_memory", memory_id))
        return {"detail": "ok: deleted"}

    async def validate_personalization_memories(self, request_data):
        self.calls.append(("validate_personalization_memories", request_data))
        return {"detail": "ok: validated 1 memories"}

    async def import_personalization_memories(self, request_data):
        self.calls.append(("import_personalization_memories", request_data))
        return {"detail": "ok: imported 1 memories"}

    async def list_personalization_explanations(self, **kwargs):
        self.calls.append(("list_personalization_explanations", kwargs))
        return {"items": [], "total": 0}


@pytest.mark.asyncio
async def test_server_personalization_service_delegates_and_normalizes_records():
    client = FakePersonalizationClient()
    service = ServerPersonalizationService(client=client)

    profile = await service.get_profile()
    opt_in = await service.set_opt_in({"enabled": False})
    preferences = await service.update_preferences({"response_style": "concise"})
    purged = await service.purge_data()

    assert profile["record_id"] == "server:personalization_profile"
    assert opt_in["enabled"] is False
    assert opt_in["record_id"] == "server:personalization_profile"
    assert preferences["response_style"] == "concise"
    assert preferences["record_id"] == "server:personalization_preferences"
    assert purged["record_id"] == "server:personalization_lifecycle:purge"
    assert all(item["backend"] == "server" for item in [profile, opt_in, preferences, purged])
    assert client.calls[0] == ("get_personalization_profile",)
    assert client.calls[-1] == ("purge_personalization_data",)


@pytest.mark.asyncio
async def test_server_personalization_service_delegates_memory_and_explanation_routes():
    client = FakePersonalizationClient()
    service = ServerPersonalizationService(client=client)

    memories = await service.list_memories(memory_type="semantic", q="memory", page=2, size=25, include_hidden=True)
    exported = await service.export_memories()
    detail = await service.get_memory("mem-1")
    created = await service.create_memory({"content": "Created"})
    updated = await service.update_memory("mem-1", {"content": "Updated"})
    deleted = await service.delete_memory("mem-1")
    validated = await service.validate_memories({"memory_ids": ["mem-1"]})
    imported = await service.import_memories({"memories": [{"content": "Imported"}]})
    explanations = await service.list_explanations(limit=5)

    assert memories["record_id"] == "server:personalization_memories"
    assert exported["record_id"] == "server:personalization_memories:export"
    assert detail["record_id"] == "server:personalization_memory:mem-1"
    assert created["record_id"] == "server:personalization_memory:mem-created"
    assert updated["record_id"] == "server:personalization_memory:mem-1"
    assert deleted["record_id"] == "server:personalization_memory:mem-1"
    assert validated["record_id"] == "server:personalization_memories:validate"
    assert imported["record_id"] == "server:personalization_memories:import"
    assert explanations["record_id"] == "server:personalization_explanations"
    assert all(item["backend"] == "server" for item in [
        memories,
        exported,
        detail,
        created,
        updated,
        deleted,
        validated,
        imported,
        explanations,
    ])
    assert client.calls[0] == (
        "list_personalization_memories",
        {
            "memory_type": "semantic",
            "q": "memory",
            "page": 2,
            "size": 25,
            "include_hidden": True,
        },
    )


@pytest.mark.asyncio
async def test_server_personalization_service_enforces_policy_actions():
    client = FakePersonalizationClient()
    policy = Mock()
    service = ServerPersonalizationService(client=client, policy_enforcer=policy)

    await service.get_profile()
    await service.set_opt_in({"enabled": True})
    await service.update_preferences({"response_style": "concise"})
    await service.purge_data()
    await service.list_memories()
    await service.export_memories()
    await service.get_memory("mem-1")
    await service.create_memory({"content": "Created"})
    await service.update_memory("mem-1", {"content": "Updated"})
    await service.delete_memory("mem-1")
    await service.validate_memories({"memory_ids": ["mem-1"]})
    await service.import_memories({"memories": [{"content": "Imported"}]})
    await service.list_explanations()

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "personalization.profile.detail.server",
        "personalization.opt_in.update.server",
        "personalization.preferences.update.server",
        "personalization.lifecycle.purge.server",
        "personalization.memories.list.server",
        "personalization.memories.export.server",
        "personalization.memories.detail.server",
        "personalization.memories.create.server",
        "personalization.memories.update.server",
        "personalization.memories.delete.server",
        "personalization.memories.validate.server",
        "personalization.memories.import.server",
        "personalization.explanations.list.server",
    ]


@pytest.mark.asyncio
async def test_server_personalization_service_hard_stops_denied_ui_policy_decision():
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
    client = FakePersonalizationClient()
    service = ServerPersonalizationService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.get_profile()

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []


def test_server_personalization_service_from_config_uses_api_client(monkeypatch):
    sentinel_client = Mock()
    build_client = Mock(return_value=sentinel_client)

    monkeypatch.setattr(
        "tldw_chatbook.Personalization_Interop.server_personalization_service.build_runtime_api_client_from_config",
        build_client,
    )

    service = ServerPersonalizationService.from_config(
        {"tldw_api": {"base_url": "https://example.com"}}
    )

    assert service.client is sentinel_client
    build_client.assert_called_once_with({"tldw_api": {"base_url": "https://example.com"}})
