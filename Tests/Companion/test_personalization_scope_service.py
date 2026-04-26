import pytest

from tldw_chatbook.Personalization_Interop import PersonalizationScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeServerPersonalizationService:
    def __init__(self):
        self.calls = []

    async def get_profile(self):
        self.calls.append(("get_profile",))
        return {"backend": "server", "record_id": "server:personalization_profile"}

    async def set_opt_in(self, request_data):
        self.calls.append(("set_opt_in", request_data))
        return {"backend": "server", "record_id": "server:personalization_profile"}

    async def update_preferences(self, request_data):
        self.calls.append(("update_preferences", request_data))
        return {"backend": "server", "record_id": "server:personalization_preferences"}

    async def purge_data(self):
        self.calls.append(("purge_data",))
        return {"backend": "server", "record_id": "server:personalization_lifecycle:purge"}

    async def list_memories(self, **kwargs):
        self.calls.append(("list_memories", kwargs))
        return {"backend": "server", "record_id": "server:personalization_memories"}

    async def export_memories(self):
        self.calls.append(("export_memories",))
        return {"backend": "server", "record_id": "server:personalization_memories:export"}

    async def get_memory(self, memory_id):
        self.calls.append(("get_memory", memory_id))
        return {"backend": "server", "record_id": f"server:personalization_memory:{memory_id}"}

    async def create_memory(self, request_data):
        self.calls.append(("create_memory", request_data))
        return {"backend": "server", "record_id": "server:personalization_memory:mem-created"}

    async def update_memory(self, memory_id, request_data):
        self.calls.append(("update_memory", memory_id, request_data))
        return {"backend": "server", "record_id": f"server:personalization_memory:{memory_id}"}

    async def delete_memory(self, memory_id):
        self.calls.append(("delete_memory", memory_id))
        return {"backend": "server", "record_id": f"server:personalization_memory:{memory_id}"}

    async def validate_memories(self, request_data):
        self.calls.append(("validate_memories", request_data))
        return {"backend": "server", "record_id": "server:personalization_memories:validate"}

    async def import_memories(self, request_data):
        self.calls.append(("import_memories", request_data))
        return {"backend": "server", "record_id": "server:personalization_memories:import"}

    async def list_explanations(self, **kwargs):
        self.calls.append(("list_explanations", kwargs))
        return {"backend": "server", "record_id": "server:personalization_explanations"}


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_personalization_scope_service_routes_server_surface():
    server = FakeServerPersonalizationService()
    policy = FakePolicyEnforcer()
    scope = PersonalizationScopeService(server_service=server, policy_enforcer=policy)

    await scope.get_profile(mode="server")
    await scope.set_opt_in({"enabled": True}, mode="server")
    await scope.update_preferences({"response_style": "concise"}, mode="server")
    await scope.purge_data(mode="server")
    await scope.list_memories(mode="server", q="memory")
    await scope.export_memories(mode="server")
    await scope.get_memory("mem-1", mode="server")
    await scope.create_memory({"content": "Created"}, mode="server")
    await scope.update_memory("mem-1", {"content": "Updated"}, mode="server")
    await scope.delete_memory("mem-1", mode="server")
    await scope.validate_memories({"memory_ids": ["mem-1"]}, mode="server")
    await scope.import_memories({"memories": [{"content": "Imported"}]}, mode="server")
    await scope.list_explanations(mode="server", limit=5)

    assert server.calls == [
        ("get_profile",),
        ("set_opt_in", {"enabled": True}),
        ("update_preferences", {"response_style": "concise"}),
        ("purge_data",),
        ("list_memories", {"q": "memory"}),
        ("export_memories",),
        ("get_memory", "mem-1"),
        ("create_memory", {"content": "Created"}),
        ("update_memory", "mem-1", {"content": "Updated"}),
        ("delete_memory", "mem-1"),
        ("validate_memories", {"memory_ids": ["mem-1"]}),
        ("import_memories", {"memories": [{"content": "Imported"}]}),
        ("list_explanations", {"limit": 5}),
    ]
    assert policy.calls == [
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
async def test_personalization_scope_service_honestly_rejects_local_mode():
    server = FakeServerPersonalizationService()
    scope = PersonalizationScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Personalization profile operations are server-only"):
        await scope.get_profile(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_personalization_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeServerPersonalizationService()
    scope = PersonalizationScopeService(
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("server_unreachable"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.update_preferences({"response_style": "concise"}, mode="server")

    assert exc.value.reason_code == "server_unreachable"
    assert server.calls == []


def test_personalization_scope_service_reports_known_unsupported_capabilities():
    scope = PersonalizationScopeService(server_service=FakeServerPersonalizationService())

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "personalization.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Server personalization profile, preferences, and purge controls are unavailable in local/offline mode.",
            "affected_action_ids": [
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
            ],
        }
    ]
    assert server_report == []
