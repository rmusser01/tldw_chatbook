import pytest

from tldw_chatbook.Companion_Interop import CompanionScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeServerCompanionService:
    def __init__(self):
        self.calls = []

    async def create_activity(self, request_data):
        self.calls.append(("create_activity", request_data))
        return {"backend": "server", "record_id": "server:companion_activity:act-1"}

    async def create_check_in(self, request_data):
        self.calls.append(("create_check_in", request_data))
        return {"backend": "server", "record_id": "server:companion_activity:act-check-in"}

    async def list_activity(self, **kwargs):
        self.calls.append(("list_activity", kwargs))
        return {"backend": "server", "record_id": "server:companion_activity"}

    async def get_activity(self, event_id):
        self.calls.append(("get_activity", event_id))
        return {"backend": "server", "record_id": f"server:companion_activity:{event_id}"}

    async def list_knowledge(self, **kwargs):
        self.calls.append(("list_knowledge", kwargs))
        return {"backend": "server", "record_id": "server:companion_knowledge"}

    async def get_knowledge(self, card_id):
        self.calls.append(("get_knowledge", card_id))
        return {"backend": "server", "record_id": f"server:companion_knowledge:{card_id}"}

    async def get_reflection(self, reflection_id):
        self.calls.append(("get_reflection", reflection_id))
        return {"backend": "server", "record_id": f"server:companion_reflection:{reflection_id}"}

    async def get_conversation_prompts(self, **kwargs):
        self.calls.append(("get_conversation_prompts", kwargs))
        return {"backend": "server", "record_id": "server:companion_conversation_prompts"}

    async def list_goals(self, **kwargs):
        self.calls.append(("list_goals", kwargs))
        return {"backend": "server", "record_id": "server:companion_goals"}

    async def create_goal(self, request_data):
        self.calls.append(("create_goal", request_data))
        return {"backend": "server", "record_id": "server:companion_goal:goal-1"}

    async def update_goal(self, goal_id, request_data):
        self.calls.append(("update_goal", goal_id, request_data))
        return {"backend": "server", "record_id": f"server:companion_goal:{goal_id}"}

    async def purge_data(self, request_data):
        self.calls.append(("purge_data", request_data))
        return {"backend": "server", "record_id": "server:companion_lifecycle:knowledge"}

    async def rebuild_data(self, request_data):
        self.calls.append(("rebuild_data", request_data))
        return {"backend": "server", "record_id": "server:companion_lifecycle:reflections"}


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
async def test_companion_scope_service_routes_server_surface():
    server = FakeServerCompanionService()
    policy = FakePolicyEnforcer()
    scope = CompanionScopeService(server_service=server, policy_enforcer=policy)

    await scope.create_activity({"event_type": "note.updated"}, mode="server")
    await scope.create_check_in({"summary": "Read"}, mode="server")
    await scope.list_activity(mode="server", limit=25, offset=5)
    await scope.get_activity("act-1", mode="server")
    await scope.list_knowledge(mode="server", status="active")
    await scope.get_knowledge("card-1", mode="server")
    await scope.get_reflection("reflection-1", mode="server")
    await scope.get_conversation_prompts(mode="server", query="progress")
    await scope.list_goals(mode="server", status="active")
    await scope.create_goal({"title": "Read", "goal_type": "habit"}, mode="server")
    await scope.update_goal("goal-1", {"title": "Read more"}, mode="server")
    await scope.purge_data({"scope": "knowledge"}, mode="server")
    await scope.rebuild_data({"scope": "reflections"}, mode="server")

    assert server.calls[0] == ("create_activity", {"event_type": "note.updated"})
    assert server.calls[-1] == ("rebuild_data", {"scope": "reflections"})
    assert policy.calls == [
        "companion.activity.create.server",
        "companion.checkins.create.server",
        "companion.activity.list.server",
        "companion.activity.detail.server",
        "companion.knowledge.list.server",
        "companion.knowledge.detail.server",
        "companion.reflections.detail.server",
        "companion.conversation_prompts.list.server",
        "companion.goals.list.server",
        "companion.goals.create.server",
        "companion.goals.update.server",
        "companion.lifecycle.purge.server",
        "companion.lifecycle.launch.server",
    ]


@pytest.mark.asyncio
async def test_companion_scope_service_honestly_rejects_local_mode():
    server = FakeServerCompanionService()
    scope = CompanionScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Companion server operations are server-only"):
        await scope.list_activity(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_companion_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeServerCompanionService()
    scope = CompanionScopeService(
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("server_unreachable"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.create_check_in({"summary": "Read"}, mode="server")

    assert exc.value.reason_code == "server_unreachable"
    assert server.calls == []


def test_companion_scope_service_reports_known_unsupported_capabilities():
    scope = CompanionScopeService(server_service=FakeServerCompanionService())

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "companion.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Companion activity, knowledge, goals, prompts, and lifecycle controls are unavailable in local/offline mode.",
            "affected_action_ids": [
                "companion.activity.create.server",
                "companion.checkins.create.server",
                "companion.activity.list.server",
                "companion.activity.detail.server",
                "companion.knowledge.list.server",
                "companion.knowledge.detail.server",
                "companion.reflections.detail.server",
                "companion.conversation_prompts.list.server",
                "companion.goals.list.server",
                "companion.goals.create.server",
                "companion.goals.update.server",
                "companion.lifecycle.purge.server",
                "companion.lifecycle.launch.server",
            ],
        }
    ]
    assert server_report == []
