from unittest.mock import Mock

import pytest

from tldw_chatbook.Companion_Interop import ServerCompanionService
from tldw_chatbook.runtime_policy import PolicyDecision, PolicyDeniedError


ACTIVITY_REQUEST = {
    "event_type": "note.updated",
    "source_type": "note",
    "source_id": "note-1",
    "surface": "notes",
    "provenance": {"note_id": "note-1"},
}


class FakeCompanionClient:
    def __init__(self):
        self.calls = []

    async def create_companion_activity(self, request_data):
        self.calls.append(("create_companion_activity", request_data))
        return {"id": "act-1", "event_type": "note.updated"}

    async def create_companion_check_in(self, request_data):
        self.calls.append(("create_companion_check_in", request_data))
        return {"id": "act-check-in", "event_type": "manual_check_in"}

    async def list_companion_activity(self, **kwargs):
        self.calls.append(("list_companion_activity", kwargs))
        return {"items": [], "total": 0, "limit": kwargs.get("limit"), "offset": kwargs.get("offset")}

    async def get_companion_activity(self, event_id):
        self.calls.append(("get_companion_activity", event_id))
        return {"id": event_id, "event_type": "note.updated"}

    async def list_companion_knowledge(self, **kwargs):
        self.calls.append(("list_companion_knowledge", kwargs))
        return {"items": [], "total": 0}

    async def get_companion_knowledge(self, card_id):
        self.calls.append(("get_companion_knowledge", card_id))
        return {"id": card_id, "title": "Knowledge"}

    async def get_companion_reflection(self, reflection_id):
        self.calls.append(("get_companion_reflection", reflection_id))
        return {"id": reflection_id, "title": "Reflection"}

    async def get_companion_conversation_prompts(self, *, query):
        self.calls.append(("get_companion_conversation_prompts", {"query": query}))
        return {"prompts": [], "prompt_source_kind": "context"}

    async def list_companion_goals(self, **kwargs):
        self.calls.append(("list_companion_goals", kwargs))
        return {"items": [], "total": 0}

    async def create_companion_goal(self, request_data):
        self.calls.append(("create_companion_goal", request_data))
        return {"id": "goal-1", "title": "Read", "goal_type": "habit"}

    async def update_companion_goal(self, goal_id, request_data):
        self.calls.append(("update_companion_goal", goal_id, request_data))
        return {"id": goal_id, "title": "Read more", "goal_type": "habit"}

    async def purge_companion_data(self, request_data):
        self.calls.append(("purge_companion_data", request_data))
        return {"status": "purged", "scope": "knowledge"}

    async def rebuild_companion_data(self, request_data):
        self.calls.append(("rebuild_companion_data", request_data))
        return {"status": "queued", "scope": "reflections", "job_id": 42}


@pytest.mark.asyncio
async def test_server_companion_service_delegates_and_normalizes_records():
    client = FakeCompanionClient()
    service = ServerCompanionService(client=client)

    created_activity = await service.create_activity(ACTIVITY_REQUEST)
    created_check_in = await service.create_check_in({"summary": "Read"})
    activity = await service.list_activity(limit=25, offset=5)
    activity_detail = await service.get_activity("act-1")
    knowledge = await service.list_knowledge(status="active")
    knowledge_detail = await service.get_knowledge("card-1")
    reflection = await service.get_reflection("reflection-1")
    prompts = await service.get_conversation_prompts(query="progress")
    goals = await service.list_goals(status="active")
    created_goal = await service.create_goal({"title": "Read", "goal_type": "habit"})
    updated_goal = await service.update_goal("goal-1", {"title": "Read more"})
    purged = await service.purge_data({"scope": "knowledge"})
    rebuilt = await service.rebuild_data({"scope": "reflections"})

    assert created_activity["record_id"] == "server:companion_activity:act-1"
    assert created_check_in["record_id"] == "server:companion_activity:act-check-in"
    assert activity["record_id"] == "server:companion_activity"
    assert activity_detail["record_id"] == "server:companion_activity:act-1"
    assert knowledge["record_id"] == "server:companion_knowledge"
    assert knowledge_detail["record_id"] == "server:companion_knowledge:card-1"
    assert reflection["record_id"] == "server:companion_reflection:reflection-1"
    assert prompts["record_id"] == "server:companion_conversation_prompts"
    assert goals["record_id"] == "server:companion_goals"
    assert created_goal["record_id"] == "server:companion_goal:goal-1"
    assert updated_goal["record_id"] == "server:companion_goal:goal-1"
    assert purged["record_id"] == "server:companion_lifecycle:knowledge"
    assert rebuilt["record_id"] == "server:companion_lifecycle:reflections"
    assert all(item["backend"] == "server" for item in [
        created_activity,
        created_check_in,
        activity,
        activity_detail,
        knowledge,
        knowledge_detail,
        reflection,
        prompts,
        goals,
        created_goal,
        updated_goal,
        purged,
        rebuilt,
    ])


@pytest.mark.asyncio
async def test_server_companion_service_enforces_policy_actions():
    client = FakeCompanionClient()
    policy = Mock()
    service = ServerCompanionService(client=client, policy_enforcer=policy)

    await service.create_activity(ACTIVITY_REQUEST)
    await service.create_check_in({"summary": "Read"})
    await service.list_activity()
    await service.get_activity("act-1")
    await service.list_knowledge()
    await service.get_knowledge("card-1")
    await service.get_reflection("reflection-1")
    await service.get_conversation_prompts(query="progress")
    await service.list_goals()
    await service.create_goal({"title": "Read", "goal_type": "habit"})
    await service.update_goal("goal-1", {"title": "Read more"})
    await service.purge_data({"scope": "knowledge"})
    await service.rebuild_data({"scope": "reflections"})

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
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
async def test_server_companion_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeCompanionClient()
    service = ServerCompanionService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_activity()

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []


def test_server_companion_service_from_config_uses_api_client(monkeypatch):
    sentinel_client = Mock()
    build_client = Mock(return_value=sentinel_client)

    monkeypatch.setattr(
        "tldw_chatbook.Companion_Interop.server_companion_service.build_runtime_api_client_from_config",
        build_client,
    )

    service = ServerCompanionService.from_config({"tldw_api": {"base_url": "https://example.com"}})

    assert service.client is sentinel_client
    build_client.assert_called_once_with({"tldw_api": {"base_url": "https://example.com"}})
