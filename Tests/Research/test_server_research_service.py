from unittest.mock import Mock

import pytest

from tldw_chatbook.Research_Interop import ServerResearchService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeResearchClient:
    def __init__(self):
        self.calls = []

    async def create_research_run(self, request_data):
        self.calls.append(("create_research_run", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": "run-1", "status": "running", "phase": "planning", "control_state": "running"}

    async def list_research_runs(self, limit=25):
        self.calls.append(("list_research_runs", limit))
        return [{"id": "run-1", "query": "MCP", "status": "running"}]

    async def get_research_run(self, session_id):
        self.calls.append(("get_research_run", session_id))
        return {"id": session_id, "status": "running"}

    async def pause_research_run(self, session_id):
        self.calls.append(("pause_research_run", session_id))
        return {"id": session_id, "control_state": "paused"}

    async def cancel_research_run(self, session_id):
        self.calls.append(("cancel_research_run", session_id))
        return {"id": session_id, "status": "cancelled"}

    async def get_research_bundle(self, session_id):
        self.calls.append(("get_research_bundle", session_id))
        return {"summary": {"answer": "Done"}}

    async def get_research_artifact(self, session_id, artifact_name):
        self.calls.append(("get_research_artifact", session_id, artifact_name))
        return {"artifact_name": artifact_name, "content_type": "application/json", "content": {"ok": True}}


@pytest.mark.asyncio
async def test_server_research_service_routes_runs_with_policy_actions():
    client = FakeResearchClient()
    policy = Mock()
    service = ServerResearchService(client=client, policy_enforcer=policy)

    launched = await service.launch_run(query="MCP governance")
    listed = await service.list_runs(limit=10)
    detail = await service.get_run("run-1")
    paused = await service.pause_run("run-1")
    cancelled = await service.cancel_run("run-1")
    bundle = await service.get_bundle("run-1")
    artifact = await service.get_artifact("run-1", "final_report")

    assert launched["id"] == "run-1"
    assert listed[0]["id"] == "run-1"
    assert detail["id"] == "run-1"
    assert paused["control_state"] == "paused"
    assert cancelled["status"] == "cancelled"
    assert bundle["summary"]["answer"] == "Done"
    assert artifact["content"] == {"ok": True}
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "research.runs.launch.server",
        "research.runs.list.server",
        "research.runs.detail.server",
        "research.runs.update.server",
        "research.runs.update.server",
        "research.runs.detail.server",
        "research.runs.detail.server",
    ]


@pytest.mark.asyncio
async def test_server_research_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeResearchClient()
    service = ServerResearchService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_runs()

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
