import pytest

from tldw_chatbook.Research_Interop import LocalResearchService
from tldw_chatbook.Research_Interop.research_scope_service import ResearchScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeResearchService:
    def __init__(self, source):
        self.source = source
        self.calls = []

    async def create_session(self, **kwargs):
        self.calls.append(("create_session", kwargs))
        return {"id": f"{self.source}-session-1", "title": kwargs["title"], "query": kwargs["query"], "version": 1}

    async def list_sessions(self, *, limit=100, offset=0, status=None):
        self.calls.append(("list_sessions", limit, offset, status))
        return [{"id": f"{self.source}-session-1", "title": "Research", "query": "MCP", "version": 1}]

    async def update_session(self, session_id, *, expected_version=None, **kwargs):
        self.calls.append(("update_session", session_id, expected_version, kwargs))
        return {"id": session_id, "title": kwargs["title"], "query": "MCP", "version": 2}

    async def delete_session(self, session_id, *, expected_version=None):
        self.calls.append(("delete_session", session_id, expected_version))
        return True

    async def launch_run(self, **kwargs):
        self.calls.append(("launch_run", kwargs))
        return {"id": f"{self.source}-run-1", "query": kwargs["query"], "status": "running", "version": 1}

    async def get_run(self, run_id):
        self.calls.append(("get_run", run_id))
        return {"id": run_id, "query": "MCP", "status": "running", "version": 1}

    async def list_runs(self, *, limit=100, offset=0, session_id=None, status=None):
        self.calls.append(("list_runs", limit, offset, session_id, status))
        return [{"id": f"{self.source}-run-1", "query": "MCP", "status": "running", "version": 1}]

    async def pause_run(self, run_id):
        self.calls.append(("pause_run", run_id))
        return {"id": run_id, "control_state": "paused", "version": 2}

    async def delete_run(self, run_id, *, expected_version=None):
        self.calls.append(("delete_run", run_id, expected_version))
        return True

    async def list_run_events(self, run_id, *, after_id=0):
        self.calls.append(("list_run_events", run_id, after_id))
        return [{"id": 1, "run_id": run_id, "event_type": "created"}]

    async def get_bundle(self, run_id):
        self.calls.append(("get_bundle", run_id))
        return {
            "run": {"id": run_id, "query": "MCP", "status": "completed"},
            "artifacts": [{"run_id": run_id, "artifact_name": "summary", "content_type": "text/markdown"}],
        }

    async def get_artifact(self, run_id, artifact_name):
        self.calls.append(("get_artifact", run_id, artifact_name))
        return {"run_id": run_id, "artifact_name": artifact_name, "content_type": "text/markdown"}


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
async def test_research_scope_service_routes_sessions_runs_and_policy_actions():
    local = FakeResearchService("local")
    server = FakeResearchService("server")
    policy = FakePolicyEnforcer()
    scope = ResearchScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    local_session = await scope.create_session(mode="local", title="Research", query="MCP")
    local_sessions = await scope.list_sessions(mode="local")
    server_run = await scope.launch_run(mode="server", query="MCP")
    server_run_detail = await scope.get_run(mode="server", run_id=server_run["id"])
    paused = await scope.pause_run(mode="server", run_id=server_run["id"])

    assert local_session["record_id"] == "local:research_session:local-session-1"
    assert local_sessions[0]["record_id"] == "local:research_session:local-session-1"
    assert server_run["record_id"] == "server:research_run:server-run-1"
    assert server_run_detail["record_id"] == "server:research_run:server-run-1"
    assert paused["control_state"] == "paused"
    assert policy.calls == [
        "research.sessions.create.local",
        "research.sessions.list.local",
        "research.runs.launch.server",
        "research.runs.detail.server",
        "research.runs.update.server",
    ]


@pytest.mark.asyncio
async def test_research_scope_service_denies_blocked_actions_before_dispatch():
    server = FakeResearchService("server")
    scope = ResearchScopeService(
        local_service=FakeResearchService("local"),
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("wrong_source"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.launch_run(mode="server", query="MCP")

    assert exc.value.reason_code == "wrong_source"
    assert server.calls == []


@pytest.mark.asyncio
async def test_research_scope_service_routes_update_and_delete_actions():
    local = FakeResearchService("local")
    policy = FakePolicyEnforcer()
    scope = ResearchScopeService(local_service=local, server_service=FakeResearchService("server"), policy_enforcer=policy)

    session = await scope.update_session(mode="local", session_id="local-session-1", title="Updated", expected_version=1)
    deleted_session = await scope.delete_session(mode="local", session_id="local-session-1", expected_version=2)
    runs = await scope.list_runs(mode="local", session_id="local-session-1")
    deleted_run = await scope.delete_run(mode="local", run_id="local-run-1", expected_version=1)

    assert session["version"] == 2
    assert deleted_session is True
    assert runs[0]["record_id"] == "local:research_run:local-run-1"
    assert deleted_run is True
    assert policy.calls == [
        "research.sessions.update.local",
        "research.sessions.delete.local",
        "research.runs.list.local",
        "research.runs.delete.local",
    ]


@pytest.mark.asyncio
async def test_research_scope_service_can_launch_local_run_from_session_query(tmp_path):
    local = LocalResearchService(tmp_path / "research.db")
    scope = ResearchScopeService(local_service=local, server_service=FakeResearchService("server"))

    session = await scope.create_session(mode="local", title="Research", query="Inherited query")
    run = await scope.launch_run(mode="local", session_id=session["id"])

    assert run["query"] == "Inherited query"
    assert run["record_id"].startswith("local:research_run:")


@pytest.mark.asyncio
async def test_research_scope_service_normalizes_bundle_artifact_and_event_records():
    server = FakeResearchService("server")
    policy = FakePolicyEnforcer()
    scope = ResearchScopeService(
        local_service=FakeResearchService("local"),
        server_service=server,
        policy_enforcer=policy,
    )

    events = await scope.observe_run_events(mode="server", run_id="server-run-1")
    bundle = await scope.get_bundle(mode="server", run_id="server-run-1")
    artifact = await scope.get_artifact(mode="server", run_id="server-run-1", artifact_name="summary")

    assert events[0]["record_id"] == "server:research_run_event:server-run-1:1"
    assert bundle["backend"] == "server"
    assert bundle["run"]["record_id"] == "server:research_run:server-run-1"
    assert bundle["artifacts"][0]["record_id"] == "server:research_artifact:server-run-1:summary"
    assert artifact["record_id"] == "server:research_artifact:server-run-1:summary"
    assert policy.calls == [
        "research.runs.observe.server",
        "research.runs.detail.server",
        "research.runs.detail.server",
    ]
