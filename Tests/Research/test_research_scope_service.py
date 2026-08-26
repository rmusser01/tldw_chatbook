import asyncio
import threading

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
        return {
            "id": f"{self.source}-session-1",
            "title": kwargs["title"],
            "query": kwargs["query"],
            "version": 1,
        }

    async def list_sessions(self, *, limit=100, offset=0, status=None):
        self.calls.append(("list_sessions", limit, offset, status))
        return [
            {
                "id": f"{self.source}-session-1",
                "title": "Research",
                "query": "MCP",
                "version": 1,
            }
        ]

    async def get_session(self, session_id):
        self.calls.append(("get_session", session_id))
        return {"id": session_id, "title": "Research", "query": "MCP", "version": 1}

    async def update_session(self, session_id, *, expected_version=None, **kwargs):
        self.calls.append(("update_session", session_id, expected_version, kwargs))
        return {
            "id": session_id,
            "title": kwargs["title"],
            "query": "MCP",
            "version": 2,
        }

    async def delete_session(self, session_id, *, expected_version=None):
        self.calls.append(("delete_session", session_id, expected_version))
        return True

    async def launch_run(self, **kwargs):
        self.calls.append(("launch_run", kwargs))
        return {
            "id": f"{self.source}-run-1",
            "query": kwargs["query"],
            "status": "running",
            "version": 1,
        }

    async def get_run(self, run_id):
        self.calls.append(("get_run", run_id))
        return {"id": run_id, "query": "MCP", "status": "running", "version": 1}

    async def list_runs(self, *, limit=100, offset=0, session_id=None, status=None):
        self.calls.append(("list_runs", limit, offset, session_id, status))
        return [
            {
                "id": f"{self.source}-run-1",
                "query": "MCP",
                "status": "running",
                "version": 1,
            }
        ]

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
            "artifacts": [
                {
                    "run_id": run_id,
                    "artifact_name": "summary",
                    "content_type": "text/markdown",
                }
            ],
        }

    async def get_artifact(self, run_id, artifact_name):
        self.calls.append(("get_artifact", run_id, artifact_name))
        return {
            "run_id": run_id,
            "artifact_name": artifact_name,
            "content_type": "text/markdown",
        }


class LimitOnlyServerResearchService:
    def __init__(self):
        self.calls = []

    async def list_runs(self, *, limit=100):
        self.calls.append(("list_runs", limit))
        return [
            {"id": "server-run-1", "query": "MCP", "status": "running", "version": 1}
        ]


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


class FakeSyncScopeService:
    def __init__(self):
        self.calls = []

    def record_dry_run_mirror_report(self, **kwargs):
        self.calls.append(kwargs)
        return {"backend": "server", "domain": kwargs["domain"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "kwargs", "action_id"),
    [
        ("list_sessions", {}, "research.sessions.list.server"),
        (
            "create_session",
            {"title": "Research", "query": "MCP"},
            "research.sessions.create.server",
        ),
        (
            "get_session",
            {"session_id": "server-session-1"},
            "research.sessions.detail.server",
        ),
        (
            "update_session",
            {
                "session_id": "server-session-1",
                "title": "Updated",
                "expected_version": 1,
            },
            "research.sessions.update.server",
        ),
        (
            "delete_session",
            {"session_id": "server-session-1", "expected_version": 1},
            "research.sessions.delete.server",
        ),
    ],
)
async def test_research_scope_service_blocks_server_session_crud_as_unsupported(
    method_name, kwargs, action_id
):
    server = object()
    policy = FakePolicyEnforcer()
    scope = ResearchScopeService(
        local_service=FakeResearchService("local"),
        server_service=server,
        policy_enforcer=policy,
    )

    with pytest.raises(
        NotImplementedError, match="does not expose separate research session CRUD"
    ):
        await getattr(scope, method_name)(mode="server", **kwargs)

    assert policy.calls == [action_id]


@pytest.mark.asyncio
async def test_research_scope_service_rejects_unsupported_server_run_list_filters_before_dispatch():
    server = LimitOnlyServerResearchService()
    policy = FakePolicyEnforcer()
    scope = ResearchScopeService(
        local_service=FakeResearchService("local"),
        server_service=server,
        policy_enforcer=policy,
    )

    with pytest.raises(
        NotImplementedError, match="does not support filtered research run lists"
    ):
        await scope.list_runs(mode="server", status="completed")

    assert server.calls == []
    assert policy.calls == ["research.runs.list.server"]


@pytest.mark.asyncio
async def test_research_scope_service_routes_server_session_crud_when_adapter_provides_it():
    server = FakeResearchService("server")
    policy = FakePolicyEnforcer()
    scope = ResearchScopeService(
        local_service=FakeResearchService("local"),
        server_service=server,
        policy_enforcer=policy,
    )

    created = await scope.create_session(mode="server", title="Research", query="MCP")
    listed = await scope.list_sessions(mode="server", status="active")
    detail = await scope.get_session(mode="server", session_id="server-session-1")
    updated = await scope.update_session(
        mode="server",
        session_id="server-session-1",
        title="Updated",
        expected_version=1,
    )
    deleted = await scope.delete_session(
        mode="server",
        session_id="server-session-1",
        expected_version=2,
    )

    assert created["record_id"] == "server:research_session:server-session-1"
    assert listed[0]["record_id"] == "server:research_session:server-session-1"
    assert detail["record_id"] == "server:research_session:server-session-1"
    assert updated["record_id"] == "server:research_session:server-session-1"
    assert deleted is True
    assert policy.calls == [
        "research.sessions.create.server",
        "research.sessions.list.server",
        "research.sessions.detail.server",
        "research.sessions.update.server",
        "research.sessions.delete.server",
    ]


@pytest.mark.asyncio
async def test_research_scope_service_routes_server_filtered_runs_and_delete_when_adapter_supports_it():
    server = FakeResearchService("server")
    policy = FakePolicyEnforcer()
    scope = ResearchScopeService(
        local_service=FakeResearchService("local"),
        server_service=server,
        policy_enforcer=policy,
    )

    runs = await scope.list_runs(
        mode="server",
        session_id="server-session-1",
        status="running",
        offset=5,
    )
    deleted = await scope.delete_run(
        mode="server", run_id="server-run-1", expected_version=1
    )

    assert runs[0]["record_id"] == "server:research_run:server-run-1"
    assert deleted is True
    assert ("list_runs", 100, 5, "server-session-1", "running") in server.calls
    assert ("delete_run", "server-run-1", 1) in server.calls
    assert policy.calls == [
        "research.runs.list.server",
        "research.runs.delete.server",
    ]


@pytest.mark.asyncio
async def test_research_scope_service_routes_sessions_runs_and_policy_actions():
    local = FakeResearchService("local")
    server = FakeResearchService("server")
    policy = FakePolicyEnforcer()
    scope = ResearchScopeService(
        local_service=local, server_service=server, policy_enforcer=policy
    )

    local_session = await scope.create_session(
        mode="local", title="Research", query="MCP"
    )
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
    scope = ResearchScopeService(
        local_service=local,
        server_service=FakeResearchService("server"),
        policy_enforcer=policy,
    )

    session = await scope.update_session(
        mode="local", session_id="local-session-1", title="Updated", expected_version=1
    )
    deleted_session = await scope.delete_session(
        mode="local", session_id="local-session-1", expected_version=2
    )
    runs = await scope.list_runs(mode="local", session_id="local-session-1")
    deleted_run = await scope.delete_run(
        mode="local", run_id="local-run-1", expected_version=1
    )

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
    scope = ResearchScopeService(
        local_service=local, server_service=FakeResearchService("server")
    )

    session = await scope.create_session(
        mode="local", title="Research", query="Inherited query"
    )
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
    artifact = await scope.get_artifact(
        mode="server", run_id="server-run-1", artifact_name="summary"
    )

    assert events[0]["record_id"] == "server:research_run_event:server-run-1:1"
    assert bundle["backend"] == "server"
    assert bundle["run"]["record_id"] == "server:research_run:server-run-1"
    assert (
        bundle["artifacts"][0]["record_id"]
        == "server:research_artifact:server-run-1:summary"
    )
    assert artifact["record_id"] == "server:research_artifact:server-run-1:summary"
    assert policy.calls == [
        "research.runs.observe.server",
        "research.runs.detail.server",
        "research.runs.detail.server",
    ]


def test_research_scope_service_reports_known_unsupported_server_capabilities():
    scope = ResearchScopeService(
        local_service=FakeResearchService("local"),
        server_service=object(),
    )

    assert scope.list_unsupported_capabilities(mode="local") == []
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert server_report == [
        {
            "operation_id": "research.sessions.server_crud",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_run_centric",
            "user_message": (
                "The current server API exposes deep research through run-centric /research/runs "
                "operations and does not support separate research session CRUD."
            ),
            "affected_action_ids": [
                "research.sessions.create.server",
                "research.sessions.list.server",
                "research.sessions.detail.server",
                "research.sessions.update.server",
                "research.sessions.delete.server",
            ],
        },
        {
            "operation_id": "research.runs.filtered_list.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": (
                "The current server API only supports limit-based research run listing; "
                "offset, session, and status filters are local-only."
            ),
            "affected_action_ids": ["research.runs.list.server"],
        },
        {
            "operation_id": "research.runs.delete.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "The current server API does not support research run deletion.",
            "affected_action_ids": ["research.runs.delete.server"],
        },
    ]

    capable_scope = ResearchScopeService(
        local_service=FakeResearchService("local"),
        server_service=FakeResearchService("server"),
    )

    assert capable_scope.list_unsupported_capabilities(mode="server") == []


def test_research_scope_service_routes_run_sync_mirror_report_to_sync_scope():
    sync_scope = FakeSyncScopeService()
    scope = ResearchScopeService(
        local_service=FakeResearchService("local"),
        server_service=FakeResearchService("server"),
        sync_scope_service=sync_scope,
    )

    result = scope.record_sync_mirror_report(
        mode="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        local_records=[{"id": "local-run-1"}],
        remote_records=[{"id": "remote-run-1"}],
    )

    assert result == {"backend": "server", "domain": "research"}
    assert sync_scope.calls == [
        {
            "mode": "server",
            "domain": "research",
            "entity_type": "research_run",
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "workspace_scope": "workspace-1",
            "local_records": [{"id": "local-run-1"}],
            "remote_records": [{"id": "remote-run-1"}],
        }
    ]


def test_research_scope_service_rejects_local_sync_mirror_report():
    scope = ResearchScopeService(
        local_service=FakeResearchService("local"),
        server_service=FakeResearchService("server"),
        sync_scope_service=FakeSyncScopeService(),
    )

    with pytest.raises(ValueError, match="Research mirror reports require server mode"):
        scope.record_sync_mirror_report(
            mode="local",
            server_profile_id="server-a",
        )


# ---------------------------------------------------------------------------
# TASK-21127: the local backend runs off the Textual event loop.
# ---------------------------------------------------------------------------


def _record_db_threads(monkeypatch):
    """Record the thread every database operation actually runs on."""
    seen: list[str] = []
    real_connect = LocalResearchService._connect

    def _recording(self):
        seen.append(threading.current_thread().name)
        return real_connect(self)

    monkeypatch.setattr(LocalResearchService, "_connect", _recording)
    return seen


@pytest.mark.asyncio
async def test_local_backend_database_work_runs_off_the_event_loop(
    tmp_path, monkeypatch
):
    """The whole point of the offload: no SQLite on the loop thread.

    Driven through the REAL object graph (scope service -> real
    LocalResearchService on a temp file), because a synchronous test double
    would satisfy a thread assertion without proving anything about the
    shipped app (TASK-21125 lesson).
    """
    service = LocalResearchService(tmp_path / "research.db")
    scope = ResearchScopeService(local_service=service, server_service=None)
    run = await scope.create_run(mode="local", query="off the loop")

    loop_thread = threading.current_thread().name
    seen = _record_db_threads(monkeypatch)
    await scope.get_run(run["id"], mode="local")
    await scope.list_runs(mode="local")
    await scope.get_bundle(run["id"], mode="local")

    assert seen, "no database operation was observed"
    assert loop_thread not in seen, f"SQLite ran on the event loop thread: {seen}"
    assert all(name.startswith("research-backend") for name in seen), seen
    service.close()


@pytest.mark.asyncio
async def test_concurrent_scope_calls_share_one_backend_thread(tmp_path, monkeypatch):
    """A SINGLE-thread executor, not the default pool: it restores exactly the
    ordering the event loop was providing for free before the offload."""
    service = LocalResearchService(tmp_path / "research.db")
    scope = ResearchScopeService(local_service=service, server_service=None)
    run = await scope.create_run(mode="local", query="serialised")

    seen = _record_db_threads(monkeypatch)
    await asyncio.gather(*(scope.get_run(run["id"], mode="local") for _ in range(12)))

    assert len(set(seen)) == 1, f"backend work fanned out across threads: {set(seen)}"
    service.close()


@pytest.mark.asyncio
async def test_stream_run_events_async_generator_is_not_offloaded(tmp_path):
    """``LocalResearchService.stream_run_events`` is an ``async def ... yield``.

    ``inspect.iscoroutinefunction`` is False for such a function, so a
    passthrough predicate that only checks for coroutines would route it
    through a thread and hand the caller a coroutine wrapping the generator
    object instead of the generator.
    """
    service = LocalResearchService(tmp_path / "research.db")
    scope = ResearchScopeService(local_service=service, server_service=None)
    run = await scope.create_run(mode="local", query="streaming")

    events = [event async for event in scope.stream_run_events(run["id"], mode="local")]

    assert events and any(event.get("event") == "created" for event in events)
    observed = await scope.observe_run_events(mode="local", run_id=run["id"])
    assert observed and observed[0]["run_id"] == run["id"]
    service.close()


def test_offload_preserves_the_wired_local_service_identity(tmp_path):
    """Only the DISPATCH path is wrapped: app wiring still sees what it passed."""
    service = LocalResearchService(tmp_path / "research.db")
    scope = ResearchScopeService(local_service=service, server_service=None)

    assert scope.local_service is service
    service.close()


@pytest.mark.asyncio
async def test_backend_exceptions_survive_the_offload(tmp_path):
    """A proxy must not swallow or re-wrap the backend's error type."""
    service = LocalResearchService(tmp_path / "research.db")
    scope = ResearchScopeService(local_service=service, server_service=None)

    with pytest.raises(ValueError, match="research run not found"):
        await scope.get_bundle("no-such-run", mode="local")
    service.close()


def _record_event_read_threads(monkeypatch):
    """Record the thread each `list_run_events` read actually executes on.

    `_record_db_threads` above hooks `_connect`, which a HELD connection only
    reaches once per thread -- too coarse to tell where a later read ran.
    """
    seen: list[str] = []
    real_list = LocalResearchService.list_run_events

    def _recording(self, *args, **kwargs):
        seen.append(threading.current_thread().name)
        return real_list(self, *args, **kwargs)

    monkeypatch.setattr(LocalResearchService, "list_run_events", _recording)
    return seen


@pytest.mark.asyncio
async def test_local_stream_run_events_reads_off_the_event_loop(tmp_path, monkeypatch):
    """The async-generator path must honour the same no-SQLite-on-the-loop rule.

    `_ThreadOffloadedBackend` passes async generators through unwrapped, which
    is right for the server backend but left `LocalResearchService.
    stream_run_events` -- an `async def ... yield` whose body is a blocking
    `list_run_events` call -- executing SQLite on the loop thread every time
    the Research window's 2 s poll consumed it.
    """
    service = LocalResearchService(tmp_path / "research.db")
    scope = ResearchScopeService(local_service=service, server_service=None)
    run = await scope.create_run(mode="local", query="streamed off the loop")
    for index in range(25):
        service.record_run_event(run["id"], f"event-{index}", {"index": index})

    loop_thread = threading.current_thread().name
    seen = _record_event_read_threads(monkeypatch)
    events = [event async for event in scope.stream_run_events(run["id"], mode="local")]

    assert events, "the stream yielded nothing, so the thread assertion is vacuous"
    assert seen, "no event read was observed at all"
    assert loop_thread not in seen, f"SQLite ran on the event loop thread: {seen}"
    assert all(name.startswith("research-backend") for name in seen), seen
    service.close()


@pytest.mark.asyncio
async def test_local_stream_run_events_output_is_unchanged_by_the_offload(tmp_path):
    """Taking the offloaded reader must yield exactly what the generator did.

    `LocalResearchService.stream_run_events` is a snapshot loop over
    `list_run_events`, so the two are required to agree item for item --
    including under `after_id`, which is the argument the poll advances.
    """
    service = LocalResearchService(tmp_path / "research.db")
    scope = ResearchScopeService(local_service=service, server_service=None)
    run = await scope.create_run(mode="local", query="identical output")
    for index in range(25):
        service.record_run_event(run["id"], f"event-{index}", {"index": index})

    direct = [event async for event in service.stream_run_events(run["id"])]
    routed = [event async for event in scope.stream_run_events(run["id"], mode="local")]
    assert routed == direct
    assert len(routed) > 1, "too few events to distinguish a truncation bug"

    tail_direct = [event async for event in service.stream_run_events(run["id"], after_id=10)]
    tail_routed = [
        event
        async for event in scope.stream_run_events(run["id"], mode="local", after_id=10)
    ]
    assert tail_routed == tail_direct
    assert len(tail_routed) < len(routed), "after_id was ignored"
    service.close()


@pytest.mark.asyncio
async def test_async_backend_stream_is_still_consumed_as_a_generator():
    """The server backend keeps its real streaming path -- no reader swap.

    The offloaded-reader preference is gated on a SYNCHRONOUS `list_run_events`.
    An async backend has none to prefer, so its `stream_run_events` must still
    be the method that runs, and must still be iterated lazily.
    """

    class AsyncStreamingBackend:
        def __init__(self):
            self.streamed = 0

        async def stream_run_events(self, run_id, *, after_id=0):
            for index in range(3):
                self.streamed += 1
                yield {"id": index + 1, "run_id": run_id, "event": f"e{index}"}

    backend = AsyncStreamingBackend()
    scope = ResearchScopeService(local_service=None, server_service=backend)

    events = [event async for event in scope.stream_run_events("r1", mode="server")]

    assert backend.streamed == 3, "the async generator was not the method used"
    assert [event["event"] for event in events] == ["e0", "e1", "e2"]
