"""Tests for local/server Research Sessions scope routing."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.DB.Research_DB import ResearchDatabase
from tldw_chatbook.Research_Interop import (
    LocalResearchService,
    ResearchScopeService,
    ServerResearchService,
)
from tldw_chatbook.tldw_api import ResearchRunResponse


@pytest.mark.asyncio
async def test_local_research_service_creates_and_lists_standalone_runs(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db", client_id="tester")
    service = LocalResearchService(db)

    run = await service.create_run(
        query="Map the local vector search options",
        source_policy="local_only",
        autonomy_mode="manual",
        limits_json={"max_sources": 5},
    )

    assert run.source == "local"
    assert run.query == "Map the local vector search options"
    assert run.status == "draft"
    assert run.phase == "planning"

    runs = await service.list_runs(limit=10)
    assert [item.id for item in runs] == [run.id]
    assert runs[0].limits_json == {"max_sources": 5}


@pytest.mark.asyncio
async def test_local_research_service_controls_and_artifacts_are_persisted(tmp_path):
    db = ResearchDatabase(tmp_path / "research.db", client_id="tester")
    service = LocalResearchService(db)
    run = await service.create_run(query="Explain agent governance", source_policy="local_only")

    resumed = await service.resume_run(run.id)
    paused = await service.pause_run(run.id)
    cancelled = await service.cancel_run(run.id)
    artifact = await service.save_artifact(
        run.id,
        artifact_name="notes.md",
        content_type="text/markdown",
        content="# Notes",
    )

    assert resumed.control_state == "running"
    assert paused.control_state == "paused"
    assert cancelled.status == "cancelled"
    assert artifact.artifact_version == 1
    assert await service.get_bundle(run.id) == {"notes.md": "# Notes"}


@pytest.mark.asyncio
async def test_server_research_service_delegates_to_tldw_client():
    client = AsyncMock()
    client.create_research_run.return_value = ResearchRunResponse(
        id="rs-server",
        status="running",
        phase="collecting",
    )
    client.list_research_runs.return_value = [
        {
            "id": "rs-server",
            "query": "Server query",
            "status": "running",
            "phase": "collecting",
            "control_state": "running",
            "created_at": "2026-04-22T00:00:00Z",
            "updated_at": "2026-04-22T00:01:00Z",
        }
    ]
    service = ServerResearchService(client=client)

    created = await service.create_run(query="Server query", source_policy="balanced")
    listed = await service.list_runs(limit=5)

    assert created.source == "server"
    assert created.id == "rs-server"
    assert listed[0].source == "server"
    client.create_research_run.assert_awaited_once()
    client.list_research_runs.assert_awaited_once_with(limit=5)


@pytest.mark.asyncio
async def test_research_scope_service_routes_without_cross_source_mutation(tmp_path):
    local_service = LocalResearchService(ResearchDatabase(tmp_path / "research.db", client_id="tester"))
    server_service = AsyncMock()
    server_service.create_run.return_value = "server-created"
    scope = ResearchScopeService(local_service=local_service, server_service=server_service)

    local_run = await scope.create_run(mode="local", query="Local query")
    server_run = await scope.create_run(mode="server", query="Server query")

    assert local_run.source == "local"
    assert server_run == "server-created"
    server_service.create_run.assert_awaited_once_with(query="Server query")
    assert [run.query for run in await local_service.list_runs()] == ["Local query"]


@pytest.mark.asyncio
async def test_research_scope_service_enforces_action_level_policy(tmp_path):
    class RecordingPolicyEnforcer:
        def __init__(self):
            self.action_ids = []

        def require_allowed(self, *, action_id):
            self.action_ids.append(action_id)

    local_service = LocalResearchService(ResearchDatabase(tmp_path / "research.db", client_id="tester"))
    server_service = AsyncMock()
    server_service.resume_run.return_value = "resumed"
    server_service.get_bundle.return_value = {"report.md": "# Report"}
    server_service.patch_and_approve_checkpoint.return_value = "approved"
    policy = RecordingPolicyEnforcer()
    scope = ResearchScopeService(
        local_service=local_service,
        server_service=server_service,
        policy_enforcer=policy,
    )

    await scope.create_run(mode="local", query="Local query")
    await scope.resume_run("rs-server", mode="server")
    await scope.get_bundle("rs-server", mode="server")
    await scope.patch_and_approve_checkpoint(
        "rs-server",
        "checkpoint-1",
        mode="server",
        patch_payload={"approved": True},
    )

    assert policy.action_ids == [
        "research.runs.create.local",
        "research.runs.launch.server",
        "research.runs.observe.server",
        "research.runs.update.server",
    ]
