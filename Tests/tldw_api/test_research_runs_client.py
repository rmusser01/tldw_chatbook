from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ResearchCheckpointPatchApproveRequest,
    ResearchRunCreateRequest,
    TLDWAPIClient,
)


def _run_payload(run_id: str = "run-1", *, status: str = "running") -> dict:
    return {
        "id": run_id,
        "status": status,
        "phase": "planning",
        "control_state": "running",
        "progress_percent": 10.0,
        "progress_message": "Planning",
        "active_job_id": "job-1",
        "latest_checkpoint_id": None,
        "completed_at": None,
        "chat_id": None,
    }


def _run_list_payload(run_id: str = "run-1") -> dict:
    return {
        **_run_payload(run_id),
        "query": "Summarize MCP governance approaches",
        "created_at": "2026-04-24T12:00:00Z",
        "updated_at": "2026-04-24T12:05:00Z",
    }


@pytest.mark.asyncio
async def test_research_runs_client_routes_lifecycle_and_artifact_calls(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            _run_payload(),
            [_run_list_payload()],
            _run_payload(),
            {**_run_payload(), "control_state": "paused"},
            _run_payload(),
            {**_run_payload(), "status": "cancelled"},
            {"summary": {"answer": "Use explicit policy gates."}},
            {"artifact_name": "final_report", "content_type": "application/json", "content": {"ok": True}},
            _run_payload(status="running"),
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_research_run(
        ResearchRunCreateRequest(
            query="Summarize MCP governance approaches",
            source_policy="balanced",
            autonomy_mode="checkpointed",
        )
    )
    listed = await client.list_research_runs(limit=10)
    fetched = await client.get_research_run("run-1")
    paused = await client.pause_research_run("run-1")
    resumed = await client.resume_research_run("run-1")
    cancelled = await client.cancel_research_run("run-1")
    bundle = await client.get_research_bundle("run-1")
    artifact = await client.get_research_artifact("run-1", "final_report")
    approved = await client.patch_and_approve_research_checkpoint(
        "run-1",
        "checkpoint-1",
        ResearchCheckpointPatchApproveRequest(patch_payload={"accepted": True}),
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/research/runs")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "query": "Summarize MCP governance approaches",
        "source_policy": "balanced",
        "autonomy_mode": "checkpointed",
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/research/runs")
    assert mocked.await_args_list[1].kwargs["params"] == {"limit": 10, "offset": 0}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/research/runs/run-1")
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/research/runs/run-1/pause")
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/research/runs/run-1/resume")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/research/runs/run-1/cancel")
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/research/runs/run-1/bundle")
    assert mocked.await_args_list[7].args[:2] == (
        "GET",
        "/api/v1/research/runs/run-1/artifacts/final_report",
    )
    assert mocked.await_args_list[8].args[:2] == (
        "POST",
        "/api/v1/research/runs/run-1/checkpoints/checkpoint-1/patch-and-approve",
    )
    assert mocked.await_args_list[8].kwargs["json_data"] == {"patch_payload": {"accepted": True}}

    assert created.id == "run-1"
    assert listed[0].query == "Summarize MCP governance approaches"
    assert fetched.id == "run-1"
    assert paused.control_state == "paused"
    assert resumed.status == "running"
    assert cancelled.status == "cancelled"
    assert bundle["summary"]["answer"] == "Use explicit policy gates."
    assert artifact.content == {"ok": True}
    assert approved.id == "run-1"


@pytest.mark.asyncio
async def test_research_runs_client_streams_sse_events(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    events = [
        {"event": "snapshot", "event_id": "1", "data": {"run": {"id": "run-1"}}},
        {"event": "terminal", "event_id": "2", "data": {"status": "completed"}},
    ]

    async def fake_sse_request(method, endpoint, params=None, headers=None):
        assert method == "GET"
        assert endpoint == "/api/v1/research/runs/run-1/events/stream"
        assert params == {"after_id": 3}
        for event in events:
            yield event

    monkeypatch.setattr(client, "_sse_request", fake_sse_request)

    streamed = [event async for event in client.stream_research_run_events("run-1", after_id=3)]

    assert streamed[0].event == "snapshot"
    assert streamed[1].data == {"status": "completed"}
