"""Tests for deep research run endpoint wiring on the shared TLDW API client."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ResearchArtifactResponse,
    ResearchCheckpointPatchApproveRequest,
    ResearchRunCreateRequest,
    ResearchRunListItemResponse,
    ResearchRunResponse,
    TLDWAPIClient,
)


def _run_payload(*, run_id: str = "rs_1", status: str = "running") -> dict:
    return {
        "id": run_id,
        "status": status,
        "phase": "collecting",
        "control_state": "running",
        "progress_percent": 42.5,
        "progress_message": "Collecting sources",
        "active_job_id": "job-1",
        "latest_checkpoint_id": None,
        "completed_at": None,
        "chat_id": "chat-1",
    }


@pytest.mark.asyncio
async def test_create_research_run_wires_payload_and_returns_typed_response():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(return_value=_run_payload(run_id="rs_2"))

    response = await client.create_research_run(
        ResearchRunCreateRequest(
            query="What changed in open model evals?",
            source_policy="balanced",
            autonomy_mode="checkpointed",
            limits_json={"max_sources": 10},
            provider_overrides={"web": {"enabled": True}},
        )
    )

    assert isinstance(response, ResearchRunResponse)
    assert response.id == "rs_2"
    client._request.assert_awaited_once_with(
        "POST",
        "/api/v1/research/runs",
        json_data={
            "query": "What changed in open model evals?",
            "source_policy": "balanced",
            "autonomy_mode": "checkpointed",
            "limits_json": {"max_sources": 10},
            "provider_overrides": {"web": {"enabled": True}},
            "chat_handoff": None,
            "follow_up": None,
        },
    )


@pytest.mark.asyncio
async def test_list_research_runs_returns_typed_items():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(
        return_value=[
            {
                **_run_payload(run_id="rs_3", status="completed"),
                "query": "Summarize vector DB tradeoffs",
                "created_at": "2026-04-22T12:00:00Z",
                "updated_at": "2026-04-22T12:05:00Z",
            }
        ]
    )

    response = await client.list_research_runs(limit=10)

    assert len(response) == 1
    assert isinstance(response[0], ResearchRunListItemResponse)
    assert response[0].query == "Summarize vector DB tradeoffs"
    client._request.assert_awaited_once_with(
        "GET",
        "/api/v1/research/runs",
        params={"limit": 10},
    )


@pytest.mark.asyncio
async def test_research_run_control_methods_use_server_paths():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(return_value=_run_payload(run_id="rs_4"))

    await client.pause_research_run("rs_4")
    await client.resume_research_run("rs_4")
    await client.cancel_research_run("rs_4")

    assert [call.args for call in client._request.await_args_list] == [
        ("POST", "/api/v1/research/runs/rs_4/pause"),
        ("POST", "/api/v1/research/runs/rs_4/resume"),
        ("POST", "/api/v1/research/runs/rs_4/cancel"),
    ]


@pytest.mark.asyncio
async def test_research_artifact_bundle_and_checkpoint_methods_use_server_paths():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(
        side_effect=[
            {"artifact_name": "report_v1.md", "content_type": "text/markdown", "content": "# Report"},
            {"report": "# Report"},
            _run_payload(run_id="rs_5"),
        ]
    )

    artifact = await client.get_research_artifact("rs_5", "report_v1.md")
    bundle = await client.get_research_bundle("rs_5")
    approved = await client.patch_and_approve_research_checkpoint(
        "rs_5",
        "checkpoint-1",
        ResearchCheckpointPatchApproveRequest(patch_payload={"approved": True}),
    )

    assert isinstance(artifact, ResearchArtifactResponse)
    assert bundle == {"report": "# Report"}
    assert isinstance(approved, ResearchRunResponse)
    assert [call.args for call in client._request.await_args_list] == [
        ("GET", "/api/v1/research/runs/rs_5/artifacts/report_v1.md"),
        ("GET", "/api/v1/research/runs/rs_5/bundle"),
        ("POST", "/api/v1/research/runs/rs_5/checkpoints/checkpoint-1/patch-and-approve"),
    ]
    assert client._request.await_args_list[2].kwargs == {
        "json_data": {"patch_payload": {"approved": True}},
    }
