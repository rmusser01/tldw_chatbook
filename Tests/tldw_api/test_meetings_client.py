from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    MeetingArtifactCreate,
    MeetingFinalizeRequest,
    MeetingFinalizeResponse,
    MeetingHealthResponse,
    MeetingSessionCreate,
    MeetingSessionResponse,
    MeetingSessionStatusUpdate,
    MeetingShareRequest,
    MeetingTemplateCreate,
    TLDWAPIClient,
)


def _session(status: str = "scheduled") -> dict:
    return {
        "id": "meeting-1",
        "title": "Weekly Sync",
        "meeting_type": "standup",
        "status": status,
        "source_type": "upload",
        "language": "en",
    }


def _template() -> dict:
    return {
        "id": "template-1",
        "name": "Default",
        "scope": "personal",
        "enabled": True,
        "is_default": False,
        "version": 1,
        "schema_json": {"sections": ["summary"]},
    }


def _artifact(kind: str = "summary") -> dict:
    return {
        "id": f"artifact-{kind}",
        "session_id": "meeting-1",
        "kind": kind,
        "format": "json",
        "payload_json": {"text": "Summary"},
        "version": 1,
    }


@pytest.mark.asyncio
async def test_meetings_routes_wire_and_stream_events(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"status": "ok", "service": "meetings"},
            _session(),
            [_session()],
            _session(),
            _session(status="live"),
            _template(),
            [_template()],
            _template(),
            _artifact(),
            [_artifact()],
            {"session_id": "meeting-1", "artifacts": [_artifact(), _artifact("action_items")]},
            {"dispatch_id": 11, "session_id": "meeting-1", "integration_type": "slack", "status": "queued"},
            {"dispatch_id": 12, "session_id": "meeting-1", "integration_type": "webhook", "status": "queued"},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    async def fake_sse_request(method, endpoint, params=None, headers=None):
        yield {"type": "session.status", "session_id": "meeting-1", "data": {"status": "live"}}
        yield {"type": "stream.complete", "session_id": "meeting-1", "data": {"count": 1}}

    monkeypatch.setattr(client, "_sse_request", fake_sse_request)

    health = await client.get_meetings_health()
    created = await client.create_meeting_session(
        MeetingSessionCreate(title="Weekly Sync", meeting_type="standup", language="en")
    )
    listed = await client.list_meeting_sessions(status_filter="scheduled", limit=10, offset=2)
    loaded = await client.get_meeting_session("meeting-1")
    transitioned = await client.update_meeting_session_status(
        "meeting-1",
        MeetingSessionStatusUpdate(status="live"),
    )
    template = await client.create_meeting_template(
        MeetingTemplateCreate(name="Default", schema_json={"sections": ["summary"]})
    )
    templates = await client.list_meeting_templates(scope="personal", include_disabled=True)
    loaded_template = await client.get_meeting_template("template-1")
    artifact = await client.create_meeting_artifact(
        "meeting-1",
        MeetingArtifactCreate(kind="summary", format="json", payload_json={"text": "Summary"}),
    )
    artifacts = await client.list_meeting_artifacts("meeting-1")
    finalized = await client.finalize_meeting_session(
        "meeting-1",
        MeetingFinalizeRequest(transcript_text="Transcript", include=["summary", "action_items"]),
    )
    slack = await client.share_meeting_session_to_slack(
        "meeting-1",
        MeetingShareRequest(webhook_url="https://example.com/slack", artifact_ids=["artifact-summary"]),
    )
    webhook = await client.share_meeting_session_to_webhook(
        "meeting-1",
        MeetingShareRequest(webhook_url="https://example.com/webhook"),
    )
    events = [event async for event in client.stream_meeting_session_events("meeting-1")]

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/meetings/health")
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/meetings/sessions")
    assert mocked.await_args_list[1].kwargs["json_data"] == {
        "title": "Weekly Sync",
        "meeting_type": "standup",
        "source_type": "upload",
        "language": "en",
    }
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/meetings/sessions")
    assert mocked.await_args_list[2].kwargs["params"] == {"status": "scheduled", "limit": 10, "offset": 2}
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/meetings/sessions/meeting-1")
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/meetings/sessions/meeting-1/status")
    assert mocked.await_args_list[4].kwargs["json_data"] == {"status": "live"}
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/meetings/templates")
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/meetings/templates")
    assert mocked.await_args_list[6].kwargs["params"] == {"scope": "personal", "include_disabled": "true"}
    assert mocked.await_args_list[7].args[:2] == ("GET", "/api/v1/meetings/templates/template-1")
    assert mocked.await_args_list[8].args[:2] == ("POST", "/api/v1/meetings/sessions/meeting-1/artifacts")
    assert mocked.await_args_list[9].args[:2] == ("GET", "/api/v1/meetings/sessions/meeting-1/artifacts")
    assert mocked.await_args_list[10].args[:2] == ("POST", "/api/v1/meetings/sessions/meeting-1/commit")
    assert mocked.await_args_list[11].args[:2] == ("POST", "/api/v1/meetings/sessions/meeting-1/share/slack")
    assert mocked.await_args_list[12].args[:2] == ("POST", "/api/v1/meetings/sessions/meeting-1/share/webhook")
    assert isinstance(health, MeetingHealthResponse)
    assert isinstance(created, MeetingSessionResponse)
    assert listed[0].id == "meeting-1"
    assert loaded.id == "meeting-1"
    assert transitioned.status == "live"
    assert template.id == "template-1"
    assert templates[0].id == "template-1"
    assert loaded_template.id == "template-1"
    assert artifact.kind == "summary"
    assert artifacts[0].id == "artifact-summary"
    assert isinstance(finalized, MeetingFinalizeResponse)
    assert len(finalized.artifacts) == 2
    assert slack.integration_type == "slack"
    assert webhook.integration_type == "webhook"
    assert events[-1]["type"] == "stream.complete"
