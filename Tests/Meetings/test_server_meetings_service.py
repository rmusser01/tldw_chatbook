import pytest

from tldw_chatbook.Meetings_Interop.server_meetings_service import ServerMeetingsService
from tldw_chatbook.runtime_policy import PolicyDeniedError
from tldw_chatbook.tldw_api import (
    MeetingArtifactCreate,
    MeetingFinalizeRequest,
    MeetingSessionCreate,
    MeetingSessionStatusUpdate,
    MeetingShareRequest,
    MeetingTemplateCreate,
)


class FakeMeetingsClient:
    def __init__(self):
        self.calls = []

    async def get_meetings_health(self):
        self.calls.append(("get_meetings_health",))
        return {"status": "ok", "service": "meetings"}

    async def create_meeting_session(self, request_data):
        self.calls.append(("create_meeting_session", request_data))
        return {"id": "meeting-1", "title": request_data.title, "meeting_type": request_data.meeting_type, "status": "scheduled", "source_type": "upload"}

    async def list_meeting_sessions(self, **kwargs):
        self.calls.append(("list_meeting_sessions", kwargs))
        return [{"id": "meeting-1", "title": "Weekly Sync", "meeting_type": "standup", "status": "scheduled", "source_type": "upload"}]

    async def get_meeting_session(self, session_id):
        self.calls.append(("get_meeting_session", session_id))
        return {"id": session_id, "title": "Weekly Sync", "meeting_type": "standup", "status": "scheduled", "source_type": "upload"}

    async def update_meeting_session_status(self, session_id, request_data):
        self.calls.append(("update_meeting_session_status", session_id, request_data))
        return {"id": session_id, "title": "Weekly Sync", "meeting_type": "standup", "status": request_data.status, "source_type": "upload"}

    async def create_meeting_template(self, request_data):
        self.calls.append(("create_meeting_template", request_data))
        return {"id": "template-1", "name": request_data.name, "scope": request_data.scope, "schema_json": request_data.template_schema}

    async def list_meeting_templates(self, **kwargs):
        self.calls.append(("list_meeting_templates", kwargs))
        return [{"id": "template-1", "name": "Default", "scope": "personal", "schema_json": {"sections": ["summary"]}}]

    async def get_meeting_template(self, template_id):
        self.calls.append(("get_meeting_template", template_id))
        return {"id": template_id, "name": "Default", "scope": "personal", "schema_json": {"sections": ["summary"]}}

    async def create_meeting_artifact(self, session_id, request_data):
        self.calls.append(("create_meeting_artifact", session_id, request_data))
        return {"id": "artifact-summary", "session_id": session_id, "kind": request_data.kind, "format": request_data.format, "payload_json": request_data.payload_json}

    async def list_meeting_artifacts(self, session_id):
        self.calls.append(("list_meeting_artifacts", session_id))
        return [{"id": "artifact-summary", "session_id": session_id, "kind": "summary", "format": "json", "payload_json": {"text": "Summary"}}]

    async def finalize_meeting_session(self, session_id, request_data):
        self.calls.append(("finalize_meeting_session", session_id, request_data))
        return {
            "session_id": session_id,
            "artifacts": [
                {"id": "artifact-summary", "session_id": session_id, "kind": "summary", "format": "json", "payload_json": {"text": "Summary"}}
            ],
        }

    async def share_meeting_session_to_slack(self, session_id, request_data):
        self.calls.append(("share_meeting_session_to_slack", session_id, request_data))
        return {"dispatch_id": 11, "session_id": session_id, "integration_type": "slack", "status": "queued"}

    async def share_meeting_session_to_webhook(self, session_id, request_data):
        self.calls.append(("share_meeting_session_to_webhook", session_id, request_data))
        return {"dispatch_id": 12, "session_id": session_id, "integration_type": "webhook", "status": "queued"}

    async def stream_meeting_session_events(self, session_id):
        self.calls.append(("stream_meeting_session_events", session_id))
        yield {"type": "session.status", "session_id": session_id, "data": {"status": "live"}}
        yield {"type": "stream.complete", "session_id": session_id}


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
async def test_server_meetings_service_routes_sessions_templates_artifacts_sharing_and_events():
    client = FakeMeetingsClient()
    policy = FakePolicyEnforcer()
    service = ServerMeetingsService(client, policy_enforcer=policy)

    health = await service.get_meetings_health()
    created = await service.create_meeting_session(
        MeetingSessionCreate(title="Weekly Sync", meeting_type="standup", language="en")
    )
    sessions = await service.list_meeting_sessions(status_filter="scheduled")
    session = await service.get_meeting_session("meeting-1")
    transitioned = await service.update_meeting_session_status(
        "meeting-1",
        MeetingSessionStatusUpdate(status="live"),
    )
    template = await service.create_meeting_template(
        MeetingTemplateCreate(name="Default", schema_json={"sections": ["summary"]})
    )
    templates = await service.list_meeting_templates(scope="personal", include_disabled=True)
    loaded_template = await service.get_meeting_template("template-1")
    artifact = await service.create_meeting_artifact(
        "meeting-1",
        MeetingArtifactCreate(kind="summary", format="json", payload_json={"text": "Summary"}),
    )
    artifacts = await service.list_meeting_artifacts("meeting-1")
    finalized = await service.finalize_meeting_session(
        "meeting-1",
        MeetingFinalizeRequest(transcript_text="Transcript", include=["summary"]),
    )
    slack = await service.share_meeting_session_to_slack(
        "meeting-1",
        MeetingShareRequest(webhook_url="https://example.test/slack"),
    )
    webhook = await service.share_meeting_session_to_webhook(
        "meeting-1",
        MeetingShareRequest(webhook_url="https://example.test/webhook"),
    )
    events = [event async for event in service.stream_meeting_session_events("meeting-1")]

    assert health["record_id"] == "server:meetings:health"
    assert created["record_id"] == "server:meeting_session:meeting-1"
    assert sessions[0]["record_id"] == "server:meeting_session:meeting-1"
    assert session["record_id"] == "server:meeting_session:meeting-1"
    assert transitioned["status"] == "live"
    assert template["record_id"] == "server:meeting_template:template-1"
    assert templates[0]["record_id"] == "server:meeting_template:template-1"
    assert loaded_template["record_id"] == "server:meeting_template:template-1"
    assert artifact["record_id"] == "server:meeting_artifact:artifact-summary"
    assert artifacts[0]["record_id"] == "server:meeting_artifact:artifact-summary"
    assert finalized["record_id"] == "server:meeting_session:meeting-1"
    assert finalized["artifacts"][0]["record_id"] == "server:meeting_artifact:artifact-summary"
    assert slack["record_id"] == "server:meeting_share:11"
    assert webhook["record_id"] == "server:meeting_share:12"
    assert events[-1]["type"] == "stream.complete"
    assert policy.calls == [
        "meetings.health.detail.server",
        "meetings.sessions.create.server",
        "meetings.sessions.list.server",
        "meetings.sessions.detail.server",
        "meetings.sessions.update.server",
        "meetings.templates.create.server",
        "meetings.templates.list.server",
        "meetings.templates.detail.server",
        "meetings.artifacts.create.server",
        "meetings.artifacts.list.server",
        "meetings.sessions.launch.server",
        "meetings.share.launch.server",
        "meetings.share.launch.server",
        "meetings.events.observe.server",
    ]


@pytest.mark.asyncio
async def test_server_meetings_service_denies_before_dispatch():
    client = FakeMeetingsClient()
    service = ServerMeetingsService(client, policy_enforcer=FakePolicyEnforcer("authority_denied"))

    with pytest.raises(PolicyDeniedError):
        await service.list_meeting_sessions()

    assert client.calls == []
