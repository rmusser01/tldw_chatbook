import pytest

from tldw_chatbook.Meetings_Interop.meetings_scope_service import MeetingsScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeServerMeetingsService:
    def __init__(self):
        self.calls = []

    async def list_meeting_sessions(self, **kwargs):
        self.calls.append(("list_meeting_sessions", kwargs))
        return [{"id": "meeting-1", "title": "Weekly Sync", "meeting_type": "standup", "status": "scheduled", "source_type": "upload"}]

    async def create_meeting_template(self, request_data):
        self.calls.append(("create_meeting_template", request_data))
        return {"id": "template-1", "name": "Default", "scope": "personal", "schema_json": {"sections": ["summary"]}}

    async def list_meeting_artifacts(self, session_id):
        self.calls.append(("list_meeting_artifacts", session_id))
        return [{"id": "artifact-summary", "session_id": session_id, "kind": "summary", "format": "json", "payload_json": {"text": "Summary"}}]

    async def stream_meeting_session_events(self, session_id):
        self.calls.append(("stream_meeting_session_events", session_id))
        yield {"type": "session.status", "session_id": session_id}


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
async def test_meetings_scope_service_routes_server_and_normalizes_records():
    server = FakeServerMeetingsService()
    policy = FakePolicyEnforcer()
    scope = MeetingsScopeService(server_service=server, policy_enforcer=policy)

    sessions = await scope.list_meeting_sessions(mode="server", status_filter="scheduled")
    template = await scope.create_meeting_template(
        mode="server",
        request_data={"name": "Default", "schema_json": {"sections": ["summary"]}},
    )
    artifacts = await scope.list_meeting_artifacts("meeting-1", mode="server")
    events = [event async for event in scope.stream_meeting_session_events("meeting-1", mode="server")]

    assert sessions[0]["record_id"] == "server:meeting_session:meeting-1"
    assert template["record_id"] == "server:meeting_template:template-1"
    assert artifacts[0]["record_id"] == "server:meeting_artifact:artifact-summary"
    assert events[0]["backend"] == "server"
    assert server.calls == [
        ("list_meeting_sessions", {"status_filter": "scheduled"}),
        ("create_meeting_template", {"name": "Default", "schema_json": {"sections": ["summary"]}}),
        ("list_meeting_artifacts", "meeting-1"),
        ("stream_meeting_session_events", "meeting-1"),
    ]
    assert policy.calls == [
        "meetings.sessions.list.server",
        "meetings.templates.create.server",
        "meetings.artifacts.list.server",
        "meetings.events.observe.server",
    ]


@pytest.mark.asyncio
async def test_meetings_scope_service_rejects_local_mode_without_dispatch():
    server = FakeServerMeetingsService()
    scope = MeetingsScopeService(server_service=server)

    with pytest.raises(ValueError, match="server-only"):
        await scope.list_meeting_sessions(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_meetings_scope_service_blocks_denied_action_before_dispatch():
    server = FakeServerMeetingsService()
    scope = MeetingsScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer("authority_denied"))

    with pytest.raises(PolicyDeniedError):
        await scope.list_meeting_sessions(mode="server")

    assert server.calls == []


def test_meetings_scope_service_reports_local_and_server_contract_gaps():
    scope = MeetingsScopeService(server_service=None)

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "meetings.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Server meeting sessions, templates, artifacts, sharing, finalization, and event streams are unavailable in local/offline mode.",
            "affected_action_ids": [],
        }
    ]
    assert server_report == [
        {
            "operation_id": "meetings.websocket_live_ingest.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_followup",
            "user_message": "REST meeting CRUD/finalization/sharing and SSE event observation are available; websocket live transcript ingestion remains follow-on.",
            "affected_action_ids": [],
        }
    ]
