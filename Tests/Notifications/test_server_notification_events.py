from __future__ import annotations

import pytest

from tldw_chatbook.Notifications import EventStateRepository
from tldw_chatbook.Notifications.server_notification_events import (
    ServerNotificationEventObserver,
    build_server_notification_feed,
    normalize_server_notification_event,
)


class FakeServerNotificationsService:
    def __init__(self, events):
        self.events = list(events)
        self.calls = []

    async def observe_feed(self, **kwargs):
        self.calls.append(kwargs)
        for event in self.events:
            yield event


def test_normalize_server_notification_event_includes_principal_scope_and_payload_hash():
    event = normalize_server_notification_event(
        {
            "event": "notification.created",
            "event_id": "evt-8",
            "data": {"id": 8, "title": "Observed", "created_at": "2026-04-29T01:02:03Z"},
        },
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_instance_id="workspace-1",
        received_at="2026-04-29T01:02:04Z",
    )

    assert event.source_authority == "server"
    assert event.server_profile_id == "server-a"
    assert event.authenticated_principal_id == "user-a"
    assert event.stream_name == "notifications"
    assert event.stream_instance_id == "workspace-1"
    assert event.event_kind == "notification.created"
    assert event.entity_ref == {"type": "notification", "id": 8}
    assert event.event_id == "evt-8"
    assert event.server_cursor == "evt-8"
    assert event.emitted_at == "2026-04-29T01:02:03Z"
    assert event.received_at == "2026-04-29T01:02:04Z"
    assert event.transport_type == "sse"
    assert event.payload_kind == "notification"
    assert event.payload_hash


@pytest.mark.asyncio
async def test_server_notification_observer_persists_actual_server_stream_events(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    service = FakeServerNotificationsService(
        [
            {
                "event": "notification.created",
                "event_id": "evt-8",
                "data": {"id": 8, "title": "Observed"},
            }
        ]
    )
    observer = ServerNotificationEventObserver(
        service=service,
        event_state_repository=repo,
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_instance_id="workspace-1",
    )

    result = await observer.observe(max_events=1)

    assert result.handled_events == 1
    assert service.calls == [{"after": 0, "last_event_id": None}]
    assert repo.get_processed_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "evt-8"
    rows = repo.list_events(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        limit=10,
    )
    assert [row["event_id"] for row in rows] == ["evt-8"]
    assert repo.get_observer_status(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    )["status"] == "idle"


@pytest.mark.asyncio
async def test_server_notification_observer_resumes_with_durable_cursor_and_dedupes(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    service = FakeServerNotificationsService(
        [
            {
                "event": "notification.created",
                "event_id": "evt-8",
                "data": {"id": 8, "title": "Observed"},
            }
        ]
    )
    observer = ServerNotificationEventObserver(
        service=service,
        event_state_repository=repo,
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_instance_id="workspace-1",
    )

    await observer.observe(max_events=1)
    replay = await observer.observe(max_events=1)

    assert replay.handled_events == 0
    assert service.calls == [
        {"after": 0, "last_event_id": None},
        {"after": 0, "last_event_id": "evt-8"},
    ]
    assert len(repo.list_events(limit=10)) == 1


def test_server_notification_feed_projects_durable_events_without_parallel_local_store(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    repo.record_event_and_advance_processed_cursor(
        normalize_server_notification_event(
            {
                "event": "notification.created",
                "event_id": "evt-8",
                "data": {"id": 8, "title": "Observed", "message": "Body"},
            },
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            stream_instance_id="workspace-1",
            received_at="2026-04-29T01:02:04Z",
        )
    )

    feed = build_server_notification_feed(
        repo,
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_instance_id="workspace-1",
        mark_presented=True,
    )

    assert feed["backend"] == "server"
    assert feed["source"] == "event_state_repository"
    assert feed["total"] == 1
    assert feed["items"][0]["record_id"] == "server:notification:8"
    assert feed["items"][0]["event_key"].startswith("server:server-a:user-a:notifications:workspace-1")
    assert feed["items"][0]["source_event_id"] == "evt-8"
    assert feed["items"][0]["title"] == "Observed"
    assert repo.get_presented_high_water(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
    ).cursor == "evt-8"


def test_server_notification_feed_reports_retention_gap_metadata(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    for event_id in ("1", "2", "3"):
        repo.record_event_and_advance_processed_cursor(
            normalize_server_notification_event(
                {
                    "event": "notification.created",
                    "event_id": event_id,
                    "data": {"id": event_id, "title": f"Observed {event_id}"},
                },
                server_profile_id="server-a",
                authenticated_principal_id="user-a",
                stream_instance_id="workspace-1",
            )
        )
    repo.prune_stream_state(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_name="notifications",
        stream_instance_id="workspace-1",
        max_count=2,
    )

    feed = build_server_notification_feed(
        repo,
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        stream_instance_id="workspace-1",
        after_cursor="1",
    )

    assert feed["replay"]["state"] == "retention_gap"
    assert feed["replay"]["server_refetch_required"] is True
    assert feed["replay"]["last_pruned_cursor"] == "1"
    assert [item["source_event_id"] for item in feed["items"]] == ["2", "3"]


def test_server_notification_feed_filters_null_principal_scope(tmp_path):
    repo = EventStateRepository(tmp_path / "events.db")
    repo.record_event_and_advance_processed_cursor(
        normalize_server_notification_event(
            {"event": "notification.created", "event_id": "evt-none", "data": {"id": 8}},
            server_profile_id="server-a",
            authenticated_principal_id=None,
            stream_instance_id="workspace-1",
        )
    )
    repo.record_event_and_advance_processed_cursor(
        normalize_server_notification_event(
            {"event": "notification.created", "event_id": "evt-user", "data": {"id": 9}},
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            stream_instance_id="workspace-1",
        )
    )

    feed = build_server_notification_feed(
        repo,
        server_profile_id="server-a",
        authenticated_principal_id=None,
        stream_instance_id="workspace-1",
    )

    assert [item["source_event_id"] for item in feed["items"]] == ["evt-none"]
